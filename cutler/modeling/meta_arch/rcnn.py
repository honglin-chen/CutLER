# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/rcnn.py

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import move_device_like
from detectron2.structures import ImageList, Instances, BoxMode
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY
from .bbnet_teacher import EvalBBNet
import torch.nn.functional as F
from torchvision.ops import masks_to_boxes
from data import detection_utils as utils
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BitMasks
import bbnet.models.teachers as teachers
import sys
import time
import os
sys.path.append('../maskcut')
from maskcut import get_affinity_matrix, second_smallest_eigenvector, get_salient_areas, check_num_fg_corners
import matplotlib.pyplot as plt

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        teacher_type: Optional[str] = None,
        single_mask: Optional[bool] = False,
        downsize_mask: Optional[bool] = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads
        self.teacher_type = teacher_type
        self.single_mask = single_mask
        self.downsize_mask = downsize_mask

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        if self.teacher_type is not None and 'bbnet' in self.teacher_type:
            self.teacher_model = EvalBBNet(distributed=False, rank=0,
                                           type='bbnet_iter_binary', pos_threshold=0.75, neg_threshold=0.25)

            if "init_dino" in self.teacher_type:
                import dino

                # DINO hyperparameters
                vit_arch = 'base'
                vit_feat = 'k'
                patch_size = 8
                # DINO pre-trained model
                url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
                feat_dim = 768
                dino_backbone = dino.ViTFeat(url, feat_dim, vit_arch, vit_feat, patch_size)
                self.dino_backbone = dino_backbone.eval().requires_grad_(False)
            elif "init_error" in self.teacher_type:
                import bbnet.models.raft_core.raft as raft
                raft_load_path = teachers.get_load_path(
                    os.path.join(
                        '/ccn2/u/honglinc/dbear/checkpoints/',
                        'raft_value_k400_ns16nt2_bs192_0'
                    ),
                    model_checkpoint=-1
                )
                self.value_raft = raft.load_raft_model(
                    load_path=raft_load_path,
                    iters=24,
                    multiframe=True,
                    scale_inputs=True,
                    small=False,
                    output_dim=1
                ).eval().requires_grad_(False)

        self.zero_loss = None

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "teacher_type": cfg.MODEL.TEACHER_TYPE,
            "single_mask": cfg.MODEL.SINGLE_MASK,
            "downsize_mask": cfg.MODEL.DOWNSIZE_MASK,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if 'bbnet' in self.teacher_type: # generate teacher segments on the fly
            device = images.tensor.device
            B, _, H, W = images.tensor.shape
            teacher_x = []

            for x in batched_inputs:
                image = F.interpolate(x['image'].float().to(device).unsqueeze(0), size=224, mode='bilinear')
                teacher_x.append(image)
            teacher_x = torch.cat(teacher_x, dim=0).contiguous()

            save_path = None
            # save_path = f"/ccn2/u/honglinc/eisen_results_v2/bbnet_teacher_test_1/teacher/{batched_inputs[0]['file_name'].split('/')[-1]}"

            with torch.no_grad():

                if "init_dino" in self.teacher_type:
                    sampling_power = 4  # or whatever you want
                    init_dist = self.get_dino_predominance(teacher_x)
                    init_dist = init_dist ** sampling_power

                elif "init_error" in self.teacher_type:
                    value_map = self.value_raft(teacher_x.unsqueeze(1)).squeeze(1)
                    sampling_power = 8  # or whatever you want
                    init_dist = value_map.sigmoid() ** sampling_power
                    init_dist -= init_dist.min()
                    init_dist /= (init_dist.max() + 1e-12)
                else:
                    init_dist = None

                _, segment_target, sampling_distributions = self.teacher_model(teacher_x, teacher_x, save_path=save_path, init_dist=init_dist) # [B, 1, 224, 224]

            gt_instances = []
            segment_list = []
            for i in range(B):
                size = batched_inputs[i]['image'].shape[-2:]
                segment = F.interpolate(segment_target[i:i+1], size=size, mode='bilinear') > 0.5
                segment = segment.float()
                segment_list.append(segment[0])

            segments = ImageList.from_tensors(
                segment_list,
                self.backbone.size_divisibility,
                padding_constraints=self.backbone.padding_constraints,
            )

            for i in range(B):
                segment = segments.tensor[i]
                if segment.sum() == 0:
                    instances = utils.annotations_to_instances([], image_size=[H, W], mask_format='bitmask')
                else:
                    annotations = dict()
                    annotations['bbox'] = masks_to_boxes(segment).cpu()[0]
                    annotations['bbox_mode'] = BoxMode.XYXY_ABS
                    annotations['segmentation'] = segment[0].cpu().detach().numpy()
                    annotations['category_id'] = torch.tensor(0)
                    instances = utils.annotations_to_instances([annotations], image_size=[H, W], mask_format='bitmask')

                    # Visualization
                    # self.visualize(images, instances, batched_inputs, sampling_distributions, i)
                    # End of visualization

                gt_instances.append(instances.to(device))


        elif "instances" in batched_inputs[0]:

            if self.single_mask:
                gt_instances = [x["instances"][0:1].to(self.device) for x in batched_inputs]

                if self.downsize_mask:
                    for i in range(len(gt_instances)):
                        mask = gt_instances[i][0].gt_masks.tensor
                        _, h, w = mask.shape
                        mask = F.interpolate(mask.unsqueeze(0).float(), size=224, mode='bilinear')
                        mask = F.interpolate(mask.float(), size=[h, w], mode='bilinear') > 0.5
                        instance_clone = gt_instances[i][0]
                        instance_clone.set('gt_masks', BitMasks(mask[0]))
                        gt_instances[i] = instance_clone


            else:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator is not None:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses

    def get_dino_predominance(self, images):

        input_dino = images / 255.
        input_dino = input_dino - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(input_dino.device)
        input_dino = input_dino / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(input_dino.device)
        # input_dino = images.tensor
        input_dino = F.interpolate(input_dino, size=224, mode='bilinear')
        feats = self.dino_backbone(input_dino)

        dims = [28, 28]
        predominence_map = []
        for i in range(images.shape[0]):
            A, D = get_affinity_matrix(feats[i], tau=0.15)
            # get the second-smallest eigenvector
            _, second_smallest_vec = second_smallest_eigenvector(A, D)
            # get salient area
            bipartition = get_salient_areas(second_smallest_vec)

            # check if we should reverse the partition based on:
            # 1) peak of the 2nd smallest eigvec 2) object centric bias
            seed = np.argmax(np.abs(second_smallest_vec))
            nc = check_num_fg_corners(bipartition, dims)
            if nc >= 3:
                reverse = True
            else:
                reverse = bipartition[seed] != 1
            if reverse:
                second_smallest_vec = 1 - second_smallest_vec
            second_smallest_vec = torch.tensor(second_smallest_vec).to(self.device).contiguous()
            map = F.interpolate(second_smallest_vec.reshape(1, 1, dims[0], dims[1]), size=224, mode='bilinear')
            map -= map.min()
            map /= map.max()
            predominence_map.append(map)
        init_dist = torch.cat(predominence_map, dim=0).detach()
        return init_dist

    def visualize(self, images, instances, batched_inputs, sampling_distributions, i):
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        original_image = images.tensor * self.pixel_std[None] + self.pixel_mean[None]
        original_h, original_w = batched_inputs[i]['image'].shape[-2:]
        original_image = original_image[:, :, :original_h, :original_w]
        axs[0].imshow(original_image[i].permute(1, 2, 0).cpu().numpy() / 255.)
        axs[0].set_title('Input Image')
        axs[0].set_axis_off()
        visualizer = Visualizer(original_image[i].permute(1, 2, 0).cpu().numpy(), metadata=None)
        vis = visualizer.overlay_instances(boxes=instances.gt_boxes.tensor,
                                           masks=instances.gt_masks.tensor[:, :original_h, :original_w])

        axs[1].imshow(vis.get_image())
        axs[1].set_title('Ours')
        axs[1].set_axis_off()

        visualizer = Visualizer(original_image[i].permute(1, 2, 0).cpu().numpy(), metadata=None)

        vis = visualizer.overlay_instances(boxes=batched_inputs[i]['instances'].gt_boxes.tensor,
                                           masks=batched_inputs[i]['instances'].gt_masks.tensor)

        axs[2].imshow(vis.get_image())
        axs[2].set_title('CutLER')
        axs[2].set_axis_off()


        if sampling_distributions is not None:
            pmap = F.interpolate(sampling_distributions[i][None], size=[original_h, original_w], mode='bilinear')#[:, :, :original_h, :original_w]
            axs[3].imshow(pmap[0, 0].cpu().numpy(), cmap='magma')
            axs[3].set_title('Sampling distributions')
            axs[3].set_axis_off()

        if 'init_dino' in self.teacher_type:
            save_path = f"/ccn2/u/honglinc/eisen_results_v2/bbnet_teacher_predominence/{batched_inputs[i]['file_name'].split('/')[-1]}"
        elif 'init_error' in self.teacher_type:
            save_path = f"/ccn2/u/honglinc/eisen_results_v2/bbnet_teacher_value_map/{batched_inputs[i]['file_name'].split('/')[-1]}"
        else:
            save_path = f"/ccn2/u/honglinc/eisen_results_v2/bbnet_teacher_none/{batched_inputs[i]['file_name'].split('/')[-1]}"

        print("Saving to {} ...".format(save_path))
        plt.savefig(save_path, bbox_inches='tight')

    # def visualize_distribution(self, images, instances, batched_inputs, sampling_distributions, i):
    #     fig, axs = plt.subplots(3, 4, figsize=(20, 15))
    #     original_image = images.tensor * self.pixel_std[None] + self.pixel_mean[None]
    #     original_h, original_w = batched_inputs[i]['image'].shape[-2:]
    #     original_image = original_image[:, :, :original_h, :original_w]
    #
    #     breakpoint()
    #     title = ['With predomiance', 'With value map', 'None']
    #     for i in range(3):
    #         axs[i, 0].imshow(original_image[i].permute(1, 2, 0).cpu().numpy() / 255.)
    #         axs[i, 0].set_title('Input Image')
    #         axs[i, 0].set_xtitle(title[i])
    #
    #     visualizer = Visualizer(original_image[i].permute(1, 2, 0).cpu().numpy(), metadata=None)
    #     vis = visualizer.overlay_instances(boxes=instances.gt_boxes.tensor,
    #                                        masks=instances.gt_masks.tensor[:, :original_h, :original_w])
    #
    #     axs[1].imshow(vis.get_image())
    #     axs[1].set_title('Ours')
    #     axs[1].set_axis_off()
    #
    #     visualizer = Visualizer(original_image[i].permute(1, 2, 0).cpu().numpy(), metadata=None)
    #
    #     vis = visualizer.overlay_instances(boxes=batched_inputs[i]['instances'].gt_boxes.tensor,
    #                                        masks=batched_inputs[i]['instances'].gt_masks.tensor)
    #
    #     axs[2].imshow(vis.get_image())
    #     axs[2].set_title('CutLER')
    #     axs[2].set_axis_off()
    #
    #
    #     if sampling_distributions is not None:
    #         pmap = F.interpolate(sampling_distributions[i][None], size=images.tensor.shape[-2:], mode='bilinear')[:, :, :original_h, :original_w]
    #         axs[3].imshow(pmap[0, 0].cpu().numpy(), cmap='magma')
    #         axs[3].set_title('Sampling distributions')
    #         axs[3].set_axis_off()
    #
    #     save_path = f"/ccn2/u/honglinc/eisen_results_v2/bbnet_teacher_test/{batched_inputs[i]['file_name'].split('/')[-1]}"
    #     print("Saving to {} ...".format(save_path))
    #     plt.savefig(save_path, bbox_inches='tight')

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results

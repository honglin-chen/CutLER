import sys
sys.path.append('/home/honglinc/BBNet')
from bbnet.models.teachers import default_head_motion_eisen_teacher, load_predictor, make_teacher, default_eisen_teacher_func, set_input_shape_parallel, default_model_dir, full_affinities_eisen_teacher_func, iteration_head_motion_eisen_teacher_fa
import bbnet.models.teachers as teachers
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

class EvalBBNet(nn.Module):
    def __init__(self, distributed, rank, type, pos_threshold, neg_threshold):
        super(EvalBBNet, self).__init__()

        self.ts = torch.tensor([0, 1]).view(1, 2)
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold

        self.type = type
        assert type in ['bbnet_init_dino', 'bbnet', 'bbnet_old', 'bbnet_float', 'bbnet_binary', 'bbnet_iter_binary', 'bbnet_iter_binary_gt', 'all', 'bbnet_patch_select'], type

        if type in ['bbnet_old', 'all']:
            phi_2frame = load_predictor(model_dir=default_model_dir)  # defaults
            full_eisen_teacher = make_teacher(
                predictor=phi_2frame.to(rank),
                teacher_func=full_affinities_eisen_teacher_func
            ).to(rank)

            if distributed:
                full_eisen_teacher.dummy = torch.nn.Parameter(torch.tensor(1.0).to(rank), requires_grad=True)
                self.full_eisen_teacher = DDP(full_eisen_teacher, device_ids=[rank], output_device=rank,
                                              find_unused_parameters=False)
            else:
                self.full_eisen_teacher = nn.DataParallel(full_eisen_teacher)
            self.full_eisen_teacher.eval().requires_grad_(False)

        if type in ['bbnet_float', 'bbnet_binary', 'all']:
            G = default_head_motion_eisen_teacher().to(rank)
            if distributed:
                G.dummy_parameter = torch.nn.Parameter(torch.tensor(1.0).to(rank), requires_grad=True)
                self.G = DDP(G, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            else:
                self.G = nn.DataParallel(G)
            self.G.eval().requires_grad_(False)

        if type in ['bbnet_init_dino', 'bbnet', 'bbnet_iter_binary', 'bbnet_iter_binary_gt', 'all']:
            self.use_float16 = True
            self.decorator = torch.cuda.amp.autocast()

            iteration_teacher = iteration_head_motion_eisen_teacher_fa()#.to(rank)
            num_raft_iters = 6
            iteration_teacher.set_raft_iters(num_raft_iters)

            if distributed:
                iteration_teacher.dummy_parameter = torch.nn.Parameter(torch.tensor(1.0).to(rank), requires_grad=True)
                self.iteration_teacher = DDP(iteration_teacher, device_ids=[rank], output_device=rank, find_unused_parameters=False)
            else:
                self.iteration_teacher = iteration_teacher.eval() # .cuda()

        if type in ['bbnet_patch_select']:
            # predictor_load_path = teachers.get_load_path(os.path.join(teachers.new_model_dir, 'baseHMC4x4_KME_mr099fmp01_IMUfmp01_ctr_bs1024_wu10_tpu0'), model_checkpoint=-1)
            from functools import partial
            flow_error_similarity_teacher = partial(
                teachers.flow_error_similarity_teacher,
                is_teacher=False)
            self.patch_selector = flow_error_similarity_teacher(
                is_teacher=False,
                model_func=teachers.error_teacher_model_func_with_fa,
                model_path='/ccn2/u/honglinc/dbear/checkpoints/baseHMC4x4_KME_mr099fmp01_IMUfmp01_ctr_bs1024_wu10_tpu0/checkpoint-210.pth'
            ).requires_grad_(False).cuda()


    def forward(self, image_1, image_2, ts=None, pos_threshold=0.9, neg_threshold=0.1, save_path=None, num_init_points=None, gt_segment=None, init_dist=None):
        B, _, H, W = image_1.shape

        # Input should have value in range 【0, 255】.
        x = torch.cat([image_1.unsqueeze(1), image_2.unsqueeze(1)], dim=1) / 255.
        ts = self.ts.expand(B, -1).to(x) if ts is None else ts
        # assert x.min() >= 0. and x.max() <= 1.0, (x.min(), x.max())
        target_points = None

        ## Old target
        if self.type in ['bbnet_old', 'all']:

            set_input_shape_parallel(self.full_eisen_teacher, x, T=x.shape[1])
            old_targets = self.full_eisen_teacher(
                x,
                target_points=[[100, 100], [150, 150], [200, 200], [80, 80]],
                init_points_are_targets=True,
                get_full_affinities=False,
                get_target_affinities=False,
                **self.full_eisen_teacher.module.full_affinities_run_kwargs)

        ## New target
        if self.type in ['bbnet_float', 'bbnet_binary', 'all']:
            targets = self.G(x, timestamps=ts, sample_batch_size=10, num_target_points=8)
            target_points = self.G.module.target_points
            del self.G.module.corrs, self.G.module.flow_samples

        if self.type in ['bbnet_init_dino', 'bbnet', 'bbnet_iter_binary', 'bbnet_iter_binary_gt', 'all']:
            ## Iteration target

            with self.decorator:
                targets, _, _ = self.iteration_teacher(x.to(torch.float16 if self.use_float16 else torch.float32), init_dist=init_dist, sample_batch_size=None)
            sampling_distribution = self.iteration_teacher.sampling_distribution
            # targets, _, _ = self.iteration_teacher(x, timestamps=ts, sample_batch_size=10, num_target_points=1)

        if self.type in ['bbnet_patch_select']:
            start = time.time()
            with torch.cuda.amp.autocast(enabled=True):
                pos_mask, neg_mask, targets = self.patch_selector(x.to(torch.float16))
            print('time for patch selection', time.time() - start)

            sampling_distribution = self.patch_selector.sampling_distribution

        if gt_segment is not None:
            assert B == 1
            gt_segment = gt_segment.to(targets.device)
            gt_segment = gt_segment.unique().view(1, -1, 1, 1) == gt_segment
            intersection = ((targets * gt_segment) > 0).sum(dim=(2, 3))
            union = ((targets + gt_segment) > 0).sum(dim=(2, 3))
            iou = intersection / (union + 1e-12)
            idx = iou.argmax(dim=1)
            targets = gt_segment[:, idx]

        ptargets = targets > self.pos_threshold
        ntargets = targets < self.neg_threshold


        if save_path is not None:
            # Visualization code
            K = targets.shape[1]
            fig, axs = plt.subplots(K, 5, figsize=(12, K*4))
            if K == 1:
                axs = axs[None]
            for i in range(axs.shape[0]):
                for j in range(axs.shape[1]):
                    axs[i, j].tick_params(axis='both', which='major', labelsize=5)
                    axs[i, j].tick_params(axis='both', which='minor', labelsize=4)
            fontsize = 11

            axs[0, 0].imshow(image_1[0].permute(1, 2, 0).cpu() / 255.)
            axs[0, 0].set_title('Input image', fontsize=fontsize)
            axs[0, 0].set_axis_off()
            axs[0, 1].set_axis_off()
            axs[0, 2].set_axis_off()
            axs[0, 3].set_axis_off()
            for i in range(0, K):
                im1 = axs[i, 1].imshow(targets[0, i].cpu())
                axs[i, 1].set_title('Target map', fontsize=fontsize)
                divider = make_axes_locatable(axs[i, 1])
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig.colorbar(im1, cax=cax, orientation='vertical')

                axs[i, 2].imshow(ptargets[0, i].cpu())
                axs[i, 2].set_title(f'Pos Target (>{self.pos_threshold})', fontsize=fontsize)
                axs[i, 3].imshow(ntargets[0, i].cpu())
                axs[i, 3].set_title(f'Neg Target (<{self.neg_threshold})', fontsize=fontsize)
                axs[i, 4].imshow((ptargets[0, i].cpu()[None] * image_1[0].cpu()).permute(1, 2, 0) / 255)
                axs[i, 4].set_title('Masked segment', fontsize=fontsize)
                axs[i, 0].set_axis_off()
                axs[i, 1].set_axis_off()
                axs[i, 2].set_axis_off()
                axs[i, 3].set_axis_off()
                axs[i, 4].set_axis_off()

            folder_path = '/'.join(save_path.split('/')[0:-1])
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            print('Save fig to', save_path)
            # plt.show()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

        '''
        if save_path is not None:
            # Visualization code
            K = targets.shape[1]
            B = targets.shape[0]
            fig, axs = plt.subplots(B, 3+1, figsize=(16, 4))

            axs = axs[None]

            for i in range(B):
                axs[i, 0].imshow(x[i, 0].permute(1, 2, 0).cpu())
                axs[i, 0].set_axis_off()
                axs[i, 0].set_title('Image')
                # axs[i+1, 0].imshow(x[i, 0].permute(1, 2, 0).cpu())
                # axs[i+1, 0].set_axis_off()
                # axs[i + 2, 0].imshow(x[i, 0].permute(1, 2, 0).cpu())
                # axs[i + 2, 0].set_axis_off()

                for k in range(K):
                    axs[i, 1].imshow((iteration_targets[i, k].unsqueeze(-1) * x[i, 0].permute(1, 2, 0)).cpu())
                    axs[i, 1].set_axis_off()
                    axs[i, 1].set_title('Iter targets')

                for k in range(K):
                    axs[i, 2].imshow((targets[i, k].unsqueeze(-1) * x[i, 0].permute(1, 2, 0)).cpu())
                    # axs[i+1, k+1].scatter(target_points[i, k, 1].cpu(), target_points[i, k, 0].cpu(), c='red', s=100)
                    axs[i, 2].set_axis_off()
                    axs[i, 2].set_title('New targets')


                for k in range(K):
                    axs[i, 3].imshow((old_targets[i, k].unsqueeze(-1) * x[i, 0].permute(1, 2, 0)).cpu())
                    axs[i, 3].set_axis_off()
                    axs[i, 3].set_title('Old targets')


            folder_path = '/'.join(save_path.split('/')[0:-1])
            if not os.path.exists(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            logger.debug('Save fig to', save_path)
            plt.savefig(save_path, bbox_inches='tight')
            plt.show()
            plt.close()
        '''

        targets > self.pos_threshold

        return target_points, targets.detach(), sampling_distribution



if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES']='0,1'
    teacher = EvalBBNet().cuda()
    video = torch.randn(2, 2, 3, 224, 224).cuda()
    teacher(video)

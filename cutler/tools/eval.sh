# Copyright (c) Meta Platforms, Inc. and affiliates.

# link to the dataset folder, model weights and the config file.
#model_weights="http://dl.fbaipublicfiles.com/cutler/checkpoints/cutler_cascade_final.pth"
#model_weights='output/model_0004999.pth'
#model_weights='/ccn2/u/honglinc/cutler_checkpoints/cutler_nocopypaste/model_0034999.pth'
#model_weights='/ccn2/u/honglinc/cutler_checkpoints/cutler_single_mask/model_0034999.pth'
model_weights='/ccn2/u/honglinc/cutler_checkpoints/bbnet_teacher_background_2corner_kinetics_static_init_dino/model_0004999.pth'
config_file="model_zoo/configs/CutLER-ImageNet/cascade_mask_rcnn_R_50_FPN_no_copy_paste.yaml"

#model_weights='/ccn2/u/honglinc/cutler_checkpoints/bbnet_teacher_0/model_0004999.pth'
#config_file="model_zoo/configs/BBNet-ImageNet/cascade_mask_rcnn_R_50_FPN.yaml"

num_gpus=8

echo "========== start evaluating the model on all 11 datasets =========="
echo $model_weights
echo $config_file
#test_dataset='cls_agnostic_clipart'
#echo "========== evaluating ${test_dataset} =========="
#python train_net.py --num-gpus ${num_gpus} \
#  --config-file ${config_file} \
#  --test-dataset ${test_dataset} --no-segm \
#  --eval-only MODEL.WEIGHTS ${model_weights}
#
#test_dataset='cls_agnostic_watercolor'
#echo "========== evaluating ${test_dataset} =========="
#python train_net.py --num-gpus ${num_gpus} \
#  --config-file ${config_file} \
#  --test-dataset ${test_dataset} --no-segm \
#  --eval-only MODEL.WEIGHTS ${model_weights}
#
#test_dataset='cls_agnostic_comic'
#echo "========== evaluating ${test_dataset} =========="
#python train_net.py --num-gpus ${num_gpus} \
#  --config-file ${config_file} \
#  --test-dataset ${test_dataset} --no-segm \
#  --eval-only MODEL.WEIGHTS ${model_weights}
#
#test_dataset='cls_agnostic_voc'
#echo "========== evaluating ${test_dataset} =========="
#python train_net.py --num-gpus ${num_gpus} \
#  --config-file ${config_file} \
#  --test-dataset ${test_dataset} --no-segm \
#  --eval-only MODEL.WEIGHTS ${model_weights}
#
#test_dataset='cls_agnostic_objects365'
#echo "========== evaluating ${test_dataset} =========="
#python train_net.py --num-gpus ${num_gpus} \
#  --config-file ${config_file} \
#  --test-dataset ${test_dataset} --no-segm \
#  --eval-only MODEL.WEIGHTS ${model_weights}
#
#test_dataset='cls_agnostic_openimages'
#echo "========== evaluating ${test_dataset} =========="
#python train_net.py --num-gpus ${num_gpus} \
#  --config-file ${config_file} \
#  --test-dataset ${test_dataset} --no-segm \
#  --eval-only MODEL.WEIGHTS ${model_weights}
#
#test_dataset='cls_agnostic_kitti'
#echo "========== evaluating ${test_dataset} =========="
#python train_net.py --num-gpus ${num_gpus} \
#  --config-file ${config_file} \
#  --test-dataset ${test_dataset} --no-segm \
#  --eval-only MODEL.WEIGHTS ${model_weights}
#
#
test_dataset='cls_agnostic_coco'
echo "========== evaluating ${test_dataset} =========="
python train_net.py --num-gpus ${num_gpus} \
  --config-file ${config_file} \
  --test-dataset ${test_dataset} \
  --eval-only MODEL.WEIGHTS ${model_weights}


#test_dataset='cls_agnostic_coco20k'
#echo "========== evaluating ${test_dataset} =========="
#python train_net.py --num-gpus ${num_gpus} \
#  --config-file ${config_file} \
#  --test-dataset ${test_dataset} \
#  --eval-only MODEL.WEIGHTS ${model_weights}


#test_dataset='cls_agnostic_lvis'
#echo "========== evaluating ${test_dataset} =========="
## LVIS should set TEST.DETECTIONS_PER_IMAGE=300
#python train_net.py --num-gpus ${num_gpus} \
#  --config-file ${config_file} \
#  --test-dataset ${test_dataset} \
#  --eval-only MODEL.WEIGHTS ${model_weights} TEST.DETECTIONS_PER_IMAGE 300
#
#test_dataset='cls_agnostic_uvo'
#echo "========== evaluating ${test_dataset} =========="
#python train_net.py --num-gpus ${num_gpus} \
#  --config-file ${config_file} \
#  --test-dataset ${test_dataset} \
#  --eval-only MODEL.WEIGHTS ${model_weights}

echo "========== evaluation is completed =========="
# Copyright (c) Meta Platforms, Inc. and affiliates.

from detectron2.config import CfgNode as CN

def add_cutler_config(cfg):
    cfg.DATALOADER.COPY_PASTE = False
    cfg.DATALOADER.COPY_PASTE_RATE = 0.0
    cfg.DATALOADER.COPY_PASTE_MIN_RATIO = 0.5
    cfg.DATALOADER.COPY_PASTE_MAX_RATIO = 1.0
    cfg.DATALOADER.COPY_PASTE_RANDOM_NUM = True
    cfg.DATALOADER.VISUALIZE_COPY_PASTE = False

    cfg.MODEL.ROI_HEADS.USE_DROPLOSS = False
    cfg.MODEL.ROI_HEADS.DROPLOSS_IOU_THRESH = 0.0
    cfg.MODEL.TEACHER_TYPE = None
    cfg.MODEL.BBNET_TEACHER_THRESH = 0.75
    cfg.MODEL.SINGLE_MASK = False
    cfg.MODEL.DOWNSIZE_MASK = False
    cfg.MODEL.DEBUG_ONLY = False

    cfg.SOLVER.BASE_LR_MULTIPLIER = 1
    cfg.SOLVER.BASE_LR_MULTIPLIER_NAMES = []

    cfg.TEST.NO_SEGM = False
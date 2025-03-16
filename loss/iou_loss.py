import torch
from monai.losses import DiceLoss

def iou_binary(pred, target):
    iou_loss = DiceLoss(include_background=True, sigmoid=False, to_onehot_y=False)
    loss = iou_loss(pred, target)
    return loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import FocalLoss as Focal

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred, target):
        # Calcolo della cross-entropy loss
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Calcolo della focal loss
        F_loss = self.alpha * ((1 - p_t) ** self.gamma) * ce_loss
        
        # Applica la riduzione scelta
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

def return_focal_loss(pred, target, gamma=2):
    focal_loss = Focal(gamma)
    return focal_loss(pred, target)

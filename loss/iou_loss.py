import torch
from monai.losses import DiceLoss

def iou_multiclass(pred, target, smooth=1e-6):
    channels = pred.size(1)  # numero di canali

    # Applicare softmax ai logit per ottenere probabilità per classe
    #pred = torch.softmax(pred, dim=1)
    #pred_labels = torch.argmax(pred, dim=1)

    print(pred.max(), pred.min())

    iou_per_class = []
    
    for ch in range(channels):
        pred_cls = (pred == ch).float()  # Predizione binaria per la classe ch
        target_cls = (target == ch).float()     # Maschera binaria per la classe ch

        # Calcolo dell'intersezione
        intersection = torch.sum(pred_cls * target_cls)
        # Calcolo dell'unione
        union = torch.sum(pred_cls) + torch.sum(target_cls) - intersection

        iou = (intersection + smooth) / (union + smooth)
        iou_per_class.append(iou.item()) 

    return (1 - sum(iou_per_class) / channels)

def iou_binary(pred, target):
    iou_loss = DiceLoss(include_background=True, sigmoid=False, to_onehot_y=False)
    loss = iou_loss(pred, target)
    return loss

# compute ssim loss for each channel and compute mean
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import torch

def ssim(pred, label):
    device = pred.device
    ssim = StructuralSimilarityIndexMeasure(
    gaussian_kernel=True,
    sigma=1.5,
    kernel_size=11,
    reduction='elementwise_mean',
    data_range=1.0,  # Imposta il range dei dati (ad esempio 1.0 per immagini normalizzate)
    k1=0.01,
    k2=0.03,
    return_full_image=False,
    return_contrast_sensitivity=False ).to(device)

    similarity = ssim(pred, label)
    loss = 1- similarity

    return loss

def ssim_single(pred, label):
    device = pred.device
    ssim = StructuralSimilarityIndexMeasure(
        data_range=1.0,
        kernel_size=3,  # Adatta a immagini piccole
        k1=0.01,
        k2=0.03
    ).to(device)

    similarity = ssim(pred, label)
    similarity = torch.clamp(similarity, min=0.0, max=1.0)
    loss = 1 - similarity
    return loss

def ms_ssim(pred, label):
    device = pred.device
    ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(
    gaussian_kernel=True,
    kernel_size=7,
    sigma=1.5,
    reduction='elementwise_mean',
    data_range=1.0,
    k1=0.01,
    k2=0.03,
    betas=(0.0448, 0.2856, 0.3001),
    normalize='relu').to(device)

    similarity = ms_ssim(pred, label)
    loss = 1- similarity

    return loss

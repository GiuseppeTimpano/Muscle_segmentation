import os
import csv
import monai.losses
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    ScaleIntensityd,
    Resized,
    RandAxisFlipd,
    AsDiscrete,
    Zoomd,
    RandGaussianSmoothd,
    CastToTyped
)
import torch
import torch.optim as optim
from models.unet3plusParallel import UNet_3Plus_3D
from loss.focal_loss import return_focal_loss
from loss.iou_loss import iou_binary
from loss.ssim_loss import ssim_single
from monai.metrics import DiceMetric
from monai.data import DataLoader
import torch
import pickle
import random
from torch.utils.data import ConcatDataset
import numpy as np


# Dataset
with open('datasetpkl/datasetpkl3d/3d_dataset_train.pkl', 'rb') as file:
    data_train = pickle.load(file)

with open('datasetpkl/datasetpkl3d/3d_dataset_val.pkl', 'rb') as file:
    data_val = pickle.load(file)

# apply thresholds for muscle identification
img_train = []
seg_train = []
for image, label in zip(data_train['images'], data_train['annotations']):
    filtered_image = np.where((image >= -29) & (image <= 150), image, 0)
    img_train.append(filtered_image)
    seg_train.append(label)

img_val = []
seg_val = []
for image, label in zip(data_val['images'], data_val['annotations']):
    filtered_image = np.where((image >= -29) & (image <= 150), image, 0)
    img_val.append(filtered_image)
    seg_val.append(label)

train_data = [
    {"img": img[:19], "seg": ann[:19]}
    for img, ann in zip(img_train, seg_train)
]

val_data = [
    {"img": img[:19], "seg": ann[:19]}
    for img, ann in zip(img_val, seg_val)
]

# Transforms
transforms = Compose(
    [
        ScaleIntensityd(keys=["img"]),
        EnsureChannelFirstd(keys=["img", "seg"], channel_dim="no_channel"),
        Resized(keys=['img'], spatial_size=(19, 256, 256), mode='area', anti_aliasing=True),
        Resized(keys=['seg'], spatial_size=(19, 256, 256), mode='nearest'),
    ]
)

augm_transforms = Compose(
    [
        ScaleIntensityd(keys=['img']),
        EnsureChannelFirstd(keys=['img', 'seg'], channel_dim='no_channel'),
        Resized(keys=['img'], spatial_size=(19, 256, 256), mode='area', anti_aliasing=True),
        Resized(keys=['seg'], spatial_size=(19, 256, 256), mode='nearest'),
        RandAxisFlipd(keys=['img', 'seg'], prob=1),
        Zoomd(keys=['img'], prob=1, zoom=1.5, mode='area'),
        Zoomd(keys=['seg'], prob=1, zoom=1.5, mode='nearest'),
        RandGaussianSmoothd(keys=['img'], prob=1, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5), sigma_z=(0.25, 0.5)),
    ]
)

post_trans = Compose([AsDiscrete(threshold=0.5)])

# Dataset and DataLoader
train_ds = monai.data.Dataset(data=train_data, transform=transforms)
train_augm = monai.data.Dataset(data=train_data, transform=augm_transforms)
train_ds = ConcatDataset([train_ds, train_augm])
val_ds = monai.data.Dataset(data=val_data, transform = transforms)

train_loader = DataLoader(
    train_ds,
    batch_size=3,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_ds,
    batch_size=3,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)

# Loss, Metric, and Model
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

def criterion(outputs, labels, loss_param):
    device = outputs.device
    labels = labels.to(device)
    focal_loss = return_focal_loss(outputs, labels)
    dice_value = iou_binary(outputs, labels)
    ssim_value = ssim_single(outputs, labels)

    if loss_param=='dice':
         return dice_value
    elif loss_param=='focal':
         return focal_loss
    elif loss_param=='ssim':
         return ssim_value
    elif loss_param=='loss_sum':
         return focal_loss + dice_value + ssim_value

if torch.cuda.device_count()>1:
    model = UNet_3Plus_3D(in_channels=1, n_classes=1, feature_scale=4, checkpoint=False)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

# Training
val_interval = 1
best_val_dice = -1  # Modificato per chiarezza

# Crea il file CSV
with open('results_pkl/3D/train_unet3plus_log_3.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Validation Dice', 'Best Dice'])

for epoch in range(300):
    print("-" * 10)
    print(f"Epoch {epoch + 1}/300")
    model.train()
    epoch_loss = 0
    for step, batch in enumerate(train_loader, 1):
        images, labels = batch["img"], batch["seg"]
        images = images.to(next(model.parameters()).device)
        labels = labels.to(images.device)
        optimizer.zero_grad()
        output = torch.sigmoid(model(images))
        loss = criterion(output, labels, 'loss_sum')
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        torch.cuda.empty_cache()
        print(f"{step}/{len(train_loader)}, Train Loss: {loss.item():.4f}")
    epoch_loss /= len(train_loader)
    print(f"Epoch {epoch + 1} Average Train Loss: {epoch_loss:.4f}")
    
    torch.cuda.empty_cache()

    # Validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for val_data in val_loader:
                val_images, val_labels = val_data["img"], val_data["seg"]
                val_images = val_images.to(next(model.parameters()).device)
                val_labels = val_labels.to(val_images.device)
                val_output = torch.sigmoid(model(val_images))
                val_loss = criterion(val_output, val_labels, 'loss_sum')
                val_loss_sum += val_loss.item()
                dice_metric(post_trans(val_output).to(val_labels.device), val_labels)

            val_loss_sum /= len(val_loader)
            dice_value_avg = dice_metric.aggregate().item()
            dice_metric.reset()
            print(
                f"Epoch {epoch + 1}, Validation Mean Dice: {dice_value_avg:.4f}, "
                f"Validation Loss: {val_loss_sum:.4f}"
            )

            if dice_value_avg > best_val_dice:
                best_val_dice = dice_value_avg
                torch.save(model.state_dict(), 'results_pkl/3D/best_unet3plusmodel_3.pth')
                print(f"Model saved at epoch {epoch + 1} with best Dice Score {dice_value_avg:.4f}")

        torch.cuda.empty_cache()

        # Scrivi i risultati nel CSV
        with open('results_pkl/3D/train_unet3plus_log.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, epoch_loss, val_loss_sum, dice_value_avg, best_val_dice])
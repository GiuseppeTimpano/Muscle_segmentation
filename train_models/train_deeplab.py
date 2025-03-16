import os
import csv
import monai.losses
from Deeplab.deeplabv3plus import DeepLabV3plus
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    ScaleIntensityd,
    Resized,
    RandAxisFlipd,
    RandGaussianSmoothd,
    AsDiscrete
)
from monai.metrics import DiceMetric, MeanIoU
from monai.data import DataLoader
import torch
import pickle

# Dataset
with open('datasetpkl/datasetpkl2d/2d_dataset_train.pkl', 'rb') as file:
    data_train = pickle.load(file)

with open('datasetpkl/datasetpkl2d/2d_dataset_val.pkl', 'rb') as file:
    data_val = pickle.load(file)


train_data = [
    {"img": img, "seg": ann}
    for img, ann in zip(data_train['images'], data_train['annotations'])
]

val_data = [
    {"img": img, "seg": ann}
    for img, ann in zip(data_val['images'], data_val['annotations'])
]

# Transforms
train_transforms = Compose(
    [
        ScaleIntensityd(keys=["img"]),
        EnsureChannelFirstd(keys=["img", "seg"], channel_dim="no_channel"),
    ]
)

augm_transforms = Compose(
    [
        ScaleIntensityd(keys=["img"]),
        EnsureChannelFirstd(keys=["img", "seg"], channel_dim="no_channel"),
        RandAxisFlipd(keys=['img', 'seg'], prob=0.5),
        RandGaussianSmoothd(keys=['img'], prob=0.5, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5))
    ]
)
augm_transforms_resize = Compose(
    [
        ScaleIntensityd(keys=["img"]),
        EnsureChannelFirstd(keys=["img", "seg"], channel_dim="no_channel"),
        Resized(keys=['img'], spatial_size=(256, 256), mode='area', anti_aliasing=True),
        Resized(keys=['seg'], spatial_size=(256, 256), mode='nearest'),
        RandAxisFlipd(keys=['img', 'seg'], prob=0.5),
        RandGaussianSmoothd(keys=['img'], prob=0.5, sigma_x=(0.25, 0.5), sigma_y=(0.25, 0.5))
    ]
)

val_transforms = Compose(
    [
        ScaleIntensityd(keys=["img"]),
        EnsureChannelFirstd(keys=["img", "seg"], channel_dim="no_channel"),
    ]
)

val_transforms_resize = Compose(
    [
        ScaleIntensityd(keys=["img"]),
        EnsureChannelFirstd(keys=["img", "seg"], channel_dim="no_channel"),
        Resized(keys=['img'], spatial_size=(256, 256), mode='area', anti_aliasing=True),
        Resized(keys=['seg'], spatial_size=(256, 256), mode='nearest'),
    ]
)

post_trans = Compose([AsDiscrete(threshold=0.5)])

# Dataset and DataLoader
train_ds = monai.data.Dataset(data=train_data, transform=augm_transforms)
val_ds = monai.data.Dataset(data=val_data, transform=val_transforms)

train_loader = DataLoader(
    train_ds,
    batch_size=6,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_ds,
    batch_size=6,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
)

# Loss, Metric, and Model
dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
dice_loss = monai.losses.DiceLoss(sigmoid=False, include_background=False)
model = DeepLabV3plus(num_classes=1)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device=device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Training
val_interval = 1
best_val_dice = -1  # Modificato per chiarezza

# Crea il file CSV
with open('results_pkl/2D/training_deeplab_log.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 'Validation Dice', 'Best Dice'])

for epoch in range(300):
    print("-" * 10)
    print(f"Epoch {epoch + 1}/300")
    model.train()
    epoch_loss = 0
    for step, batch in enumerate(train_loader, 1):
        inputs, labels = batch["img"].to(device), batch["seg"].to(device)
        optimizer.zero_grad()
        output = torch.sigmoid(model(inputs))
        loss = dice_loss(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
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
                val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                val_output = torch.sigmoid(model(val_images))
                val_output_post = post_trans(val_output)
                val_loss = dice_loss(val_output, val_labels)
                val_loss_sum += val_loss.item()
                dice_metric(val_output_post, val_labels)

            val_loss_sum /= len(val_loader)
            dice_value_avg = dice_metric.aggregate().item()
            dice_metric.reset()
            print(
                f"Epoch {epoch + 1}, Validation Mean Dice: {dice_value_avg:.4f}, "
                f"Validation Loss: {val_loss_sum:.4f}"
            )

            torch.cuda.empty_cache()

            if dice_value_avg > best_val_dice:
                best_val_dice = dice_value_avg
                torch.save(model.state_dict(), 'results_pkl/2D/best_model_deeplab.pth')
                print(f"Model saved at epoch {epoch + 1} with best Dice Score {dice_value_avg:.4f}")

        # Scrivi i risultati nel CSV
        with open('results_pkl/2D/training_deeplab_log.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, epoch_loss, val_loss_sum, dice_value_avg, best_val_dice])
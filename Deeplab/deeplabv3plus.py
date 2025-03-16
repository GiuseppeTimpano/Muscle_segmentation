import torch.nn as nn
import torch
import torch.nn.functional as F
from Deeplab.modules import ResNet50Backbone
from Deeplab.modules import ASPP
from Deeplab.layers import AtrousConvolution

class DeepLabV3plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3plus, self).__init__()
        
        self.backbone = ResNet50Backbone(output_layer='layer3')
        self.low_level_features = ResNet50Backbone(output_layer='layer1')

        self.aspp = ASPP(in_channel=1024, out_channel=256)

        self.conv1x1 = AtrousConvolution(
            input_channel = 256,
            out_channel = 48,
            kernel_size = 1,
            dilated = 1,
            padding=0
        )

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, input):
        x_backbone = self.backbone(input)
        low_level_features = self.low_level_features(input)
        aspp = self.aspp(x_backbone)
        aspp_upscaled = F.interpolate(
            aspp, scale_factor=(4,4),
            mode='bilinear', align_corners=True
        )
        conv1x1 = self.conv1x1(low_level_features)
        concat = torch.cat([conv1x1, aspp_upscaled], dim=1)
        conv3x3 = self.conv3x3(concat)
        conv3x3_upscaled = F.interpolate(
            conv3x3, scale_factor=(4,4),
            mode='bilinear', align_corners=True
        )
        out = self.classifier(conv3x3_upscaled)
        return out
    
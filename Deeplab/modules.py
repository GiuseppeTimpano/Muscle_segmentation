from Deeplab.layers import AtrousConvolution
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as models

class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ASPP, self).__init__()
        self.conv1x1 = AtrousConvolution(in_channel, out_channel, kernel_size=1, dilated=1, padding=0)
        self.conv6x6 = AtrousConvolution(in_channel, out_channel, kernel_size=3, dilated=6, padding=6)
        self.conv12x12 = AtrousConvolution(in_channel, out_channel, kernel_size=3, dilated=12, padding=12)
        self.conv18x18 = AtrousConvolution(in_channel, out_channel, kernel_size=3, dilated=18, padding=18)

        self.pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(out_channel * 5, out_channel, kernel_size=1)

    def forward(self, input):
        x1_conv = self.conv1x1(input)
        x6_conv = self.conv6x6(input)
        x12_conv = self.conv12x12(input)
        x18_conv = self.conv18x18(input)
        pooling = self.pooling(input)
        pooling = F.interpolate(pooling, size=input.size()[2:], mode='bilinear', align_corners=True)
        concat = torch.cat([x1_conv, x6_conv, x12_conv, x18_conv, pooling], dim=1)
        return self.final_conv(concat)
    
class ResNet50Backbone(nn.Module):
    def __init__(self, output_layer=None):
        super(ResNet50Backbone, self).__init__()

        self.pretrained = models.resnet50(pretrained=True)

        self.pretrained.conv1 = nn.Conv2d(
            in_channels=1,  # Cambia il numero di canali in ingresso a 1
            out_channels=self.pretrained.conv1.out_channels,
            kernel_size=self.pretrained.conv1.kernel_size,
            stride=self.pretrained.conv1.stride,
            padding=self.pretrained.conv1.padding,
            bias=False
        )

        self.output_layer = output_layer

        self.layers = list(self.pretrained._modules.keys())

        self.layer_count = 0

        for l in self.layers:
            if l != self.output_layer:
                self.layer_count +=1
            else:
                break
        
        for i in range(1, len(self.layers) - self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        self.net = nn.Sequential(*self.pretrained._modules.values())
        self.pretrained = None
    
    def forward(self, input):
        return self.net(input)
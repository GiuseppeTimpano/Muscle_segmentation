import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import unetConv3d as unetConv3D  # Assumendo che tu abbia definito unetConv3D

class Encoder1(nn.Module):
    def __init__(self, in_channels, is_batchnorm=True):
        super(Encoder1, self).__init__()
        filters = [16, 32, 64, 128, 512]
        self.conv = unetConv3D(in_channels, filters[0], is_batchnorm)
        self.maxpool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input):
        h1 = self.conv(input)
        return self.maxpool(h1)  # h1

class Encoder2(nn.Module):
    def __init__(self, is_batchnorm=True):
        super(Encoder2, self).__init__()
        filters = [16, 32, 64, 128, 512]
        self.conv = unetConv3D(filters[0], filters[1], is_batchnorm)
        self.maxpool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input):
        h1_pool = input  # Riceve solo h1_pool da Encoder1
        h2 = self.conv(h1_pool)
        h2_pool = self.maxpool(h2)
        return h1_pool, h2_pool  # h1, h2

class Encoder3(nn.Module):
    def __init__(self, is_batchnorm=True):
        super(Encoder3, self).__init__()
        filters = [16, 32, 64, 128, 512]
        self.conv = unetConv3D(filters[1], filters[2], is_batchnorm)
        self.maxpool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input):
        h1_pool, h2_pool = input  # Riceve h2 e h2_pool da Encoder2
        h3 = self.conv(h2_pool)
        h3_pool = self.maxpool(h3)
        return h1_pool, h2_pool, h3_pool  # h1, h2, h3

class Encoder4(nn.Module):
    def __init__(self, is_batchnorm=True):
        super(Encoder4, self).__init__()
        filters = [16, 32, 64, 128, 512]
        self.conv = unetConv3D(filters[2], filters[3], is_batchnorm)
        self.maxpool = nn.MaxPool3d(kernel_size=2)

    def forward(self, input):
        h1_pool, h2_pool, h3_pool = input  # Riceve h2, h3 e h3_pool da Encoder3
        h4 = self.conv(h3_pool)
        h4_pool = self.maxpool(h4)
        return h1_pool, h2_pool, h3_pool, h4_pool  # Restituisce h2, h3, h4 (non poolato) e h4_pool

class Bottleneck(nn.Module):
    def __init__(self, is_batchnorm=True):
        super(Bottleneck, self).__init__()
        filters = [16, 32, 64, 128, 512]
        self.conv = unetConv3D(filters[3], filters[4], is_batchnorm)

    def forward(self, input):
        h1_pool, h2_pool, h3_pool, h4_pool = input  # Riceve h2, h3, h4 e h4_pool da Encoder4
        bottleneck = self.conv(h4_pool)
        return h1_pool, h2_pool, h3_pool, h4_pool, bottleneck  # Restituisce h2, h3, h4 e bottleneck

class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        filters = [16, 32, 64, 128, 512]
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        self.h1_PT_hd4 = nn.MaxPool3d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd4 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h3_PT_hd4 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h4_Cat_hd4_conv = nn.Conv3d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.hd5_UT_hd4_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        self.conv4d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn4d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)
    
    def forward(self, input):
        h1, h2, h3, h4, hd5 = input

        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))

        target_size = h1_PT_hd4.size()[2:]
        h2_PT_hd4 = F.interpolate(h2_PT_hd4, size=target_size, mode='trilinear', align_corners=True)
        h3_PT_hd4 = F.interpolate(h3_PT_hd4, size=target_size, mode='trilinear', align_corners=True)
        h4_Cat_hd4 = F.interpolate(h4_Cat_hd4, size=target_size, mode='trilinear', align_corners=True)
        hd5_UT_hd4 = F.interpolate(hd5_UT_hd4, size=target_size, mode='trilinear', align_corners=True)

        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))
        print("Decoder4 hd4 shape:", hd4.shape)

        return h1, h2, h3, h4, hd5, hd4
    
class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
        filters = [16, 32, 64, 128, 512]

        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        # h1, h2 e h3
        self.h1_PT_hd3 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd3 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h3_Cat_hd3_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4, hd5 per upsample
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='trilinear')
        self.hd4_UT_hd3_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='trilinear')
        self.hd5_UT_hd3_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # Conv finale per Decoder3
        self.conv3d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn3d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

    def forward(self, input):
        h1, h2, h3, h4, hd5, hd4 = input

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))

        target_size = h1_PT_hd3.size()[2:]
        h2_PT_hd3 = F.interpolate(h2_PT_hd3, size=target_size, mode='trilinear', align_corners=True)
        h3_Cat_hd3 = F.interpolate(h3_Cat_hd3, size=target_size, mode='trilinear', align_corners=True)
        hd4_UT_hd3 = F.interpolate(hd4_UT_hd3, size=target_size, mode='trilinear', align_corners=True)
        hd5_UT_hd3 = F.interpolate(hd5_UT_hd3, size=target_size, mode='trilinear', align_corners=True)

        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))
        print("Decoder3 hd3 shape:", hd3.shape)
        return h1, h2, h3, h4, hd5, hd4, hd3


class Decoder2(nn.Module):
    def __init__(self):
        super(Decoder2, self).__init__()
        filters = [16, 32, 64, 128, 512]

        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 2d '''
        # h1->320*320*320, hd2->160*160*160, Pooling 2 times
        self.h1_PT_hd2 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->160*160*160, hd2->160*160*160, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->80*80*80, hd2->160*160*160, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='trilinear')  # 14*14*14
        self.hd3_UT_hd2_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->40*40*40, hd2->160*160*160, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='trilinear')  # 14*14*14
        self.hd4_UT_hd2_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->20*20*20, hd2->160*160*160, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='trilinear')  # 14*14*14
        self.hd5_UT_hd2_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.conv2d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn2d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

    def forward(self, input):
        h1, h2, h3, h4, hd5, hd4, hd3 = input

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        print(hd5_UT_hd2.size(), h1_PT_hd2.size())
        
        target_size = hd5_UT_hd2.size()[2:]
        h1_PT_hd2 = F.interpolate(h1_PT_hd2, size=target_size, mode='trilinear', align_corners=True)
        h2_Cat_hd2 = F.interpolate(h2_Cat_hd2, size=target_size, mode='trilinear', align_corners=True)
        hd3_UT_hd2 = F.interpolate(hd3_UT_hd2, size=target_size, mode='trilinear', align_corners=True)
        hd4_UT_hd2 = F.interpolate(hd4_UT_hd2, size=target_size, mode='trilinear', align_corners=True)
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))
        print("Decoder2 hd2 shape:", hd2.shape)
        return h1, h2, h3, h4, hd5, hd4, hd3, hd2

class Decoder1(nn.Module):
    def __init__(self, n_classes):
        super(Decoder1, self).__init__()
        filters = [16, 32, 64, 128, 512]

        self.n_classes = n_classes
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 1d '''
        # h1->n*m*d, hd1->n*m*d, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->160*160*160, hd1->320*320*320, Upsample 2 times
        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='trilinear')  # 14*14*14
        self.hd2_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->80*80*80, hd1->320*320*320, Upsample 4 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='trilinear')  # 14*14*14
        self.hd3_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->40*40*40, hd1->320*320*320, Upsample 8 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='trilinear')  # 14*14*14
        self.hd4_UT_hd1_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->20*20*20, hd1->320*320*320, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='trilinear')  # 14*14*14
        self.hd5_UT_hd1_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.conv1d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn1d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.outconv1 = nn.Conv3d(self.UpChannels, self.n_classes, 3, padding=1)

    def forward(self, input):
        h1, hd5, hd4, hd3, hd2 = input[0], input[4], input[5], input[6], input[7]

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))

        target_size = h1_Cat_hd1.size()[2:]
        hd5_UT_hd1 = F.interpolate(hd5_UT_hd1, size=target_size, mode='trilinear', align_corners=True)
        hd2_UT_hd1 = F.interpolate(hd2_UT_hd1, size=target_size, mode='trilinear', align_corners=True)
        hd3_UT_hd1 = F.interpolate(hd3_UT_hd1, size=target_size, mode='trilinear', align_corners=True)
        hd4_UT_hd1 = F.interpolate(hd4_UT_hd1, size=target_size, mode='trilinear', align_corners=True)
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))
        print("Decoder1 hd1 shape:", hd1.shape)
        return self.outconv1(hd1)
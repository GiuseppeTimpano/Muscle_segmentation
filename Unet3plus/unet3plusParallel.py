import torch.nn as nn
import torch
import torch.nn.functional as F
from models.layers import unetConv3d as unetConv3D
import torch.utils.checkpoint as checkpoint

class UNet_3Plus_3D(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet_3Plus_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes = n_classes

        filters = [8, 16, 32, 64, 128]

        ## -------------Encoder--------------
        # Encoder su GPU 0 e 1
        self.conv1 = unetConv3D(self.in_channels, filters[0], self.is_batchnorm).to('cuda:0')
        self.maxpool1 = nn.MaxPool3d(kernel_size=2).to('cuda:0')

        self.conv2 = unetConv3D(filters[0], filters[1], self.is_batchnorm).to('cuda:1')
        self.maxpool2 = nn.MaxPool3d(kernel_size=2).to('cuda:1')

        self.conv3 = unetConv3D(filters[1], filters[2], self.is_batchnorm).to('cuda:2')
        self.maxpool3 = nn.MaxPool3d(kernel_size=2).to('cuda:2')

        self.conv4 = unetConv3D(filters[2], filters[3], self.is_batchnorm).to('cuda:3')
        self.maxpool4 = nn.MaxPool3d(kernel_size=2).to('cuda:3')

        ## -------------Bottleneck--------------
        # Bottleneck su GPU 4
        self.conv5 = unetConv3D(filters[3], filters[4], self.is_batchnorm).to('cuda:4')

        ## -------------Decoder--------------
        # Stage 4d - primo blocco del decoder su GPU 5
        self.h1_PT_hd4 = nn.MaxPool3d(8, 8, ceil_mode=True).to('cuda:5')
        self.h1_PT_hd4_conv = nn.Conv3d(filters[0], filters[0], 3, padding=1).to('cuda:5')
        self.h1_PT_hd4_bn = nn.BatchNorm3d(filters[0]).to('cuda:5')
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True).to('cuda:5')

        self.h2_PT_hd4 = nn.MaxPool3d(4, 4, ceil_mode=True).to('cuda:5')
        self.h2_PT_hd4_conv = nn.Conv3d(filters[1], filters[0], 3, padding=1).to('cuda:5')
        self.h2_PT_hd4_bn = nn.BatchNorm3d(filters[0]).to('cuda:5')
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True).to('cuda:5')

        self.h3_PT_hd4 = nn.MaxPool3d(2, 2, ceil_mode=True).to('cuda:5')
        self.h3_PT_hd4_conv = nn.Conv3d(filters[2], filters[0], 3, padding=1).to('cuda:5')
        self.h3_PT_hd4_bn = nn.BatchNorm3d(filters[0]).to('cuda:5')
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True).to('cuda:5')

        self.h4_Cat_hd4_conv = nn.Conv3d(filters[3], filters[0], 3, padding=1).to('cuda:5')
        self.h4_Cat_hd4_bn = nn.BatchNorm3d(filters[0]).to('cuda:5')
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True).to('cuda:5')

        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='trilinear').to('cuda:5')
        self.hd5_UT_hd4_conv = nn.Conv3d(filters[4], filters[0], 3, padding=1).to('cuda:5')
        self.hd5_UT_hd4_bn = nn.BatchNorm3d(filters[0]).to('cuda:5')
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True).to('cuda:5')

        self.conv4d_1 = nn.Conv3d(filters[0] * 5, filters[0] * 5, 3, padding=1).to('cuda:5')
        self.bn4d_1 = nn.BatchNorm3d(filters[0] * 5).to('cuda:5')
        self.relu4d_1 = nn.ReLU(inplace=True).to('cuda:5')

        # Stage 3d - secondo blocco del decoder su GPU 6
        self.h1_PT_hd3 = nn.MaxPool3d(4, 4, ceil_mode=True).to('cuda:6')
        self.h1_PT_hd3_conv = nn.Conv3d(filters[0], filters[0], 3, padding=1).to('cuda:6')
        self.h1_PT_hd3_bn = nn.BatchNorm3d(filters[0]).to('cuda:6')
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True).to('cuda:6')

        self.h2_PT_hd3 = nn.MaxPool3d(2, 2, ceil_mode=True).to('cuda:6')
        self.h2_PT_hd3_conv = nn.Conv3d(filters[1], filters[0], 3, padding=1).to('cuda:6')
        self.h2_PT_hd3_bn = nn.BatchNorm3d(filters[0]).to('cuda:6')
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True).to('cuda:6')

        self.h3_Cat_hd3_conv = nn.Conv3d(filters[2], filters[0], 3, padding=1).to('cuda:6')
        self.h3_Cat_hd3_bn = nn.BatchNorm3d(filters[0]).to('cuda:6')
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True).to('cuda:6')

        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='trilinear').to('cuda:6')
        self.hd4_UT_hd3_conv = nn.Conv3d(filters[0] * 5, filters[0], 3, padding=1).to('cuda:6')
        self.hd4_UT_hd3_bn = nn.BatchNorm3d(filters[0]).to('cuda:6')
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True).to('cuda:6')

        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='trilinear').to('cuda:6')
        self.hd5_UT_hd3_conv = nn.Conv3d(filters[4], filters[0], 3, padding=1).to('cuda:6')
        self.hd5_UT_hd3_bn = nn.BatchNorm3d(filters[0]).to('cuda:6')
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True).to('cuda:6')

        self.conv3d_1 = nn.Conv3d(filters[0] * 5, filters[0] * 5, 3, padding=1).to('cuda:6')
        self.bn3d_1 = nn.BatchNorm3d(filters[0] * 5).to('cuda:6')
        self.relu3d_1 = nn.ReLU(inplace=True).to('cuda:6')

        # Stage 2d - terzo blocco del decoder su GPU 7
        self.h1_PT_hd2 = nn.MaxPool3d(2, 2, ceil_mode=True).to('cuda:7')
        self.h1_PT_hd2_conv = nn.Conv3d(filters[0], filters[0], 3, padding=1).to('cuda:7')
        self.h1_PT_hd2_bn = nn.BatchNorm3d(filters[0]).to('cuda:7')
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True).to('cuda:7')

        self.h2_Cat_hd2_conv = nn.Conv3d(filters[1], filters[0], 3, padding=1).to('cuda:7')
        self.h2_Cat_hd2_bn = nn.BatchNorm3d(filters[0]).to('cuda:7')
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True).to('cuda:7')

        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='trilinear').to('cuda:7')
        self.hd3_UT_hd2_conv = nn.Conv3d(filters[0] * 5, filters[0], 3, padding=1).to('cuda:7')
        self.hd3_UT_hd2_bn = nn.BatchNorm3d(filters[0]).to('cuda:7')
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True).to('cuda:7')

        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='trilinear').to('cuda:7')
        self.hd4_UT_hd2_conv = nn.Conv3d(filters[0] * 5, filters[0], 3, padding=1).to('cuda:7')
        self.hd4_UT_hd2_bn = nn.BatchNorm3d(filters[0]).to('cuda:7')
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True).to('cuda:7')

        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='trilinear').to('cuda:7')
        self.hd5_UT_hd2_conv = nn.Conv3d(filters[4], filters[0], 3, padding=1).to('cuda:7')
        self.hd5_UT_hd2_bn = nn.BatchNorm3d(filters[0]).to('cuda:7')
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True).to('cuda:7')

        self.conv2d_1 = nn.Conv3d(filters[0] * 5, filters[0] * 5, 3, padding=1).to('cuda:7')
        self.bn2d_1 = nn.BatchNorm3d(filters[0] * 5).to('cuda:7')
        self.relu2d_1 = nn.ReLU(inplace=True).to('cuda:7')

        # Decoder 1d - Ultimo blocco su GPU 7
        self.h1_Cat_hd1_conv = nn.Conv3d(filters[0], filters[0], 3, padding=1).to('cuda:7')
        self.h1_Cat_hd1_bn = nn.BatchNorm3d(filters[0]).to('cuda:7')
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True).to('cuda:7')

        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode='trilinear').to('cuda:7')
        self.hd2_UT_hd1_conv = nn.Conv3d(filters[0] * 5, filters[0], 3, padding=1).to('cuda:7')
        self.hd2_UT_hd1_bn = nn.BatchNorm3d(filters[0]).to('cuda:7')
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True).to('cuda:7')

        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode='trilinear').to('cuda:7')
        self.hd3_UT_hd1_conv = nn.Conv3d(filters[0] * 5, filters[0], 3, padding=1).to('cuda:7')
        self.hd3_UT_hd1_bn = nn.BatchNorm3d(filters[0]).to('cuda:7')
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True).to('cuda:7')

        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode='trilinear').to('cuda:7')
        self.hd4_UT_hd1_conv = nn.Conv3d(filters[0] * 5, filters[0], 3, padding=1).to('cuda:7')
        self.hd4_UT_hd1_bn = nn.BatchNorm3d(filters[0]).to('cuda:7')
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True).to('cuda:7')

        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode='trilinear').to('cuda:7')
        self.hd5_UT_hd1_conv = nn.Conv3d(filters[4], filters[0], 3, padding=1).to('cuda:7')
        self.hd5_UT_hd1_bn = nn.BatchNorm3d(filters[0]).to('cuda:7')
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True).to('cuda:7')

        self.conv1d_1 = nn.Conv3d(filters[0] * 5, filters[0] * 5, 3, padding=1).to('cuda:7')
        self.bn1d_1 = nn.BatchNorm3d(filters[0] * 5).to('cuda:7')
        self.relu1d_1 = nn.ReLU(inplace=True).to('cuda:7')

        '''output layer su GPU 7'''
        self.outconv1 = nn.Conv3d(filters[0] * 5, self.n_classes, 3, padding=1).to('cuda:7')

    def forward(self, inputs):
        # Encoder su GPU 0, 1, 2, 3
        h1 = self.conv1(inputs.to('cuda:0'))
        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2.to('cuda:1'))
        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3.to('cuda:2'))
        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4.to('cuda:3'))
        h5 = self.maxpool4(h4)

        # Bottleneck su GPU 4
        hd5 = self.conv5(h5.to('cuda:4'))

        # Decoder - Stage 4d su GPU 5
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1.to('cuda:5')))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2.to('cuda:5')))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3.to('cuda:5')))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4.to('cuda:5'))))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5.to('cuda:5')))))
        target_size = h1_PT_hd4.size()[2:]
        h2_PT_hd4 = F.interpolate(h2_PT_hd4, size=target_size, mode='trilinear', align_corners=True)
        h3_PT_hd4 = F.interpolate(h3_PT_hd4, size=target_size, mode='trilinear', align_corners=True)
        h4_Cat_hd4 = F.interpolate(h4_Cat_hd4, size=target_size, mode='trilinear', align_corners=True)
        hd5_UT_hd4 = F.interpolate(hd5_UT_hd4, size=target_size, mode='trilinear', align_corners=True)
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))

        # Decoder - Stage 3d su GPU 6
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1.to('cuda:6')))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2.to('cuda:6')))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3.to('cuda:6'))))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4.to('cuda:6')))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5.to('cuda:6')))))
        target_size = h1_PT_hd3.size()[2:]
        h2_PT_hd3 = F.interpolate(h2_PT_hd3, size=target_size, mode='trilinear', align_corners=True)
        h3_Cat_hd3 = F.interpolate(h3_Cat_hd3, size=target_size, mode='trilinear', align_corners=True)
        hd4_UT_hd3 = F.interpolate(hd4_UT_hd3, size=target_size, mode='trilinear', align_corners=True)
        hd5_UT_hd3 = F.interpolate(hd5_UT_hd3, size=target_size, mode='trilinear', align_corners=True)
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))

        # Decoder - Stage 2d su GPU 7
        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1.to('cuda:7')))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2.to('cuda:7'))))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3.to('cuda:7')))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4.to('cuda:7')))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5.to('cuda:7')))))
        target_size = h1_PT_hd2.size()[2:]
        h2_Cat_hd2 = F.interpolate(h2_Cat_hd2, size=target_size, mode='trilinear', align_corners=True)
        hd3_UT_hd2 = F.interpolate(hd3_UT_hd2, size=target_size, mode='trilinear', align_corners=True)
        hd4_UT_hd2 = F.interpolate(hd4_UT_hd2, size=target_size, mode='trilinear', align_corners=True)
        hd5_UT_hd2 = F.interpolate(hd5_UT_hd2, size=target_size, mode='trilinear', align_corners=True)
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))
        
        # Decoder - Stage 1d su GPU 7
        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1.to('cuda:7'))))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3.to('cuda:7')))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4.to('cuda:7')))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5.to('cuda:7')))))
        target_size = h1_Cat_hd1.size()[2:]
        hd2_UT_hd1 = F.interpolate(hd2_UT_hd1, size=target_size, mode='trilinear', align_corners=True)
        hd3_UT_hd1 = F.interpolate(hd3_UT_hd1, size=target_size, mode='trilinear', align_corners=True)
        hd4_UT_hd1 = F.interpolate(hd4_UT_hd1, size=target_size, mode='trilinear', align_corners=True)
        hd5_UT_hd1 = F.interpolate(hd5_UT_hd1, size=target_size, mode='trilinear', align_corners=True)
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))
    
        # Output layer su GPU 7
        d1 = self.outconv1(hd1)

        return d1
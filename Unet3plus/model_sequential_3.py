import torch
import torch.nn as nn
from models.layers import unetConv3d
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, in_channels=3, filters=[], is_batchnorm=True):
        super(Encoder, self).__init__()
        self.filters = filters
        
        self.conv1 = unetConv3d(in_channels, filters[0], is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=2)
        
        self.conv2 = unetConv3d(filters[0], filters[1], is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2)
        
        self.conv3 = unetConv3d(filters[1], filters[2], is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2)
        
        self.conv4 = unetConv3d(filters[2], filters[3], is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2)
    
    def forward(self, x):
        h1 = self.conv1(x)
        h2 = self.conv2(self.maxpool1(h1))
        h3 = self.conv3(self.maxpool2(h2))
        h4 = self.conv4(self.maxpool3(h3))
        hd5 = self.maxpool4(h4)
        return h1, h2, h3, h4, hd5


class Bottleneck(nn.Module):
    def __init__(self, filters, is_batchnorm=True):
        super(Bottleneck, self).__init__()
        self.filters = filters
        self.conv5 = unetConv3d(filters[3], filters[4], is_batchnorm)
    
    def forward(self, inputs):
        h1, h2, h3, h4, hd5 = inputs  
        hd5 = self.conv5(hd5) 
        return h1, h2, h3, h4, hd5


class Decoder(nn.Module):
    def __init__(self, filters, n_classes):
        super(Decoder, self).__init__()
        self.filters = filters
        CatChannels = filters[0]
        UpChannels = CatChannels * 5
        self.n_classes = n_classes

        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        '''stage 4d'''
        # h1->320*320*320, hd4->40*40*40, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool3d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->160*160*160, hd4->40*40*40, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->80*80*80, hd4->40*40*40, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->40*40*40, hd4->40*40*40, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv3d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->20*20*20, hd4->40*40*40, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='trilinear')  # 14*14*14
        self.hd5_UT_hd4_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn4d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        '''stage 3d'''
        # h1->320*320*320, hd3->80*80*80, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool3d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv3d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->160*160*160, hd3->80*80*80, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool3d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv3d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->80*80*80, hd3->80*80*80, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv3d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->40*40*40, hd4->80*80*80, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='trilinear')  # 14*14*14
        self.hd4_UT_hd3_conv = nn.Conv3d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->20*20*20, hd4->80*80*80, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='trilinear')  # 14*14*14
        self.hd5_UT_hd3_conv = nn.Conv3d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm3d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv3d(self.UpChannels, self.UpChannels, 3, padding=1)  # 16
        self.bn3d_1 = nn.BatchNorm3d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

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

        '''output'''
        self.outconv1 = nn.Conv3d(self.UpChannels, self.n_classes, 3, padding=1)

    def forward(self, inputs):
        h1, h2, h3, h4, hd5 = inputs

        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))

        # interpolate to match tensors size
        target_size = h1_PT_hd4.size()[2:]
        hd5_UT_hd4 = F.interpolate(hd5_UT_hd4, size=target_size, mode='trilinear', align_corners=True)
        h2_PT_hd4 = F.interpolate(h2_PT_hd4, size=target_size, mode='trilinear', align_corners=True)
        h3_PT_hd4 = F.interpolate(h3_PT_hd4, size=target_size, mode='trilinear', align_corners=True)
        h4_Cat_hd4 = F.interpolate(h4_Cat_hd4, size=target_size, mode='trilinear', align_corners=True)
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))

        target_size = h1_PT_hd3.size()[2:]
        hd5_UT_hd3 = F.interpolate(hd5_UT_hd3, size=target_size, mode='trilinear', align_corners=True)
        h2_PT_hd3 = F.interpolate(h2_PT_hd3, size=target_size, mode='trilinear', align_corners=True)
        h3_Cat_hd3 = F.interpolate(h3_Cat_hd3, size=target_size, mode='trilinear', align_corners=True)
        hd4_UT_hd3 = F.interpolate(hd4_UT_hd3, size=target_size, mode='trilinear', align_corners=True)
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        
        target_size = hd5_UT_hd2.size()[2:]
        h1_PT_hd2 = F.interpolate(h1_PT_hd2, size=target_size, mode='trilinear', align_corners=True)
        h2_Cat_hd2 = F.interpolate(h2_Cat_hd2, size=target_size, mode='trilinear', align_corners=True)
        hd3_UT_hd2 = F.interpolate(hd3_UT_hd2, size=target_size, mode='trilinear', align_corners=True)
        hd4_UT_hd2 = F.interpolate(hd4_UT_hd2, size=target_size, mode='trilinear', align_corners=True)
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))

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

        d1 = self.outconv1(hd1)  # d1->320*320*320
        return d1


import torch
import torch.nn as nn
import torch.nn.functional as F
from Unet3plus.init_weights import init_weights

# Conv3D module for U-NET 3plus
# ks: kernel_size
class unetConv3d(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv3d, self).__init__()
        self.n = n 
        self.ks = ks 
        self.is_batchnorm = is_batchnorm
        self.s = stride
        self.p = padding

        if self.is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv3d(in_size, out_size, self.ks, self.s, self.p),
                    nn.BatchNorm3d(out_size),
                    nn.ReLU(inplace=True)
                )
                setattr(self, 'conv%d' %i, conv)
                in_size = out_size
        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(
                    nn.Conv3d(in_size, out_size, self.ks, self.s, self.p),
                    nn.ReLU(inplace=True)
                )
                setattr(self, 'conv%d' %i, conv) # save each convolution block 
                in_size = out_size

        # inzialize weights: kaiming for conv3d layers
        # self.children return all layers
        for m in self.children():
            init_weights(m, init_type='kaiming')


    def forward(self, input):
        x = input
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d' %i) # take attr conv defined before
            x = conv(x)
        return x


class unetUp3D(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat):
        super(unetUp3D, self).__init__()
        self.conv = unetConv3d(out_size*2, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else: 
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        #inizialize weights here
        for m in self.children():
            if m.__class__.__name__.find('unetConv3d') != -1: continue
            init_weights(m, init_type='kaiming')

    # *input = variable parameter as input
    def forward(self, inputs0, *input): 
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            # concatenate up features with encoder features
            outputs0 = torch.cat([outputs0, input[i]], 1) 
        return self.conv(outputs0)


class unetUp3D_origin(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat):
        super(unetUp3D_origin, self).__init__()
        if is_deconv:
            self.conv = unetConv3d(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else: 
            self.conv = unetConv3d(in_size + (n_concat - 2) * out_size, out_size, False)
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        
        for m in self.children():
            if m.__class__.__name__.find('unetConv3d') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs0, *input):
        outputs0 = self.up(inputs0)
        for i in range(len(input)):
            outputs0 = torch.cat([outputs0, input[i]], 1)
        return self.conv(outputs0)
import torch
import torch.nn as nn
import torch.nn.functional as F

class AtrousConvolution(nn.Module):
    def __init__ (self, input_channel, out_channel, kernel_size, dilated, padding):
        super(AtrousConvolution, self).__init__()
        self.input_channel = input_channel
        self.out_channel = out_channel
        self.dilated = dilated
        self.padding = padding
        self.kernel_size = kernel_size
    
        self.conv2d = nn.Conv2d(in_channels = self.input_channel, out_channels = self.out_channel, 
                                kernel_size = self.kernel_size, padding = self.padding, 
                                dilation = self.dilated)
        
        self.batchnorm = nn.BatchNorm2d(self.out_channel)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.batchnorm(self.conv2d(x)))
        return x
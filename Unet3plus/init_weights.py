import torch 
import torch.nn as nn
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Conv3d') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') !=-1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('BatchNorm3d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0) 

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Conv3d') != -1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('Linear') !=-1:
        init.xavier_normal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1 or classname.find('BatchNorm3d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0) 

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Conv3d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') !=-1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1 or classname.find('BatchNorm3d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0) 

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Conv3d') != -1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('Linear') !=-1:
        init.orthogonal_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1 or classname.find('BatchNorm3d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0) 

def init_weights(net, init_type='normal'):
    if init_type=='normal':
        net.apply(weights_init_normal)
    elif init_type=='xavier':
        net.apply(weights_init_xavier)
    elif init_type=='orthogonal':
        net.apply(weights_init_orthogonal)
    elif init_type=='kaiming':
        net.apply(weights_init_kaiming)
    else: 
        raise NotImplementedError('Inizialization method [%s] is not implemented' %init_type)
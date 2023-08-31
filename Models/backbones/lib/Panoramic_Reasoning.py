import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import time

class _PanoramicReasoningBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True, model_confs=None, mode=None):
        super(_PanoramicReasoningBlockND, self).__init__()
        self.ReasoningMap = None
        assert dimension in [1, 2, 3]
        self.dimension = dimension
        self.sub_sample = sub_sample
        self.mode = mode

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
            
        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)


    def reasoning(self, x, function_type='dot_product'):
        '''
        :param x: (b, c, t, h, w) # input feature map
        :return z: (b, c', t, h, w) # relational feature
        '''
        # x: (b, c, h, w)
        b = x.size(0)
        g_x = self.g(x).view(b, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)


        theta_x = self.theta(x).view(b, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(b, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        if function_type == 'dot_product':
            N = f.size(-1)
            f_div_C = f / N
        elif function_type =='embedded_gaussian':
            f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(b, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z
    
    def forward(self, x, boxes_tensor=None):
        '''
        :param x: (b, c, t, h, w), boxes_tensor:(b, k, 4)
        :return:
        '''
        # x: (b, c, h, w)
        since = time.time()
        
        if self.mode is 'NonLocal':
            return self.reasoning(x, 'dot_product')
        else:
            b, _, h, w = x.size()
            if boxes_tensor is not None:
                for idx in range(b):
                    for box in boxes_tensor[idx]:
                        left, top, right, bottom = box.int()
                        x[idx, :, top:bottom+1, left:right+1] = self.reasoning(torch.unsqueeze(x[idx, :, top:bottom+1, left:right+1], 0))
#                 print ('Execute spatial reasoning, takes', (time.time()-since), 's')
                return x
            else:
                kernel_size = 3
                stride = 3
                for _h in range(0, h-kernel_size+1, stride):
                    for _w in range(0, w-kernel_size+1, stride):
                        x[:, :, _h:_h+kernel_size, _w:_w+kernel_size] = self.reasoning(x[:, :, _h:_h+kernel_size, _w:_w+kernel_size])
#                 print 'Execute spatial reasoning, takes', (time.time()-since), 's'
                return x
        

class PanoramicReasoningBlock1D(_PanoramicReasoningBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, model_confs=None, mode=None):
        super(PanoramicReasoningBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer, model_confs=model_confs,
                                            mode=mode)


class PanoramicReasoningBlock2D(_PanoramicReasoningBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, model_confs=None, mode=None):
        super(PanoramicReasoningBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer, model_confs=model_confs,
                                            mode=mode)


class PanoramicReasoningBlock3D(_PanoramicReasoningBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True, model_confs=None, mode=None):
        super(PanoramicReasoningBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer, model_confs=model_confs,
                                            mode=mode)
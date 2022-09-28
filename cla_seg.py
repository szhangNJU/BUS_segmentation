# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:37:29 2020

@author: zs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

def weights_init(m):
    
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight,a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(m.bias, -bound, bound)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(UNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        self.cls = nn.Sequential(
            nn.Conv2d(filters[4],256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
            )
        self.linear = nn.Linear(256,1)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

       # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        cls = self.cls(e5)
        cls = cls.view(cls.size(0),-1)
        cls = self.linear(cls)
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        out = self.Conv(d2)
        return out,cls

class conv_block_nested(nn.Module):
    
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output
    
#Nested Unet

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """
    def __init__(self, in_ch=3, out_ch=1):
        super(NestedUNet, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.cls = nn.Sequential(
            nn.Conv2d(filters[4],256,3,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
            )
        self.linear = nn.Linear(256,1)

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], out_ch, kernel_size=1)


    def forward(self, x):
        
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        cls = self.cls(x4_0)
        cls = cls.view(cls.size(0),-1)
        cls = self.linear(cls)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output,cls

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate, cat = True):
        super(Bottleneck, self).__init__()
        self.cat = cat
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.cat:
            out = torch.cat([out,x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(x, 2)
        return out

class UpTransition(nn.Module):
    def __init__(self, in_ch, out_ch, up):
        super(UpTransition, self).__init__()
        self.bn = nn.BatchNorm2d(in_ch)
        if up =='upsample':
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        out = self.up(F.relu(self.bn(x)))
        return out

class DenseUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, block = Bottleneck, n_blocks=[3,4,8,12], growth = 12, reduction=0.5, up='upsample' ):
        super(DenseUNet, self).__init__()
        self.growth_rate = growth
        n_ch = 2*growth
        self.conv = nn.Conv2d(in_ch, n_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.densedown = nn.ModuleList()
        self.downconv = nn.ModuleList()
        self.down = nn.ModuleList()
        self.denseup = nn.ModuleList()
        self.up = nn.ModuleList()
        self.n_b = len(n_blocks)
        skip_ch = []
        for i in range(self.n_b-1):
            self.densedown.append(self._make_dense_layers(block,n_ch,n_blocks[i]))
            n_ch += growth * n_blocks[i]
            o_ch = int(math.floor(n_ch * reduction))
            self.down.append(Transition(n_ch, o_ch))
            self.downconv.append(block(n_ch,o_ch,cat=False))
            n_ch = o_ch
            skip_ch.append(n_ch)
        self.densedown.append(self._make_dense_layers(block, n_ch, n_blocks[-1]))
        n_ch += growth * n_blocks[-1]
        o_ch = int(math.floor(n_ch * reduction))
        self.downconv.append(block(n_ch, o_ch, cat=False))
        self.cls = nn.Sequential(
            nn.BatchNorm2d(o_ch),
            nn.Conv2d(o_ch,256,3,2,1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
            )
        #self.se = nn.Conv2d(256,o_ch,1)
        self.linear = nn.Linear(256,1)
        n_ch = o_ch # * 2 # *2 for concat
        skip_ch.append(n_ch)
        skip_ch = skip_ch[::-1]
        n_blocks = n_blocks[::-1]
        for i in range(1,self.n_b):
            o_ch = growth*n_blocks[i]
            self.up.append(UpTransition(n_ch,o_ch,up))
            n_ch = o_ch + skip_ch[i]
            self.denseup.append(self._make_dense_layers(block, n_ch, n_blocks[i]))
            n_ch += growth * n_blocks[i]
        self.top = nn.Sequential(
            nn.BatchNorm2d(n_ch),
            nn.ConvTranspose2d(n_ch,256,3,2,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,out_ch,1)
        )
    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        skip = []
        for i in range(self.n_b-1):
            x = self.densedown[i](x)
            skip.append(self.downconv[i](x))
            x = self.down[i](x)
        x=self.downconv[-1](self.densedown[-1](x))
        cls = self.cls(x)
        #se = self.se(cls)
        #se = torch.sigmoid(se) # for channel attention
        cls = cls.view(cls.size(0),-1)
        cls = self.linear(cls)
        skip = skip[::-1]
        #x = x * se  # channel attention
        #x = torch.cat([x,F.interpolate(se,x.shape[2:])],1) # concat
        for i in range(self.n_b-1):
            x = self.up[i](x)
            x = self.denseup[i](torch.cat([x,skip[i]],1))
        x = self.top(x)
        return x,cls

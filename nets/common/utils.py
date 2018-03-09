#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-09-14 11:55:36
'''
import math
import torch
import torch.nn as nn


class Conv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, use_relu=True, use_bn=False, groups=1, padding=True,
                 dilation=1, bias=True, init_std=0.01, init_bias=0.0,
                 init='kaiming', momentum=0.1):
        super(Conv2d, self).__init__()
        self.use_relu = use_relu
        self.use_bn = use_bn

        padding = int((kernel_size - 1) / 2) if padding else 0
        padding = dilation if dilation > 1 else padding
        bias = False if self.use_bn else bias
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size, stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channel, momentum=momentum)
        if self.use_relu:
            self.relu = nn.ReLU(inplace=True)

        # Initialize
        if init == 'kaiming':
            torch.nn.init.kaiming_normal(self.conv.weight.data)
        else:
            torch.nn.init.normal(self.conv.weight.data, std=init_std)
        if bias:
            torch.nn.init.constant(self.conv.bias.data, init_bias)
        if self.use_bn:
            torch.nn.init.normal(self.bn.weight.data, 1.0)
            torch.nn.init.constant(self.bn.bias.data, 0.0)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class SeperableConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size=3,
                 stride=1, padding=True, dilation=1, use_bn=True,
                 use_relu=True, init='kaiming', momentum=0.1):
        super(SeperableConv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if padding else 0
        padding = dilation if dilation > 1 else padding
        self.depthwise = self.conv_dw(in_channel, kernel_size, stride, padding,
                                      dilation, momentum=momentum)
        self.pointwise = self.conv_pw(
            in_channel,
            out_channel,
            stride,
            momentum=momentum,
            use_bn=use_bn,
            use_relu=use_relu)

        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def conv_dw(self, in_channel, kernel_size=3, stride=1, padding=1,
                dilation=1, momentum=0.1):
        # No relu in depthwise conv
        return nn.Sequential(
            nn.Conv2d(
                in_channel, in_channel, kernel_size, stride, padding=padding,
                groups=in_channel, dilation=dilation, bias=False),
            nn.BatchNorm2d(in_channel, momentum=momentum),)

    def conv_pw(self, in_channel, out_channel, stride=1,
                momentum=0.1, use_bn=True, use_relu=True):
        modules = [nn.Conv2d(in_channel, out_channel, 1, stride,
                             padding=0, bias=False)]
        if use_bn:
            modules.append(nn.BatchNorm2d(out_channel, momentum=momentum))
        if use_relu:
            modules.append(nn.ReLU(inplace=True))
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class LargeKernelConv2d(nn.Module):

    def __init__(self, in_channel, out_channel, mid_channel=128,
                 kernel_size=15, init='kaiming', init_std=0.01, init_bias=0.):
        super(LargeKernelConv2d, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv11 = nn.Conv2d(in_channel, mid_channel,
                                (kernel_size, 1), stride=1,
                                padding=(padding, 0), bias=True)
        self.conv12 = nn.Conv2d(mid_channel, out_channel,
                                (1, kernel_size), stride=1,
                                padding=(0, padding), bias=True)
        self.conv21 = nn.Conv2d(in_channel, mid_channel,
                                (1, kernel_size), stride=1,
                                padding=(0, padding), bias=True)
        self.conv22 = nn.Conv2d(mid_channel, out_channel,
                                (kernel_size, 1), stride=1,
                                padding=(padding, 0), bias=True)
        # Initialize
        for i, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                if init == 'kaiming':
                    torch.nn.init.kaiming_normal(m.weight.data)
                else:
                    torch.nn.init.normal(m.weight.data, std=init_std)
                torch.nn.init.constant(m.bias.data, init_bias)

    def forward(self, x):
        p1 = self.conv12(self.conv11(x))
        p2 = self.conv22(self.conv21(x))
        return p1 + p2

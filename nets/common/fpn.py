#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-09-14 11:55:36
'''

import torch.nn as nn
import torch.nn.functional as F

from .utils import Conv2d, LargeKernelConv2d

class FeaturePyramid(nn.Module):

    def __init__(self, in_channels, out_channel, fpn_layer,
                 conv='conv', init='normal'):
        super(FeaturePyramid, self).__init__()
        # Do not use relu and batchnorm in feature pyramid
        #kernel_sizes = [25, 17, 9, 5, 3]
        for i in range(len(in_channels)):
            if conv == 'conv':
                self.add_module(
                    'lateral%d' % (i+3),
                    Conv2d(in_channels[i], out_channel, 1, use_relu=False, init=init))
            else:
                self.add_module(
                    'lateral%d' % (i+3),
                    LargeKernelConv2d(in_channels[i], out_channel, init=init))

        for i in range(len(in_channels)-1):
            self.add_module('adjust%d' % (i+3),
                            Conv2d(out_channel, out_channel, 3, use_relu=False, init=init))

        if fpn_layer > len(in_channels):
            self.p6 = Conv2d(in_channels[-1], out_channel, 3, stride=2,
                             use_relu=False, init=init)
            if fpn_layer > len(in_channels) + 1:
                self.p7 = Conv2d(out_channel, out_channel, 3,
                                 stride=2, use_relu=True, init=init)


    def forward(self, x):
        p = [getattr(self, 'lateral%d' % (i+3))(x[i]) for i in range(len(x))]

        for i in range(len(x)-2, -1, -1):
            size = p[i].size()
            resize = F.upsample(p[i+1], size=(size[2], size[3]), mode='bilinear')
            p[i] = resize + p[i]
            p[i] = getattr(self, 'adjust%d' % (i+3))(p[i])

        if hasattr(self, 'p6'):
            # For P6, use 3x3, stride 2 conv on C5
            p.append(self.p6(x[-1]))
        if hasattr(self, 'p7'):
            # Obtain p7 by relu and 3x3, stride 3 conv
            p.append(self.p7(p[-1]))

        return p


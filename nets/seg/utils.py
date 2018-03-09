#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-09-14 11:55:36
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common import Conv2d


class MaskHead(nn.Module):

    def __init__(self, num_classes, fpn_layer,
                 in_channel=256, out_channel=256):
        super(MaskHead, self).__init__()

        for i in range(fpn_layer):
            mask_convs = [
                Conv2d(in_channel, in_channel, 3, use_bn=True),
                Conv2d(in_channel, out_channel, 3, use_bn=True)
            ]
            self.add_module('conv%s' % (i+3), nn.Sequential(*mask_convs))

        self.cls_out = Conv2d(fpn_layer*out_channel, num_classes, 3, use_relu=False)

    def forward(self, x):
        p = [getattr(self, 'conv%s' % (i+3))(x[i]) for i in range(len(x))]

        out = [p[0]]
        size = p[0].size()
        for i in range(len(p)-1):
            resize = F.upsample(p[i+1], size=(size[2], size[3]), mode='bilinear')
            out.append(resize)

        out = torch.cat(out, dim=1)
        out = self.cls_out(out)
        size = out.size()
        out = F.upsample(out, size=(8*size[2], 8*size[3]), mode='bilinear')

        return out




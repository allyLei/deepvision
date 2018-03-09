#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-09-14 11:55:36
'''

import torch.nn as nn

from ..common import Conv2d

class HeadNetwork(nn.Module):

    def __init__(self, num_classes, num_anchors=9, num_channel=256,
                 dilation=1, use_bn=False,
                 use_residual=False, init='normal'):
        super(HeadNetwork, self).__init__()
        self.use_residual = use_residual
        cls_convs = [Conv2d(num_channel, num_channel, 3, use_bn=use_bn,
                          dilation=dilation, init=init) for i in range(4)]
        reg_convs = [Conv2d(num_channel, num_channel, 3, use_bn=use_bn,
                          dilation=dilation, init=init) for i in range(4)]
        self.cls_conv = nn.Sequential(*cls_convs)
        self.reg_conv = nn.Sequential(*reg_convs)

        self.cls_out = Conv2d(num_channel, num_anchors*num_classes, 3,
                              use_relu=False, init=init, init_bias=- 4.5)
        self.reg_out = Conv2d(num_channel, num_anchors*4, 3,
                              init=init, use_relu=False)

    def forward(self, x):
        _cls = x
        _cls = self.cls_conv(_cls)
        if self.use_residual:
            _cls = _cls + x
        logits = self.cls_out(_cls)

        _reg = x
        _reg = self.reg_conv(_reg)
        if self.use_residual:
            _reg = _reg + x
        preds = self.reg_out(_reg)
        return logits, preds



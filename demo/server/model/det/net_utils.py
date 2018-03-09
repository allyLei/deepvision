#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-09-14 11:55:36
'''

import torch.nn as nn
import torch.nn.functional as F


class Conv2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, use_relu=True, use_bn=False, padding=True,
                 init_std=0.01, init_bias=0.0):
        super(Conv2d, self).__init__()
        self.use_relu = use_relu
        self.use_bn = use_bn

        padding = int((kernel_size - 1) / 2) if padding else 0
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size, stride, padding=padding)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channel, momentum=0.001)
        if self.use_relu:
            self.relu = nn.ReLU(inplace=True)

        self.conv.weight.data.normal_(0, init_std)
        self.conv.bias.data.fill_(init_bias)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x


class FeaturePyramid(nn.Module):

    def __init__(self, fpn_layer, in_channels, out_channel):
        super(FeaturePyramid, self).__init__()
        self.fpn_layer = fpn_layer

        # Do not use relu and batchnorm in feature pyramid
        self.lateral3 = Conv2d(in_channels[0], out_channel, 1, use_relu=False)
        self.lateral4 = Conv2d(in_channels[1], out_channel, 1, use_relu=False)
        self.lateral5 = Conv2d(in_channels[2], out_channel, 1, use_relu=False)
        self.adjust3 = Conv2d(out_channel, out_channel, 3, use_relu=False)
        self.adjust4 = Conv2d(out_channel, out_channel, 3, use_relu=False)
        self.p6 = Conv2d(in_channels[-1], out_channel, 3, stride=2,
                         use_relu=False)
        if self.fpn_layer > len(in_channels):
            self.p7 = Conv2d(out_channel, out_channel, 3,
                             stride=2, use_relu=True)

    def forward(self, x):
        p = [self.lateral3(x[0]),
             self.lateral4(x[1]),
             self.lateral5(x[2])]

        for i in range(len(x)-2, -1, -1):
            size = p[i].size()
            resize = F.upsample(
                p[i + 1],
                size=(size[2],
                      size[3]),
                mode='bilinear')
            p[i] = resize + p[i]

        p[0] = self.adjust3(p[0])
        p[1] = self.adjust4(p[1])

        # For P6, use 3x3, stride 2 conv on C5
        p.append(self.p6(x[-1]))
        # Obtain p7 by relu and 3x3, stride 3 conv
        if len(p) < self.fpn_layer:
            p.append(self.p7(p[-1]))

        return p


class HeadNetwork(nn.Module):

    def __init__(self, num_classes, num_anchors=9, num_channel=256,
                 use_bn=False, use_residual=False, use_focal_loss=True):
        super(HeadNetwork, self).__init__()
        self.use_residual = use_residual
        cls_convs = []
        reg_convs = []
        for i in range(4):
            cls_conv = Conv2d(num_channel, num_channel, 3, use_bn=use_bn)
            reg_conv = Conv2d(num_channel, num_channel, 3, use_bn=use_bn)
            setattr(self, 'cls%d' % (i+1), cls_conv)
            setattr(self, 'reg%d' % (i+1), reg_conv)
            cls_convs.append(cls_conv)
            reg_convs.append(reg_conv)
        self.cls_conv = nn.Sequential(*cls_convs)
        self.reg_conv = nn.Sequential(*reg_convs)

        if use_focal_loss:
            self.cls_out = Conv2d(num_channel, num_anchors*num_classes, 3,
                                  use_relu=False, init_bias=- 4.5)
        else:
            self.cls_out = Conv2d(num_channel, num_anchors*num_classes, 3,
                                  use_relu=False, init_bias=0.0)
        self.reg_out = Conv2d(num_channel, num_anchors*4, 3, use_relu=False)

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
        return [logits, preds]


class MaskNetwork(nn.Module):

    def __init__(self, num_classes, num_channel=256,
                 use_bn=False, use_focal_loss=False,
                 init_std=0.01, init_bias=0.0):
        super(MaskNetwork, self).__init__()
        cls_convs = []
        reg_convs = []
        for i in range(3):
            cls_convs.append(
                Conv2d(
                    num_channel,
                    num_channel,
                    3,
                    use_bn=use_bn,
                    padding=False))
            reg_convs.append(
                Conv2d(
                    num_channel,
                    num_channel,
                    3,
                    use_bn=use_bn,
                    padding=False))
        self.cls_conv = nn.Sequential(*cls_convs)
        self.reg_conv = nn.Sequential(*reg_convs)

        mask_convs = []
        for i in range(4):
            mask_convs.append(
                Conv2d(
                    num_channel,
                    num_channel,
                    3,
                    use_bn=use_bn))
        self.mask_conv = nn.Sequential(*mask_convs)
        self.mask_deconv = nn.ConvTranspose2d(num_channel, num_channel, 2,
                                              stride=2, padding=0)

        self.cls_out = nn.Linear(num_channel, num_classes)
        self.reg_out = nn.Linear(num_channel, 4)
        self.mask_out = Conv2d(num_channel, 1, 1)

        # Initialize weights and bias
        self.cls_out.weight.data.normal_(0, init_std)
        self.reg_out.weight.data.normal_(0, init_std)
        self.mask_deconv.weight.data.normal_(0, init_std)
        self.cls_out.bias.data.fill_(init_bias)
        self.reg_out.bias.data.fill_(init_bias)
        self.mask_deconv.bias.data.fill_(init_bias)

    def forward(self, x):
        _x = F.max_pool2d(x, 2, 2)
        batch_size = _x.size(0)

        _cls = self.cls_conv(_x)
        logits = self.cls_out(_cls.view(batch_size, -1))

        _reg = self.reg_conv(_x)
        preds = self.reg_out(_reg.view(batch_size, -1))

        _mask = self.mask_conv(x)
        _mask = self.mask_deconv(_mask)
        masks = self.mask_out(_mask)
        masks = masks.squeeze(1)
        return [logits, preds, masks]

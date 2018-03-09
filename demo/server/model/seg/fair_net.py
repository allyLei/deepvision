#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Mail: sunlei@conew.com
Last modified: 2017-08-12 13:50
'''

import torch.nn as nn
import torch.nn.functional as F

from .. import contrib
from ..common import FeaturePyramid
from ..common import preprocessing
from .utils import MaskHead
from .postprocessing import postprocessing


class FairNet(nn.Module):

    def __init__(self, num_classes, backbone='resnet101',
                 groups=1, conv='conv', dilation=1, fpn_layer=3,
                 use_postprocessing=False):
        super(FairNet, self).__init__()
        self.num_classes = num_classes
        self.backbone_arch = backbone
        self.use_postprocessing = use_postprocessing
        num_channel = 256
        self.backbone = getattr(contrib, backbone)(groups=groups,
                                                   dilation=dilation,
                                                   conv=conv)
        in_channels = self.backbone.get_channels()
        self.fpn = FeaturePyramid(in_channels, num_channel,
                                  fpn_layer, conv=conv, init='kaiming')
        self.mask = MaskHead(num_classes, fpn_layer, num_channel,
                             deform=conv == 'deform')

    def forward(self, x, y=None, ws=None, hs=None):
        x = preprocessing(x, self.backbone_arch)
        _, feat_map = self.backbone(x)
        feat_map = self.fpn(feat_map)
        logits = self.mask(feat_map)

        if y is None:
            probs = F.softmax(logits, dim=1)
            _, preds = probs.topk(1, 1)
            preds = preds.squeeze(1)
            if not self.use_postprocessing:
                return preds
            return postprocessing(preds, ws, hs)
        else:
            y = y.long()
            return F.cross_entropy(logits, y, ignore_index=255)

#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Mail: sunlei@conew.com
Last modified: 2017-08-12 13:50
'''

import torch
import torch.nn as nn

from . import resnet
from .net_utils import FeaturePyramid, HeadNetwork
from .layer_utils import decode_bboxes
from ..common import preprocessing

class RetinaNet(nn.Module):

    def __init__(self, num_classes, num_anchors=9,
                 fpn_layer=4, backbone='resnet101',
                 use_bn=False, use_residual=False,
                 fix_backbone_bn=True, weight_stddev=0.01):
        super(RetinaNet, self).__init__()
        self.num_classes = num_classes
        self.backbone_arch = backbone
        num_channel = 256
        self.losses = []
        self.backbone = getattr(resnet, backbone)(False, fix_bn=fix_backbone_bn)
        in_channels = self.backbone.get_channels()
        self.fpn = FeaturePyramid(fpn_layer, in_channels, num_channel)
        self.head = HeadNetwork(num_classes, num_anchors, num_channel,
                                use_bn=use_bn, use_residual=use_residual,
                                use_focal_loss=True)

    def _concat(self, logits, preds):
        """ Transpose (N, C, H, W) to (N, H, W, C) and then reshape,
        to (N, -1, num_classes or 4), and concat them together
        """
        logits = list(map(lambda x: x.permute(0, 2, 3, 1).contiguous(), logits))
        preds = list(map(lambda x: x.permute(0, 2, 3, 1).contiguous(), preds))

        n = logits[0].size()[0]
        logits = list(map(lambda x: x.view(n, -1, self.num_classes), logits))
        preds = list(map(lambda x: x.view(n, -1, 4), preds))

        logits = torch.cat(logits, dim=1)
        preds = torch.cat(preds, dim=1)
        return logits, preds

    def forward(self, x, anchors):
        x = preprocessing(x, self.backbone_arch)
        _, feat_map = self.backbone(x)
        x = self.fpn(feat_map)

        logits = []
        preds = []
        for _x in x:
            _logits, _preds = self.head(_x)
            logits.append(_logits)
            preds.append(_preds)

        logits, preds = self._concat(logits, preds)

        probs = torch.sigmoid(logits)
        bboxes = decode_bboxes(preds, anchors)
        return probs, bboxes

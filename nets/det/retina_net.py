#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Mail: sunlei@conew.com
Last modified: 2017-08-12 13:50
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import contrib
from ..common import preprocessing
from ..common import FeaturePyramid
from .utils import HeadNetwork
from .loss import sigmoid_focal_loss
from .layers import decode_bboxes


class RetinaNet(nn.Module):

    def __init__(self, num_classes, backbone='resnet101',
                 groups=1, dilation=1, conv='conv', fpn_layer=4, num_anchors=9,
                 fix_backbone_bn=True, use_bn=False, use_residual=False):
        super(RetinaNet, self).__init__()
        self.num_classes = num_classes
        num_channel = 256
        self.backbone_arch = backbone
        self.backbone = getattr(contrib, backbone)(groups=groups,
                                                   fix_bn=fix_backbone_bn,
                                                   dilation=dilation)
        in_channels = self.backbone.get_channels()
        self.fpn = FeaturePyramid(in_channels, num_channel, fpn_layer, conv)
        self.head = HeadNetwork(num_classes, num_anchors, num_channel,
                                deform=conv=='deform', use_bn=use_bn,
                                use_residual=use_residual)

    def concat(self, logits, preds):
        """ Transpose (N, C, H, W) to (N, H, W, C) and then reshape,
        to (N, -1, num_classes or 4), and concat them together"""
        n = logits[0].size(0)
        logits = [x.permute(0,2,3,1).contiguous() for x in logits]
        preds = [x.permute(0,2,3,1).contiguous() for x in preds]
        logits = torch.cat([x.view(n,-1,self.num_classes) for x in logits], dim=1)
        preds = torch.cat([x.view(n,-1,4) for x in preds], dim=1)
        return logits, preds

    def forward(self, x, labels=None, targets=None, anchors=None):
        x = preprocessing(x, self.backbone_arch)
        _, feat_map = self.backbone(x)
        feat_map = self.fpn(feat_map)

        logits = []
        preds = []
        for x in feat_map:
            _logits, _preds = self.head(x)
            logits.append(_logits)
            preds.append(_preds)

        logits, preds = self.concat(logits, preds)

        if labels is None:
            probs = torch.sigmoid(logits)
            bboxes = decode_bboxes(preds, anchors)
            return probs, bboxes
        else:
            return self.loss(logits, preds, labels, targets)

    def loss(self, logits, preds, labels, targets):
        """ Focal losses for RetinaNet.
        Args:
            logits, [batch_size, total_anchors, num_classes]
            preds, [batch_size, total_anchors, 4]
            labels, [batch_size, total_anchors]
            targets, [batch_size, total_anchors, 4]
        """
        lvalid = labels != 255
        pvalid = ((labels > 0) & (labels != 255)).long()
        norm = max(pvalid.sum().data[0], 1.0)
        pvalid = pvalid.unsqueeze(2).expand_as(preds).byte()

        logits = logits[lvalid.long().unsqueeze(2).expand_as(logits).byte()]
        cls_loss = sigmoid_focal_loss(logits.view(-1, self.num_classes),
                                      labels[lvalid].long()) / norm
        pred_loss = F.smooth_l1_loss(preds[pvalid].view(-1, 4),
                                     targets[pvalid].view(-1, 4),
                                     size_average=False) / norm
        return cls_loss + pred_loss, cls_loss, pred_loss

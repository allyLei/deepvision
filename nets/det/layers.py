#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-11-07 22:56:14
'''

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def encode_bboxes(bboxes, anchors):
    """ Encode bboxes according to anchors by transforming
    bboxes to the form that suits regression network.
    """
    # Bboxes format [x1, y1, x2, y2]
    w_a = anchors[..., 2] - anchors[..., 0] + 1.0
    h_a = anchors[..., 3] - anchors[..., 1] + 1.0
    x_a = anchors[..., 0] + 0.5 * (w_a - 1.0)
    y_a = anchors[..., 1] + 0.5 * (h_a - 1.0)

    w_p = bboxes[..., 2] - bboxes[..., 0] + 1.0
    h_p = bboxes[..., 3] - bboxes[..., 1] + 1.0
    x_p = bboxes[..., 0] + 0.5 * (w_p - 1.0)
    y_p = bboxes[..., 1] + 0.5 * (h_p - 1.0)

    w_t = 5.0 * torch.log(w_p / w_a)
    h_t = 5.0 * torch.log(h_p / h_a)
    x_t = 10.0 * (x_p - x_a) / w_a
    y_t = 10.0 * (y_p - y_a) / h_a

    targets = torch.stack([x_t, y_t, w_t, h_t], dim=-1)
    return targets


def decode_bboxes(preds, anchors):
    """ Decode network regression preds into bboxes, according
    to referenced anchors.
    """
    w_a = anchors[..., 2] - anchors[..., 0] + 1.0
    h_a = anchors[..., 3] - anchors[..., 1] + 1.0
    x_a = anchors[..., 0] + 0.5 * w_a
    y_a = anchors[..., 1] + 0.5 * h_a

    w_p = torch.exp(preds[..., 2] / 5.0) * w_a
    h_p = torch.exp(preds[..., 3] / 5.0) * h_a
    x_p = (preds[..., 0] / 10.0 * w_a + x_a)
    y_p = (preds[..., 1] / 10.0 * h_a + y_a)

    x1 = x_p - 0.5 * (w_p - 1.0)
    y1 = y_p - 0.5 * (h_p - 1.0)
    x2 = x_p + 0.5 * (w_p - 1.0)
    y2 = y_p + 0.5 * (h_p - 1.0)

    bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
    return bboxes


def clip_bboxes(bboxes, width, height):
    bboxes = torch.stack(
        [bboxes[..., 0].clamp(0, width-1),
         bboxes[..., 1].clamp(0, height-1),
         bboxes[..., 2].clamp(0, width-1),
         bboxes[..., 3].clamp(0, height-1)
         ], dim=-1)
    return bboxes


def bbox_overlaps(bboxes, gt_bboxes):
    """ Compute overlaps between bboxes and gt_bboxes.
    Args:
        1. bboxes, [N, 4] tensor or variable
        2. gt_bboxes, [K, 4] tensor or variable
    Returns:
        overlaps [N, K] each item indicates the overlap ratio
    """
    bbox_areas = (bboxes[..., 2] - bboxes[..., 0] + 1) * \
        (bboxes[..., 3] - bboxes[..., 1] + 1)
    gt_bbox_areas = (gt_bboxes[..., 2] - gt_bboxes[..., 0] + 1) * \
        (gt_bboxes[..., 3] - gt_bboxes[..., 1] + 1)
    x1 = torch.max(bboxes[..., 0:1], gt_bboxes[..., 0:1].t())
    x2 = torch.min(bboxes[..., 2:3], gt_bboxes[..., 2:3].t())
    y1 = torch.max(bboxes[..., 1:2], gt_bboxes[..., 1:2].t())
    y2 = torch.min(bboxes[..., 3:4], gt_bboxes[..., 3:4].t())
    iw = (x2 - x1 + 1).clamp(min=0)
    ih = (y2 - y1 + 1).clamp(min=0)
    ua = bbox_areas.view(-1, 1) + gt_bbox_areas.view(1, -1) - iw * ih
    return iw * ih / ua


def crop_pool(feat_map, bboxes, stride=8, pool_size=14):
    bboxes = bboxes.detach()
    x1 = bboxes[..., 0:1] / stride
    y1 = bboxes[..., 1:2] / stride
    x2 = bboxes[..., 2:3] / stride
    y2 = bboxes[..., 3:4] / stride

    n = bboxes.size()[0]
    c, h, w = feat_map.size(1), feat_map.size(2), feat_map.size(3)
    theta = Variable(bboxes.data.new(n, 2, 3).zero_())
    theta[..., 0, 0] = (x2 - x1) / (w - 1)
    theta[..., 0, 2] = (x1 + x2 - w + 1) / (w - 1)
    theta[..., 1, 1] = (y2 - y1) / (h - 1)
    theta[..., 1, 2] = (y1 + y2 - h + 1) / (h - 1)

    grid = F.affine_grid(theta, torch.Size((n, 1, pool_size, pool_size)))
    crops = F.grid_sample(feat_map.expand(n, c, h, w), grid)
    return crops

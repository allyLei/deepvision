#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-11-17 17:10:21
'''

import numpy as np


def encode_bboxes(anchors, bboxes):
    """ According to anchors, do transform the bboxes,
        refer to paper 'Faster R-CNN: Towards Real-Time Object
        Detection with Region Proposal Networks' for details.
    """
    # x1, y1, x2, y2
    w_a = anchors[:, 2] - anchors[:, 0] + 1.0
    h_a = anchors[:, 3] - anchors[:, 1] + 1.0
    x_a = anchors[:, 0] + 0.5 * (w_a - 1.0)
    y_a = anchors[:, 1] + 0.5 * (h_a - 1.0)

    w_p = bboxes[:, 2] - bboxes[:, 0] + 1.0
    h_p = bboxes[:, 3] - bboxes[:, 1] + 1.0
    x_p = bboxes[:, 0] + 0.5 * (w_p - 1.0)
    y_p = bboxes[:, 1] + 0.5 * (h_p - 1.0)

    w_t = 5.0 * np.log(w_p / w_a)
    h_t = 5.0 * np.log(h_p / h_a)
    x_t = 10.0 * (x_p - x_a) / w_a
    y_t = 10.0 * (y_p - y_a) / h_a

    return np.vstack((x_t, y_t, w_t, h_t)).transpose()


def decode_bboxes(anchors, bboxes):
    """ According to anchors, convert the tranformed bboxes into normal bbox.
        This is an inverse of bbox_transform.
    """
    # x1, y1, x2, y2
    w_a = anchors[:, 2] - anchors[:, 0] + 1.0
    h_a = anchors[:, 3] - anchors[:, 1] + 1.0
    x_a = anchors[:, 0] + 0.5 * w_a
    y_a = anchors[:, 1] + 0.5 * h_a

    w_p = np.exp(bboxes[:, 2] / 5.0) * w_a
    h_p = np.exp(bboxes[:, 3] / 5.0) * h_a
    x_p = (bboxes[:, 0] / 10.0 * w_a + x_a)
    y_p = (bboxes[:, 1] / 10.0 * h_a + y_a)

    x1 = x_p - 0.5 * (w_p - 1.0)
    y1 = y_p - 0.5 * (h_p - 1.0)
    x2 = x_p + 0.5 * (w_p - 1.0)
    y2 = y_p + 0.5 * (h_p - 1.0)
    return np.vstack((x1, y1, x2, y2)).transpose()


def clip_bboxes(dets, width, height):
    dets[:, 0] = np.minimum(np.maximum(dets[:, 0], 0.0), width)
    dets[:, 1] = np.minimum(np.maximum(dets[:, 1], 0.0), height)
    dets[:, 2] = np.minimum(np.maximum(dets[:, 2], 0.0), width)
    dets[:, 3] = np.minimum(np.maximum(dets[:, 3], 0.0), height)
    return dets


def resize_bboxes(dets, ratio_x, ratio_y):
    dets[:, 0] = ratio_x * dets[:, 0]
    dets[:, 1] = ratio_y * dets[:, 1]
    dets[:, 2] = ratio_x * dets[:, 2]
    dets[:, 3] = ratio_y * dets[:, 3]
    return dets


def bbox_overlaps(bboxes, gt_bboxes):
    """ Compute overlaps between bboxes and gt_bboxes.
    Args:
        1. bboxes, [N, 4] numpy array
        2. gt_bboxes, [K, 4] numpy array
    Returns:
        overlaps [N, K] each item indicates the overlap ratio
    """
    bbox_areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * \
        (bboxes[:, 3] - bboxes[:, 1] + 1)
    gt_bbox_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * \
        (gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)

    x1 = np.maximum(bboxes[:, 0:1], np.transpose(gt_bboxes[:, 0:1]))
    x2 = np.minimum(bboxes[:, 2:3], np.transpose(gt_bboxes[:, 2:3]))
    y1 = np.maximum(bboxes[:, 1:2], np.transpose(gt_bboxes[:, 1:2]))
    y2 = np.minimum(bboxes[:, 3:4], np.transpose(gt_bboxes[:, 3:4]))
    iw = np.maximum(x2 - x1 + 1, 0)
    ih = np.maximum(y2 - y1 + 1, 0)
    ua = bbox_areas.reshape(-1, 1) + gt_bbox_areas.reshape(1, -1) - iw * ih
    return iw * ih / ua

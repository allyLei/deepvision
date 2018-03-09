#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-09-14 22:10:18
'''

import numpy as np
from .bbox import encode_bboxes
from ..nms.cython_bbox import bbox_overlaps, bbox_overlaps_self

def filter_border(width, height, anchors, border_ratio):
    # Ignore anchors that are cross borders
    border = border_ratio * 128
    select = np.where(
            (anchors[:, 0] >= -border) &
            (anchors[:, 1] >= -border) &
            (anchors[:, 2] <= width + border) &
            (anchors[:, 3] <= height + border))[0]
    anchors = anchors[select, :]
    return anchors, select


def gen_samples(opt, bboxes, anchors, crowd_indexes):
    if bboxes.shape[0] == 0:
        n = anchors.shape[0]
        gt_labels = np.zeros(n, dtype=np.int32)
        gt_bboxes = np.zeros([n, 4], dtype=np.float32)
        fg_indexes = np.zeros(0, dtype=np.int32)
        bg_indexes = np.asarray(list(set(range(n)) -
                                     set(crowd_indexes))).astype(np.int32)
        return gt_labels, gt_bboxes, fg_indexes, bg_indexes

    # Generate positive anchors and negative anchors
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(bboxes[:, :4], dtype=np.float))  # shape [n1, M]
    argmax_overlaps = np.argmax(overlaps, axis=1)
    max_overlaps = overlaps[np.arange(
        overlaps.shape[0]), argmax_overlaps]  # shape [n1]
    gt_labels = bboxes[argmax_overlaps, 4].astype(np.int32)
    gt_bboxes = bboxes[argmax_overlaps, : 4]

    fg_indexes = set(np.append(
        np.argmax(overlaps, axis=0),
        np.where(max_overlaps >= opt.rpn_pos_overlap)[0]))
    tmp = set(np.where(max_overlaps < opt.rpn_low_neg_overlap)[0])
    fg_indexes = fg_indexes - tmp
    bg_indexes = set(np.where(max_overlaps < opt.rpn_high_neg_overlap)[0])
    bg_indexes = bg_indexes - fg_indexes - set(crowd_indexes)
    fg_indexes = np.asarray(list(fg_indexes)).astype(np.int32)
    bg_indexes = np.asarray(list(bg_indexes)).astype(np.int32)
    return gt_labels, gt_bboxes, fg_indexes, bg_indexes

def remove_crowd(opt, bboxes, anchors):
    # Remove crowd anchors
    if bboxes.shape[0] == 0:
        return np.zeros(0, dtype=np.int32)

    overlaps = bbox_overlaps_self(
        np.ascontiguousarray(bboxes[:, :4], dtype=np.float),
        np.ascontiguousarray(anchors, dtype=np.float))
    max_overlaps = np.max(overlaps, axis=0)
    return np.where(max_overlaps >= opt.crowd_overlap)[0]

def sample_selection(opt, fg_indexes, bg_indexes):
    # TODO: Online Hard Negative Example Mining
    if opt.sample_strategy == 'keep_3to1':
        # Keep all positive anchors, and keep pos: neg = 1:3
        n_bg_anchors = 3 * fg_indexes.shape[0]
        if n_bg_anchors > 0 and bg_indexes.shape[0] > n_bg_anchors:
            bg_indexes = np.random.choice(
                bg_indexes, size=n_bg_anchors, replace=False)
            '''
            if random.uniform(0, 1) > 0.5:
                # Hard exampling mining
                ind = np.argsort(max_overlaps[bg_indexes])[-n_bg_anchors:]
                bg_indexes = bg_indexes[ind]
            else:
                # Random sampling
                bg_indexes = np.random.choice(
    bg_indexes, size=n_bg_anchors, replace=False)
            '''


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret

def gen_anchor_targets(opt, width, height, bboxes, anchors):
    """ Sample anchors according to pos/neg ratio and bboxes.
    Args:
        opt: global training options
        width: image width
        height: image height
        bboxes: [N, 5], (x1,y1,x2,y2,cat_id), cat_id=-1 means invalid
        anchors: [S, 4], S=h*w*num_anchors_per_pixel
    Returns:
        labels: [S], S=h*w*num_anchors_per_pixel
        targets: [S, 4]
    """
    total_anchors = anchors
    anchors, select = filter_border(width, height, anchors, opt.border_ratio)

    # Crowd bboxes
    invalid = np.where(bboxes[:, 4] < 0)[0]
    crowd = bboxes[invalid, :]
    crowd_indexes = remove_crowd(opt, crowd, anchors)

    # Generate positive and negative samples
    valid = np.where(bboxes[:, 4] >= 0)[0]
    bboxes = bboxes[valid]
    gt_labels, gt_bboxes, fg_indexes, bg_indexes = gen_samples(
        opt, bboxes, anchors, crowd_indexes)

    # Sample selection, e.g. hard negative mining
    # bg_indexes = sample_selection(opt, fg_indexes, bg_indexes)

    labels = np.ones(anchors.shape[0], dtype=np.uint8) * 255
    labels[fg_indexes] = gt_labels[fg_indexes]
    labels[bg_indexes] = 0
    targets = np.zeros((anchors.shape[0], 4), dtype=np.float32)
    targets = encode_bboxes(anchors, gt_bboxes).astype(np.float32, copy=False)

    # Unmap to all anchors
    n = total_anchors.shape[0]
    labels = _unmap(labels, n, select, fill=255).astype(np.uint8)
    targets = _unmap(targets, n, select, fill=0).astype(np.float32)
    return labels, targets

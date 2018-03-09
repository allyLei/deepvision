#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-08 14:38:01
'''

import numpy as np

from libs.det.nms.nms_wrapper import nms
from libs.det.anchor.bbox import resize_bboxes, clip_bboxes


def nms_per_class(probs, bboxes, nw, nh, w, h, cls_thresh=0.05, nms_thresh=0.5):
    ret = []
    num_classes = probs.shape[-1]
    for i in range(num_classes):
        dets_per_class = np.hstack([bboxes, probs[:,i:i+1]])
        select = np.where(dets_per_class[:,4] > cls_thresh)[0]
        dets_per_class = dets_per_class[select, :]

        select = np.argsort(dets_per_class[:,4])[-100:]
        dets_per_class = dets_per_class[select, :].astype(np.float32)

        nms_indexes = nms(dets_per_class, nms_thresh)
        dets_per_class = dets_per_class[nms_indexes]
        labels = np.array([i+1] * dets_per_class.shape[0])
        ret.append(np.hstack([dets_per_class, labels[:, np.newaxis]]))

    if len(ret) == 0:
        return np.zeros([0, 6], dtype=np.float32)
    dets = np.vstack(ret)
    order = np.argsort(dets[:, 4])[::-1]
    ratio_x, ratio_y = nw * 1. / w, nh * 1. / h
    dets = resize_bboxes(dets[order, :][:100], ratio_x, ratio_y)
    dets = clip_bboxes(dets, nw, nh)
    return dets


def postprocessing(probs, bboxes, w, h, ws, hs, cls_thresh=0.05, nms_thresh=0.5):
    ''' Do nms and resize to original image space.
    Args:
        probs, [batch_size, total_anchors, num_classes]
        bboxes, [batch_size, total_anchors, 4]
        w, int, inference image width
        h, int, infernece image height
        ws, original image widths
        hs, original image heights
    Returns:
        a list of dets, which is of size [n, 5] (x1, y1, x2, y2, prob, cls)
    '''
    probs = probs.cpu().data.numpy()
    bboxes = bboxes.cpu().data.numpy()

    rets = []
    for i in range(probs.shape[0]):
        rets.append(nms_per_class(probs[i], bboxes[i], ws[i], hs[i], w=w, h=h,
                                  cls_thresh=cls_thresh, nms_thresh=nms_thresh))

    return list(rets)







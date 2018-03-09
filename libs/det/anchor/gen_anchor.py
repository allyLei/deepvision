#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Mail: sunlei@conew.com
Last modified: 2017-07-31 21:55
'''


import math
import numpy as np


def _enum_ratios(base_anchor, ratios):
    w, h, cx, cy = _get_size(base_anchor)
    sizes = w * h / ratios
    ws = np.sqrt(sizes)
    hs = ws * ratios
    anchors = _mkanchors(ws, hs, cx, cy)
    return anchors


def _enum_scales(base_anchor, scales):
    w, h, cx, cy = _get_size(base_anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, cx, cy)
    return anchors


def _get_size(anchor):
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    cx = anchor[0] + 0.5 * (w - 1)
    cy = anchor[1] + 0.5 * (h - 1)
    return w, h, cx, cy


def _mkanchors(ws, hs, cx, cy):
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((cx - 0.5 * (ws - 1),
                         cy - 0.5 * (hs - 1),
                         cx + 0.5 * (ws - 1),
                         cy + 0.5 * (hs - 1)))
    return anchors


def _gen_anchors(base_size=16, ratios=np.array([0.5, 1, 2]),
                 scales=2**np.arange(3, 6)):
    """ Generate anchors by enumerating aspect ratio and scales. """
    base_size = 0.5 * (base_size - 1.0)
    base_anchor = np.array([-base_size, -base_size, base_size, base_size], dtype=np.float32)
    base_anchor = np.array([0, 0, 15, 15], dtype=np.float32)
    ratio_anchors = _enum_ratios(base_anchor, ratios)
    #print('ratio_anchors: %s' % ratio_anchors)
    anchors = np.vstack([_enum_scales(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    return anchors


def gen_anchors(height, width, feat_stride,
                anchor_scales=[8, 16, 32], anchor_ratios=[0.5, 1, 2]):
    """ Generate anchors for an intermediate feature map layer of size
        [height, width], with anchor scales and anchor ratios. Note that
        height, width is not the image size, but anchors are generated
        on the true image size.
    """
    # shape [n_scales*n_ratios, 4]
    anchors = _gen_anchors(ratios=np.array(anchor_ratios),
                           scales=np.array(anchor_scales))

    shift_x = np.arange(0, width) * feat_stride
    shift_y = np.arange(0, height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # [height*width, 4]
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    anchors = anchors.reshape((1, anchors.shape[0], 4)) + \
        shifts.reshape((1, shifts.shape[0], 4)).transpose((1, 0, 2))
    # shape [height*width*(n_scales*n_ratios), 4]
    anchors = anchors.reshape((-1, 4)).astype(np.float32, copy=False)
    return anchors


def gen_anchor_info():
    feat_strides = [8, 16, 32, 64, 128]
    anchor_ratios = [0.5, 1.0, 2.0]

    _scales = [0.0, 1.0, 2.0]
    _scales = [-2.0, -1.0, 0.0]

    anchor_scales = []
    base_scales = [2, 4, 8, 16, 32]
    for scale in base_scales:
        anchors = list(map(lambda x: scale * (2 ** (x/3.0)), _scales))
        anchor_scales.append(anchors)

    #_anchor_scales = map(lambda x: map(lambda y: 16*y, x), anchor_scales)
    #print('anchor scales: %s' % _anchor_scales)
    #print('anchor ratios: %s' % anchor_ratios)

    return anchor_ratios, anchor_scales, feat_strides


def gen_anchors_list(w, h):
    """ Multiple feature layer anchor generation"""
    anchor_ratios, anchor_scales, feat_strides = gen_anchor_info()
    image_scales = []
    for feat_stride in feat_strides:
        _w = math.ceil(1.0 * w / feat_stride)
        _h = math.ceil(1.0 * h / feat_stride)
        if min(_w, _h) >= 4:
            image_scales.append((_h, _w))

    #print('image size: %d,%d, image_scales: %s' % (h, w, image_scales))

    anchors_list = []
    for i, image_scale in enumerate(image_scales):
        anchors = gen_anchors(
            image_scale[0], image_scale[1], feat_strides[i],
            anchor_scales=anchor_scales[i], anchor_ratios=anchor_ratios)
        anchors_list.append(anchors)
        #print('image size: %d, %d, image_scale: %s, anchor size: %d' %
        #      (h, w, image_scale, len(anchors)))

    '''
    anchors = anchors_list[-1]
    #print('last layer anchors: %s' % anchors_list[-1])
    border = 0.2 * feat_strides[-1]
    print('border: %s' % border)
    select = np.where((anchors[:, 0] >= -border) &
                      (anchors[:, 1] >= -border) &
                      (anchors[:, 2] <= w + border) &
                      (anchors[:, 3] <= h + border))[0]
    anchors = anchors[select, :]
    print('last layer valid anchors: %s' % anchors.shape[0])
    print('last layer anchors: %s' % anchors)
    '''

    return anchors_list


if __name__ == '__main__':
    image_sizes = [[300, 300], [300, 400], [400, 300]]
    for image_size in image_sizes:
        ret = gen_anchors_list(image_size[0], image_size[1])
        anchors = np.vstack(ret)
        print('image size: {}, anchors: {}'.format(image_size, anchors.shape))


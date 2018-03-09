#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-03-07 13:40:01
'''

import numpy as np
from PIL import Image
import torch
import collections
from torch.autograd import Variable

# ----------------------------
# np utils
# ----------------------------

def imresize(array, shape, mode=Image.BILINEAR,
             is_binary=False, rotate=None):
    if is_binary:
        array = 255 * array
    im = Image.fromarray(np.uint8(array))
    im = im.resize(shape, mode)
    if rotate is not None:
        im = im.rotate(rotate, expand=True)
    array = np.asarray(im, dtype=np.float32)
    if is_binary:
        array = array / 255.0
    return array


# ---------------------------
# torch utils
# ---------------------------

def to_cuda(x):
    if isinstance(x, np.ndarray):
        return Variable(
            torch.from_numpy(x),
            requires_grad=False).cuda(
            async=True)
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.LongTensor) or\
            isinstance(x, torch.IntTensor) or isinstance(x, torch.DoubleTensor) \
            or isinstance(x, torch.ByteTensor):
        return Variable(x).cuda(async=True)
    elif isinstance(x, Variable):
        return x.cuda(async=True)
    elif isinstance(x, list) or isinstance(x, tuple):
        y = []
        for i, e in enumerate(x):
            y.append(to_cuda(e))
        return y
    elif isinstance(x, dict):
        y = dict(map(lambda _x: (_x[0], to_cuda(_x[1])), x.items()))
        return y
    else:
        raise TypeError



# -------------------------
# clothing specific utils
# -------------------------

def _smooth_func(mask, count, classes):
    max_count = 0
    max_id = None
    for _class in classes:
        if count[_class] > max_count:
            max_id = _class
            max_count = count[_class]
    if max_id is not None:
        y = np.zeros(mask.shape, dtype=np.bool)
        for _class in classes:
            y = y | (mask == _class)
        select = np.where(y)
        mask[select] = max_id

def smooth(mask):
    """ Smooth our 13 classes segmentation. """
    count = collections.defaultdict(int)
    classes = np.unique(mask)
    for _class in classes:
        if _class == 0:
            continue
        # Filter out small converage area
        select = (mask == _class)
        count[_class] = select.sum()
        if count[_class] <= 100:
            mask[select] = 0
            count[_class] = 0

    # Smooth skirt and dress
    _smooth_func(mask, count, [5, 6])
    # Smooth outer bottom
    if count[7] > 0 and (count[5] > 0 or count[6] > 0):
        if count[8] > 0:
            _smooth_func(mask, count, [7, 8])
        else:
            _smooth_func(mask, count, [5, 6, 7])

    return mask


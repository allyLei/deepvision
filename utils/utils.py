#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-08 14:19:17
'''

import io
import numpy as np
from PIL import Image


# ================================
# Numpy utils
# ================================

def encode_array(array):
    """ Encode numpy array (dtype=np.uint8) to image buffer. """

    def _encode(x):
        im = Image.fromarray(x)
        stream = io.BytesIO()
        im.save(stream, 'JPEG')
        image_buffer = stream.getvalue()
        return image_buffer

    if len(array.shape) == 2:
        return _encode(array)
    elif len(array.shape) == 3:
        return [_encode(x), for x in list(array)]
    else:
        raise ValueError(f'array shape: {array.shape} not supported!!!')

def decode_array(image_buffer):
    """ Decode image buffer to numpy array (dtype=np.uint8)"""

    def _decode(x):
        im = Image.open(io.BytesIO(x))
        return np.array(im)

    if isinstance(image_buffer, str):
        return _decode(image_buffer)
    elif isinstance(image_buffer, list):
        return np.array([_decode(x) for x in image_buffer])
    else:
        raise TypeError(f'type {type(image_buffer)} not supported!!!')

def padding_array(x_list):
    """ Padding a list of non-constance [m0, m1, m2] to
    [batch_size, max(m0), m1, m2]
    """
    m = max([x.shape[0] for x in x_list])
    size = [len(x_list), m] + list(x_list[0].shape[1:])
    ret = np.zeros(size, dtype=x_list[0].dtype)
    for i, x in enumerate(x_list):
        _m = x.shape[0]
        ret[i, :_m] = x
    return ret

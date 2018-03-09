#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-02 13:36:32
'''

import random
import numpy as np
from PIL import Image, ImageEnhance

def adjust_brightness(img, brightness_factor):
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness_factor)
    return img

def adjust_contrast(img, contrast_factor):
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(contrast_factor)
    return img

def adjust_saturation(img, saturation_factor):
    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(saturation_factor)
    return img

def adjust_hue(img, hue_factor):
    if not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    input_mode = img.mode
    if input_mode in {'L', '1', 'I', 'F'}:
        return img

    h, s, v = img.convert('HSV').split()

    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over='ignore'):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, 'L')

    img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    return img

def crop(image, box, fill=0):
    ''' Crop image, box [x, y, w, h]. Note that x and y can be
    negative. '''
    h, w = image.shape[0], image.shape[1]
    rx, ry, rw, rh = box
    ux, uy = min(rx, 0), min(ry, 0)
    uw, uh = max(rx+rw, w) - ux, max(ry+rh, h) - uy
    rx -= ux
    ry -= uy

    shape = list(image.shape)
    shape[0], shape[1] = uh, uw
    ret = np.ones(shape, image.dtype) * fill
    ret[-uy:h-uy, -ux:w-ux, ...] = image
    ret = ret[ry:ry+rh, rx:rx+rw, ...]
    return ret

def random_crop(image, scale, fill=0):
    ''' Random crop an image to a scale. '''
    if image is None:
        return
    h, w = image.shape[0], image.shape[1]
    rangex, rangey = [0, w-scale[0]], [0, h-scale[1]]
    if rangex[0] > rangex[1]:
        rangex = rangex[::-1]
    if rangey[0] > rangey[1]:
        rangey = rangey[::-1]
    x, y = random.randint(*rangex), random.randint(*rangey)
    return crop(image, [x, y, scale[0], scale[1]], fill), \
        [x, y, scale[0], scale[1]]


def resize(image, scale, interpolation=Image.BILINEAR, by_short_side=False):
    ''' Resize an image or a mask into a fix scale: [w, h]. '''
    if image is None:
        return

    if by_short_side:
        h, w = image.shape[0], image.shape[1]
        if h > w:
            scale = (scale, int(1.0 * h * scale / w))
        else:
            scale = (int(1.0 * w * scale / h), scale)

    def _resize(im, scale, interpolation):
        im = Image.fromarray(im)
        im = im.resize(scale, interpolation)
        return np.array(im)

    if interpolation == Image.NEAREST and len(image.shape) > 2:
        ret = [_resize(image[:, :, i], scale, interpolation)
               for i in range(image.shape[2])]
        if len(ret) > 0:
            return np.stack(ret, axis=-1)
        else:
            return np.zeros((scale[1], scale[0], 0), image.dtype)
    else:
        return _resize(image, scale, interpolation)

def resize_bbox(bbox, ratio):
    ''' Resize bbox [x1, y1, x2, y2] according to ratio [ratio_x, ratio_y]. '''
    if bbox is None:
        return
    ret = np.copy(bbox)
    ret[...,0] = bbox[...,0] * ratio[0]
    ret[...,1] = bbox[...,1] * ratio[1]
    ret[...,2] = bbox[...,2] * ratio[0]
    ret[...,3] = bbox[...,3] * ratio[1]
    return ret

def to_bbox(mask, bbox=None):
    ''' Gen bbox according to mask. '''
    if mask is None:
        return

    def _to_bbox(mask):
        indexes = np.where(mask > 0)
        if len(indexes[0]) > 0 and len(indexes[1]) > 0:
            y1, y2 = min(indexes[0]), max(indexes[0])
            x1, x2 = min(indexes[1]), max(indexes[1])
            return np.array([x1, y1, x2, y2], dtype=np.float32)
        else:
            return np.array([0, 0, 0, 0], dtype=np.float32)


    if len(mask.shape) > 2:
        ret = [_to_bbox(mask[:,:,i]) for i in range(mask.shape[2])]
        if len(ret) > 0:
            ret = np.stack(ret, axis=0)
        else:
            ret = np.zeros((0, 4), dtype=np.float32)
    else:
        ret = _to_bbox(mask)

    if bbox is not None:
        ret = np.hstack([ret, bbox[:,4:5]])
    return ret


def valid_seg(mask, tmask):
    #area = ((mask > 0) * (mask < 255)).sum()
    #tarea = ((tmask > 0) * (tmask < 255)).sum()
    #if tarea / area > 0.2:
    #    return True
    #return False
    return True

def valid_det(mask, tmask):
    # TODO
    return True

def transpose(image):
    return np.ascontiguousarray(image.transpose([2,0,1]))


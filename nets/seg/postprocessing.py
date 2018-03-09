#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-08 13:59:52
'''

import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor

def resize(pred, w, h):
    im = Image.fromarray(pred)
    mask = np.asarray(im.resize([w, h], Image.NEAREST)).astype(np.uint8)
    return mask

def postprocessing(preds, ws, hs):
    ''' Convert model predictions into segmentation of original sizes.
    Args:
        preds, model prediction of size [b, h, w]
        hs, original image heights [b]
        ws, original image widths [b]
    Returns:
        a list of numpy mask
    '''
    preds = preds.cpu().data.numpy().astype(np.uint8)
    with ProcessPoolExecutor(max_workers=50) as executor:
        rets = executor.map(resize, preds, ws, hs)

    return list(rets)



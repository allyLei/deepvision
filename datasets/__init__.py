#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-12-28 17:45:31
'''

import sys
import torch
from . import imagenet, voc, coco, clothing, gan
from .dataloader import DataLoader

def gen_dataloader(opt, train):
    try:
        module = sys.modules[__name__]
        Dataset = getattr(module, opt.dataset).Dataset
        dataset = Dataset(opt, train)
    except Exception as e:
        raise ImportError(f'msg: {e}')

    num_workers = opt.num_workers if train else 1
    return DataLoader(dataset,
                      batch_size=opt.batch_size,
                      shuffle=train,
                      num_workers=num_workers,
                      train=train)



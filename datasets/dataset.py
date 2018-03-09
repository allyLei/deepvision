#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-12-28 17:32:36
'''

import random
import logging
import collections

import torch
from torch.utils.data.dataloader import default_collate

from libs.visualizer import vis

class Dataset(object):

    def __init__(self, opt, train=True, scale_idx=0):
        """
        A base dataset class.
        Arguments:
            opt: options indicated by user
            train: training mode or validation mode
        """
        super(Dataset, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.opt = opt
        self.train = train
        self.vis = 0
        self.vis_idx = vis.gen_idx()
        self.read()
        self.gen_transforms()

    def read(self):
        raise NotImplementedError

    def gen_transforms(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.items)




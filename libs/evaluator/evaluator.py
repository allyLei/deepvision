#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-08 13:43:37
'''

import os
import logging
import torch

from ..trainer.utils import load
from libs.visualizer import vis

class Evaluator(object):

    def __init__(self, opt, dataloader):
        self.logger = logging.getLogger('eval')
        self.opt = opt
        self.dataloader = dataloader
        self.vis_idx = vis.gen_idx()

    def init(self, model):
        # Model parallel
        model = torch.nn.DataParallel(model).cuda()
        # Load model
        if not os.path.exists(self.opt.pretrained):
            self.logger.error(f'ckpt {self.opt.pretrained} not found!!!')
        load(model, self.opt.pretrained, self.opt.exclude)
        return model

    def evaluate(self, model):
        raise NotImplementedError


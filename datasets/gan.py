#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-12-28 17:32:36
'''

import os
import json
from PIL import Image
import numpy as np

from . import dataset
from .transforms import transforms as transform
from .transforms import functional as F

from libs.visualizer import vis, vis_util


class Dataset(dataset.Dataset):

    def read(self):
        self.items = []
        data_dir = os.path.join(self.opt.data_dir, self.opt.mode)
        for file_name in os.listdir(data_dir):
            file_name = os.path.join(data_dir, file_name)
            self.items.append(file_name)
        self.logger.info(f'read total {len(self.items)} images')

    def gen_transforms(self):
        sampler = transform.Sampler(self.opt.scales)
        if self.train:
            transforms = [
                transform.ResizeAndCrop(scale=self.opt.short_side_scale,
                                        ratio=[1.0, 1.0],
                                        scale_sampler=sampler,
                                        valid_func=F.valid_seg,
                                        mode=[Image.BICUBIC, Image.BICUBIC]),
                transform.RandomHorizontalFlip()
            ]
        else:
            transforms = [transform.Resize(sampler)]
        self.transforms = transform.Compose(transforms)


    def __getitem__(self, index):
        file_name = self.items[index]
        concat = np.array(Image.open(file_name).convert('RGB'))
        h, w, _ = concat.shape
        A, B = concat[:, w/2, :], concat[:, w/2, :]
        if self.opt.reverse_AB:
            A, B = B, A
        A, B = self.transforms(A, B)
        if self.opt.input_nc == 1:
            A = A[...,0] * 0.299 + A[...,1] * 0.587 + A[...,2] * 0.114
            A = A[...,np.newaxis]
        if self.opt.output_nc == 1:
            B = B[...,0] * 0.299 + B[...,1] * 0.587 + B[...,2] * 0.114
            B = B[...,np.newaxis]
        return F.transpose(A), F.transpose(B)


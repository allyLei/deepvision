#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-12-28 23:29:51
'''

import os
import json
import numpy as np
from PIL import Image
import torchvision

from . import dataset
from .transforms import transforms as transform
from .transforms import functional as F


class Dataset(dataset.Dataset):

    def read(self):
        self.items = []
        self.gen_label_mapping()
        mode = 'train' if self.train else 'val'
        image_dir = os.path.join(self.opt.data_dir, mode)
        for sub_dir in os.listdir(image_dir):
            assert sub_dir in self.reverse_label_mapping, \
                f'sub dir {sub_dir} must in label mapping'
            label = self.reverse_label_mapping[sub_dir]
            self.logger.debug(f'read {sub_dir}, readed {len(self.items)} images')
            sub_dir = os.path.join(image_dir, sub_dir)
            for file_name in os.listdir(sub_dir):
                file_name = os.path.join(sub_dir, file_name)
                self.items.append([file_name, label])

        self.logger.info(f'total read {len(self.items)} images for {mode}')

    def gen_label_mapping(self):
        file_name = os.path.join(self.opt.data_dir, 'human_readable_file.json')
        doc = json.load(open(file_name))
        self.reverse_label_mapping = {}
        for item in doc:
            self.reverse_label_mapping[item['name']] = item['id']

    def gen_transforms(self):
        sampler = transform.Sampler(self.opt.scales)
        if self.train:
            transforms = [
                transform.CropAndResize(scale_sampler=sampler),
                transform.ColorJitter()
            ]
        else:
            transforms = [transform.Resize(scale_sampler=sampler)]
        self.transforms = torchvision.transforms.transforms.Compose(transforms)

    def __getitem__(self, index):
        file_name, label = self.items[index]
        image = np.array(Image.open(file_name).convert('RGB'))
        image = self.transforms(image)
        return F.transpose(image), label


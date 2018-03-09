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
        if self.opt.use_aug:
            file_name = 'train_aug.txt' if self.train else 'val.txt'
        else:
            file_name = 'train.txt' if self.train else 'val.txt'
        items = map(lambda line: line.strip(),
                    open(os.path.join(self.opt.data_dir, file_name)))
        items = list(map(lambda line: [
            os.path.join(self.opt.data_dir, line.split(' ')[0][1:]),
            os.path.join(self.opt.data_dir, line.split(' ')[1][1:])
            ], items))
        self.items = list(items)
        self.logger.info('total read {} images from {}' .format(
            len(self.items), 'train' if self.train else 'val'))

        self.readable_label_mapping = {}
        file_name = os.path.join(self.opt.data_dir, 'label_mapping.json')
        doc = map(lambda line: json.loads(line.strip()), open(file_name))
        for item in doc:
            self.readable_label_mapping[int(item['id'])] = item['name']

    def gen_transforms(self):
        sampler = transform.Sampler(self.opt.scales)
        if self.train:
            transforms = [
                transform.Select(transform.ColorJitter(0.1,0.1,0.1,0.05), index=0),
                transform.RandomChoice([
                transform.Resize(sampler),
                transform.ResizeAndCrop(scale_sampler=sampler,
                                        valid_func=F.valid_seg, fill=255)]),
                #transform.CropAndResize(scale_sampler=sampler,
                #                        valid_func=F.valid_seg)]),
                transform.RandomHorizontalFlip()]
        else:
            transforms = [transform.Resize(sampler)]
        self.transforms = transform.Compose(transforms)

    def visualize(self, image, mask, timage, tmask):
        self.vis += 1
        if self.vis % self.opt.visualize_step != 0:
            return
        self.vis = 0
        before_trans = np.copy(image)
        vis_util.visualize_bbox_and_masks_on_image_array(
            before_trans, None, mask, self.readable_label_mapping)
        after_trans = np.copy(timage)
        vis_util.visualize_bbox_and_masks_on_image_array(
            after_trans, None, tmask, self.readable_label_mapping)
        self.vis_idx = vis.image([before_trans, after_trans],
                                 title='voc_transforms', idx=self.vis_idx)


    def __getitem__(self, index):
        im_file, mask_file = self.items[index]
        image = np.array(Image.open(im_file).convert('RGB'))
        mask = np.array(Image.open(mask_file))
        height, width = image.shape[0], image.shape[1]
        assert height == mask.shape[0] and width == mask.shape[1], \
            f'image shape {image.shape} not equal with mask shape {mask.shape}'
        if self.train:
            timage, tmask = self.transforms(image, mask)
            self.visualize(image, mask, timage, tmask)
            return F.transpose(timage), tmask
        else:
            return F.transpose(self.transforms(image)), mask_file, width, height

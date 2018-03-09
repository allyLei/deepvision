#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-12-29 01:14:03
'''

import os
import logging
import json
import numpy as np
from PIL import Image

from . import dataset
from .transforms import transforms as transform
from .transforms import functional as F

from libs.visualizer import vis, vis_util


class Dataset(dataset.Dataset):

    def read(self):
        self.items = []
        data_dir = self.opt.data_dir
        if self.train:
            # CCP dataset
            image_dir = os.path.join(data_dir, 'ccp', 'photos')
            anno_dir = os.path.join(data_dir, 'ccp', 'trainval', 'train')
            self.read_one(image_dir, anno_dir, 'ccp')
            # Fashionista dataset
            image_dir = os.path.join(data_dir, 'fashionista', 'photos')
            anno_dir = os.path.join(data_dir, 'fashionista', 'mask')
            self.read_one(image_dir, anno_dir, 'fashionista')
            # Label data
            image_dir = os.path.join(data_dir, 'label_data', 'photos')
            anno_dir = os.path.join(data_dir, 'label_data', 'mask')
            self.read_one(image_dir, anno_dir, 'labeldata')
            # Label data v2
            image_dir = os.path.join(data_dir, 'label_data', 'photos_v2')
            anno_dir = os.path.join(data_dir, 'label_data', 'mask_v2')
            self.read_one(image_dir, anno_dir, 'labeldatav2')
        else:
            # Use part of CCP dataset as validation
            image_dir = os.path.join(data_dir, 'ccp', 'photos')
            anno_dir = os.path.join(data_dir, 'ccp', 'trainval', 'minival')
            self.read_one(image_dir, anno_dir, 'ccp')
        self.logger.info('total {} images from {}'.format(
            len(self.items), 'train' if self.train else 'val'))

        self.readable_label_mapping = {}
        file_name = os.path.join(data_dir, 'label_mapping.json')
        doc = map(lambda line: json.loads(line.strip()), open(file_name))
        for item in doc:
            self.readable_label_mapping[int(item['id'])] = item['name']

    def read_one(self, image_dir, anno_dir, name):
        for i, file_name in enumerate(os.listdir(anno_dir)):
            assert file_name.endswith('.png'), \
                '{} not valid'.format(file_name)
            image_id = file_name.split('.')[0]
            mask_file = os.path.join(anno_dir, file_name)
            image_file = os.path.join(image_dir, f'{image_id}.jpg')
            assert os.path.exists(image_file), \
                f'{image_file} file does not exist'
            self.items.append([image_file, mask_file])

            if len(self.items) % 100 == 0:
                self.logger.debug('read {} images from {}'.format(
                                      len(self.items), name))

    def gen_transforms(self):
        sampler = transform.Sampler(self.opt.scales)
        if self.train:
            background_dir = os.path.join(self.opt.data_dir, 'background')
            transforms = [
                transform.Background(background_dir, prob=0.4),
                transform.Select(transform.ColorJitter(0.1,0.1,0.1,0.05), index=0),
                transform.CropAndResize(area_ratio=[0.5, 2.0],
                                        scale_sampler=sampler,
                                        fill=255, valid_func=F.valid_seg),
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
                                 title='cloth_transforms', idx=self.vis_idx)

    def __getitem__(self, index):
        image_file, mask_file = self.items[index]
        image = np.array(Image.open(image_file).convert('RGB'))
        mask = np.array(Image.open(mask_file))
        height, width = image.shape[0], image.shape[1]
        assert height == mask.shape[0] and width == mask.shape[1], \
            f'image shape {image.shape} does not match mask shape {mask.shape}'
        if self.train:
            timage, tmask = self.transforms(image, mask)
            self.visualize(image, mask, timage, tmask)
            return F.transpose(timage), tmask
        else:
            return F.transpose(self.transforms(image)), mask_file, width, height

    def __len__(self):
        if self.train:
            return 3548
        else:
            return 94


if __name__ == '__main__':
    logging.basicConfig(
        level='DEBUG',
        format='[%(asctime)s-%(levelname)s-%(name)s-%(lineno)s]:%(message)s')
    from easydict import EasyDict as edict
    opt = edict()
    opt.data_dir = '/home/sunlei/workspace/detect/datasets/clothing'
    dataset = Dataset(opt, True)

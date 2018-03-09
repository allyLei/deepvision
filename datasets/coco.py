#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-12-29 01:44:04
'''

import os
import json
import collections
import numpy as np
from PIL import Image

from . import dataset
from .pycocotools.coco import COCO
from .transforms import transforms as transform
from .transforms import functional as F

from libs.det import gen_anchors_list, gen_anchor_targets
from libs.visualizer import vis, vis_util

class Dataset(dataset.Dataset):

    def read(self):
        if self.train:
            anno_file = 'instances_trainval35k.json'
            self.image_dir = os.path.join(self.opt.data_dir, 'trainval35k')
        else:
            anno_file = 'instances_minival2014.json'
            self.image_dir = os.path.join(self.opt.data_dir, 'minival')
        file_name = os.path.join(self.opt.data_dir, 'annotations', anno_file)
        self.coco = COCO(file_name)
        self.items = list(self.coco.imgs.items())
        self.anchors = {}
        # Read label mapping
        file_name = os.path.join(self.opt.data_dir,
                                 'annotations', 'coco_label_mapping.json')
        doc = json.load(open(file_name))
        self.label_mapping = {}
        self.readable_label_mapping = {}
        for item in doc:
            self.label_mapping[item['id']] = item['mid']
            self.readable_label_mapping[item['mid']] = item['name']


    def parse_example(self, image_id, height, width):
        ann_ids = self.coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        ret = collections.defaultdict(list)
        for ann in anns:
            m = self.coco.annToMask(ann)
            if m.shape[0] != height or m.shape[1] != width:
                self.logger.debug(f'imageid {image_id}, mask shape {m.shape} '
                                  f'does not match {height},{width}')
                continue
            cat_id = self.label_mapping[ann['category_id']]
            cat_id = -1 if ann['iscrowd'] else cat_id
            ret['bboxes'].append(ann['bbox'])
            ret['classes'].append(cat_id)
            ret['masks'].append(m)

        if len(ret['bboxes']) == 0:
            self.logger.debug(f'imageid {image_id}, no annotations')
            return np.zeros([0, 5], dtype=np.float32), \
                np.zeros([height, width, 0], dtype=np.uint8)
        classes = np.asarray(ret['classes'])
        bboxes = np.asarray(ret['bboxes'])
        masks = np.stack(ret['masks'], axis=-1)
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        bboxes = np.hstack((bboxes, classes[:, np.newaxis]))
        return bboxes.astype(np.float32), masks.astype(np.uint8)

    def gen_transforms(self):
        sampler = transform.Sampler(self.opt.scales)
        if self.train:
            transforms = [
                transform.RandomChoice([
                    transform.Resize(sampler),
                    transform.CropAndResize(scale_sampler=sampler,
                                            valid_func=F.valid_det,
                                            use_bbox=True)]),
                transform.RandomHorizontalFlip(),
                transform.FilterBbox(self.opt.bbox_min_size),
                transform.Select(transform.ColorJitter(), index=0)
            ]
        else:
            transforms = [transform.Resize(sampler)]
        self.transforms = transform.Compose(transforms)

    def gen_anchors(self, width, height):
        scale = '{}_{}'.format(width, height)
        if scale not in self.anchors:
            anchors_list = gen_anchors_list(width, height)
            total_anchors = np.vstack(anchors_list)
            self.anchors[scale] = total_anchors
        return self.anchors[scale]

    def visualize(self, image, bboxes, timage, tbboxes):
        self.vis += 1
        if self.vis % self.opt.visualize_step != 0:
            return
        self.vis = 0
        before_trans = np.copy(image)
        vis_util.visualize_boxes_and_labels_on_image_array(
            before_trans, bboxes[:,:4],
            bboxes[:,4].astype(np.int32),
            np.ones(bboxes.shape[0]).astype(np.float32),
            self.readable_label_mapping)
        after_trans = np.copy(timage)
        vis_util.visualize_boxes_and_labels_on_image_array(
            after_trans, tbboxes[:,:4],
            tbboxes[:,4].astype(np.int32),
            np.ones(tbboxes.shape[0]).astype(np.float32),
            self.readable_label_mapping)
        vis.image([before_trans, after_trans],
                   title='coco_transforms', idx=self.vis_idx)


    def __getitem__(self, index):
        try:
            image_id, item = self.items[index]
            image_file = os.path.join(self.image_dir, item['file_name'])
            image = np.array(Image.open(image_file).convert('RGB'))
            height, width = item['height'], item['width']
            if not self.train:
                image = self.transforms(image)
                anchors = self.gen_anchors(image.shape[1], image.shape[0])
                return F.transpose(image), anchors, width, height, image_id

            bboxes, masks = self.parse_example(image_id, height, width)
            timage, tmasks, tbboxes = self.transforms(image, masks, bboxes)
            height, width = timage.shape[0], timage.shape[1]
            anchors = self.gen_anchors(width, height)
            labels, targets = gen_anchor_targets(
                self.opt, width, height, tbboxes, anchors)
            self.visualize(image, bboxes, timage, tbboxes)
            if self.opt.use_mask:
                return F.transpose(timage), labels, targets, tmasks
            else:
                return F.transpose(timage), labels, targets
        except Exception as e:
            self.logger.error(f'coco: {e}')



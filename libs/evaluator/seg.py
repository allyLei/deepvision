#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-08 20:24:57
'''

import os
import sys
import json
from functools import partial
import pprint
from concurrent.futures import ProcessPoolExecutor

import torch
import numpy as np
from PIL import Image
from torch.autograd import Variable

from . import evaluator
from nets.seg.postprocessing import postprocessing
from libs.visualizer import vis, vis_util


def conf_matrix_per_image(pred, mask, num_classes):
    conf_matrix = np.zeros([num_classes, num_classes],
                           dtype=np.float64)
    pred = pred.astype(np.int32).reshape(-1)
    mask = mask.astype(np.int32).reshape(-1)
    for _pd, _gt in zip(pred, mask):
        if _gt == 255:
            continue
        conf_matrix[_gt, _pd] += 1.0
    return conf_matrix


class Evaluator(evaluator.Evaluator):

    def __init__(self, opt, dataloader):
        super(Evaluator, self).__init__(opt, dataloader)
        self.num_classes = self.opt.num_classes
        file_name = os.path.join(self.opt.data_dir, 'label_mapping.json')
        doc = map(lambda line: json.loads(line.strip()), open(file_name))
        self.readable_label_mapping = {}
        for item in doc:
            self.readable_label_mapping[int(item['id'])] = item['name']

    def evaluate(self, model):
        model.eval()
        rets = {'preds': [], 'masks': []}
        for images, mask_files, ws, hs in self.dataloader:
            x = Variable(images, volatile=True)
            preds = model(x)
            preds = postprocessing(preds, ws, hs)
            assert len(preds) == len(mask_files), \
                'len(preds) {} must equal len(masks) {}' \
                .format(len(preds), len(mask_files))
            masks = [np.array(Image.open(file_name)) for file_name in mask_files]
            self.visualize(images[0], preds[0], masks[0], ws[0], hs[0])
            rets['preds'].extend(preds)
            rets['masks'].extend(masks)

        self.metric_eval(rets)

    def visualize(self, image, pred, gt, width, height):
        if isinstance(image, torch.ByteTensor):
            image = image.numpy().copy().transpose([1,2,0]).copy()
        image = np.array(Image.fromarray(image).resize([width, height],
                                                       Image.BILINEAR))
        pred_im = np.copy(image)
        vis_util.visualize_bbox_and_masks_on_image_array(
            pred_im, None, pred, self.readable_label_mapping)
        gt_im = np.copy(image)
        vis_util.visualize_bbox_and_masks_on_image_array(
            gt_im, None, gt, self.readable_label_mapping)
        vis.image([pred_im, gt_im],
                  title='seg_result', idx=self.vis_idx)

    def metric_eval(self, items):
        with ProcessPoolExecutor(max_workers=50) as executor:
            func = partial(conf_matrix_per_image, num_classes=self.num_classes)
            rets = executor.map(func, items['preds'], items['masks'])

        conf_matrix = np.zeros([self.num_classes, self.num_classes],
                               dtype=np.float64)
        for ret in rets:
            conf_matrix += ret

        I = np.diag(conf_matrix)
        U = np.sum(conf_matrix, axis=0) + np.sum(conf_matrix, axis=1) - I
        IoU = I / U
        mIoU = np.mean(IoU)

        IoU = [{'class': self.readable_label_mapping[_class],
                'IoU': IoU[_class]} for _class in range(self.num_classes)]
        IoU = sorted(IoU, key=lambda x: x['IoU'], reverse=True)
        pprint.pprint(IoU)
        sys.stdout.flush()
        self.logger.info(f'mIoU: {mIoU:.4f}')
        return mIoU

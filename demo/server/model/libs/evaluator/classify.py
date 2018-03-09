#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-09 15:32:21
'''

import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch
from torch.autograd import Variable

from . import evaluator
from libs.visualizer import vis, vis_util


def topk(preds, gt_labels, k=5):
    preds = preds.astype(np.int32)
    gt_labels = gt_labels.astype(np.int32)
    top1 = (preds[:, 0] == gt_labels)
    topx = (preds[:, 0] == gt_labels)
    for i in range(1, k):
        topx |= (preds[:, i] == gt_labels)

    return top1.sum(), topx.sum(), preds.shape[0]


class Evaluator(evaluator.Evaluator):

    def __init__(self, opt, dataloader):
        super(Evaluator, self).__init__(self, opt, dataloader)
        self.num_classes = self.opt.num_classes
        file_name = os.path.join(self.opt.data_dir, 'human_readable_file.json')
        doc = json.load(open(file_name))
        self.readable_label_mapping = {}
        for item in doc:
            self.readable_label_mapping[int(item['id'])] = \
                item['human_readable_name']

    def evaluate(self, model):
        model.eval()
        rets = {'preds': [], 'labels': []}
        for images, labels in self.dataloader:
            x = Variable(images, volatile=True)
            preds = model(x)
            _, preds = preds.topk(5, dim=1)
            self.visualize(images[0], preds[0], labels[0])
            rets['preds'].append(preds.numpy())
            rets['labels'].append(labels.numpy())

        self.metric_eval(rets['preds'], rets['labels'])

    def visualize(self, image, preds, label):
        if isinstance(image, torch.ByteTensor):
            image = image.numpy().copy()
        vis_util.visualize_labels_on_image_array(image, preds,
                                                 self.readable_label_mapping)
        title = 'gt: {}'.format(self.readable_label_mapping[label])
        vis.image(image, title=title, idx=self.vis_idx)

    def metric_eval(self, preds, labels):
        with ProcessPoolExecutor(max_workers=50) as executor:
            func = partial(topk, k=self.opt.topk)
            rets = executor.map(func, preds, labels)

        metric = {'top1': 0.0, 'topx': 0.0, 'total': 0.0}
        for v1, vx, vtotal in rets:
            metric['top1'] += v1
            metric['topx'] += vx
            metric['total'] += vtotal

        top1 = metric['top1'] / metric['total']
        topx = metric['topx'] / metric['total']
        self.logger.info(f'top1: {top1}, top{self.opt.topk}: {topx}')
        return top1

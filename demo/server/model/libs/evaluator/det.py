#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-08 17:17:25
'''

import os
import json
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

from . import evaluator
from nets.det.postprocessing import postprocessing
from libs.visualizer import vis, vis_util
from datasets.pycocotools.coco import COCO
from datasets.pycocotools.cocoeval import COCOeval


class Evaluator(evaluator.Evaluator):

    def __init__(self, opt, dataloader):
        super(Evaluator, self).__init__(opt, dataloader)
        file_name = os.path.join(self.opt.data_dir,
                                 'annotations', 'coco_label_mapping.json')
        doc = json.load(open(file_name))
        self.reverse_label_mapping = {}
        self.readable_label_mapping = {}
        for item in doc:
            self.reverse_label_mapping[item['mid']] = item['id']
            self.readable_label_mapping[item['mid']] = item['name']

    def evaluate(self, model):
        model.eval()
        rets = []
        for images, anchors, ws, hs, image_ids in self.dataloader:
            h, w = images.shape[2], images.shape[3]
            _images = Variable(images, volatile=True)
            _anchors = Variable(anchors, volatile=True)
            probs, bboxes = model(_images, anchors=_anchors)
            dets = postprocessing(probs, bboxes, w, h, ws, hs,
                                  self.opt.cls_thresh, self.opt.nms_thresh)
            self.visualize(images[0], dets[0], ws[0], hs[0])

            for j, det in enumerate(dets):
                image_id = image_ids[j]
                rets.extend([{'image_id': image_id,
                              'category_id': self.reverse_label_mapping[int(det[i, 5])],
                              'score': round(float(det[i, 4]), 2),
                              'bbox': [round(float(det[i, 0]), 2), round(float(det[i, 1]), 2),
                                       round(float(det[i, 2]-det[i, 0]), 2),
                                       round(float(det[i, 3]-det[i, 1]), 2)]
                              } for i in range(det.shape[0])])

        self.metric_eval(rets)

    def visualize(self, image, dets, width, height):
        if isinstance(image, torch.ByteTensor):
            image = image.numpy().transpose([1,2,0]).copy()
        image = np.array(Image.fromarray(image).resize([width, height],
                                                       Image.BILINEAR))
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            dets[:, :4],
            dets[:, 5].astype(np.int32),
            dets[:, 4],
            self.readable_label_mapping,
            use_normalized_coordinates=False)
        vis.image(image, title='det_result', idx=self.vis_idx)

    def metric_eval(self, dets):
        image_ids = list(set(map(lambda x: x['image_id'], dets)))
        self.logger.info(f'evaluate over {len(image_ids)} images')
        if not os.path.exists(self.opt.ckpt_dir):
            os.makedirs(self.opt.ckpt_dir)
        file_name = os.path.join(self.opt.ckpt_dir, 'detect.json')
        json.dump(dets, open(file_name, 'w'))
        anno_file = 'instances_minival2014.json'
        anno_file = os.path.join(self.opt.data_dir, 'annotations', anno_file)
        cocoGt = COCO(anno_file)
        cocoDt = cocoGt.loadRes(file_name)
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        return cocoEval.stats[0]

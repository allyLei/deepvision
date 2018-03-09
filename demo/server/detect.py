#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-10-26 11:27:19
'''
import os
import sys
import json
import logging

import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable


from model.det.retina_net import RetinaNet
from model.utils import to_cuda, to_numpy
from model.libs.det import gen_anchors_list
from model.libs.det.anchor.bbox import resize_bboxes, clip_bboxes
from model.libs.det.nms.nms_wrapper import nms
from model.visualizer import vis_util

logging.basicConfig(level='INFO',
                    format='[%(asctime)s-%(levelname)s-%(name)s-%(lineno)s]:%(message)s')

class Detector(object):

    def __init__(self, config_file):
        self.logger = logging.getLogger('detector')
        config = json.load(open(config_file))
        self.model_name = config['name']
        self.label_mapping_file = config['label_mapping_file']
        self.human_readable_file = config['human_readable_file']
        self.image_size = config['image_size']
        self.fpn_layer = config['fpn_layer']
        self.ckpt_path = config['ckpt_path']
        self.backbone = config['backbone']
        self.use_residual = config.get('use_residual', False)
        self.use_bn = config.get('use_bn', False)
        self.nms_thresh = config.get('nms_thresh', 0.5)
        self.cls_thresh = config.get('cls_thresh', 0.25)
        self.num_classes = config.get('num_classes', 80)
        self.visualize_path = config.get('visualize_path', './tmp')
        self.prepare()
        self.model = self.gen_model()

    def prepare(self):
        self.label_mapping = {}
        self.reverse_label_mapping = {}
        tmp = json.load(open(self.label_mapping_file))
        for key, value in tmp.items():
            self.label_mapping[int(key)] = int(value)
            self.reverse_label_mapping[int(value)] = int(key)

        self.human_readable_labels = {}
        doc = json.load(open(self.human_readable_file))
        for item in doc:
            self.human_readable_labels[int(item['id'])] = item['name']
        self.logger.info('init label mapping and anchors done')

        # Generate anchors
        anchors_list = gen_anchors_list(self.image_size[0], self.image_size[1])
        self.anchors = Variable(torch.FloatTensor(np.vstack(anchors_list)), volatile=True).cuda()
        self.logger.info(f'generate {self.anchors.size()} anchors done~')

    def gen_model(self):
        model = RetinaNet(self.num_classes, backbone=self.backbone,
                fpn_layer=self.fpn_layer, use_bn=self.use_bn,
                use_residual=self.use_residual)
        model = torch.nn.DataParallel(model).cuda()
        model.eval()

        ckpt = torch.load(self.ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
        self.logger.info(f'succ load model from {self.ckpt_path}')
        return model

    def preprocessing(self, image):
        w, h = image.size
        np_image = np.copy(np.asarray(image, dtype=np.uint8))
        ratio_x = 1.0 * w / self.image_size[0]
        ratio_y = 1.0 * h / self.image_size[1]
        image = image.resize([self.image_size[0], self.image_size[1]], Image.BILINEAR)
        image = np.asarray(image, dtype=np.uint8).transpose([2,0,1])
        image = np.ascontiguousarray(image)
        image = Variable(torch.ByteTensor(image), volatile=True)
        return np_image, image, ratio_x, ratio_y, w, h

    def postprocessing(self, probs, bboxes, ratio_x, ratio_y, width, height):
        """ Post processing includes nms and resize. """
        probs, bboxes = probs.squeeze(0), bboxes.squeeze(0)
        probs, labels = torch.max(probs, dim=-1)
        indexes = (probs > self.cls_thresh).data.nonzero().view(-1)
        if indexes.numel() == 0:
            return None, None, None
        probs = probs[indexes].unsqueeze(-1)
        labels = (labels[indexes] + 1).unsqueeze(-1).float()
        bboxes = bboxes[indexes, :]
        dets = to_numpy(torch.cat([bboxes, probs, labels], dim=1))
        select = np.argsort(dets[:, 4])[-500:]
        dets = dets[select, :]
        nms_indexes = nms(dets[:, :5], self.nms_thresh)[:100]
        dets = dets[nms_indexes, :]
        dets = resize_bboxes(dets, ratio_x, ratio_y)
        dets = clip_bboxes(dets, width, height)
        boxes, scores, classes = dets[:, :4], dets[:, 4], dets[:, 5]
        classes = np.array([self.reverse_label_mapping[int(x)] for x in classes])
        return boxes, scores, classes


    def detect(self, image, display=False, image_id=None):
        try:
            np_image, image, ratio_x, ratio_y, w, h = self.preprocessing(image)
            image = to_cuda(image).unsqueeze(0)
            probs, bboxes = self.model(image, anchors=self.anchors)
            boxes, scores, classes = self.postprocessing(probs, bboxes,
                                                         ratio_x, ratio_y, w, h)
            if boxes is None:
                return []

            if display:
                self.visualize(np_image, boxes, scores, classes, image_id)

            ret = [{
                    'category': self.human_readable_labels[int(classes[i])],
                    'score': round(float(scores[i]), 2),
                    'bbox': [round(float(boxes[i][0])/w, 2), round(float(boxes[i][1])/h, 2),
                             round(float(boxes[i][2])/w, 2), round(float(boxes[i][3])/h, 2)]
                   } for i in range(boxes.shape[0])]

            return ret
        except Exception as e:
            self.logger.exception(f'exception during detection {e}')
            return []


    def visualize(self, image, bboxes, scores, classes, image_id=None):
        vis_util.visualize_boxes_and_labels_on_image_array(
            image, bboxes, classes.astype(np.int32),
            scores, self.human_readable_labels,
            use_normalized_coordinates=False,
            line_thickness=3)
        image = Image.fromarray(image)

        if not os.path.exists(self.visualize_path):
            os.makedirs(self.visualize_path)
        image_id = image_id if image_id else 'test'
        image.save(os.path.join(self.visualize_path, f'{image_id}.jpg'), 'JPEG')
        #import pdb
        #pdb.set_trace()


def test(model, image_dir):
    for i, file_name in enumerate(os.listdir(image_dir)):
        if i % 100 == 0:
            logging.info(f'have process {i} images')
            display = True

        display = True
        image_id = file_name.split('.')[0]
        file_name = os.path.join(image_dir, file_name)
        image = Image.open(file_name).convert('RGB')
        model.detect(image, display=display, image_id=image_id)
        display = False


if __name__ == '__main__':
    logging.basicConfig(level='INFO',
                    format='[%(asctime)s-%(levelname)s-%(name)s-%(lineno)s]:%(message)s')
    config_file = sys.argv[1]
    image_dir = sys.argv[2]
    logging.info(f'model config file: {config_file}')
    logging.info(f'detect image dir: {image_dir}')

    model = Detector(config_file)
    test(model, image_dir)

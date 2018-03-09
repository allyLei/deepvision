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

from model.seg.fair_net import FairNet
from model.utils import to_cuda, to_numpy, imresize, smooth
from model.visualizer import vis_util


class Segmentor(object):

    def __init__(self, config_file, rotate=False):
        self.logger = logging.getLogger()
        config = json.load(open(config_file))
        self.human_readable_file = config['human_readable_file']
        self.image_size = config['image_size']
        self.rotate = rotate
        self.backbone = config['backbone']
        self.dilation = config['dilation']
        self.fpn_layer = config['fpn_layer']
        self.groups = config['groups']
        self.ckpt_path = config['ckpt_path']
        self.num_classes = config.get('num_classes', 13)
        self.visualize_path = config.get('visualize_path', './tmp')
        self.prepare()
        self.model = self.gen_model()

    def prepare(self):
        self.human_readable_labels = {}
        doc = map(lambda line: json.loads(line.strip()), open(self.human_readable_file))
        for item in doc:
            self.human_readable_labels[int(item['id'])] = item['name']
        self.logger.info('init {} label mapping done'.format(len(self.human_readable_labels)))

    def gen_model(self):
        model = FairNet(self.num_classes, self.backbone, self.groups,
                        dilation=self.dilation, fpn_layer=self.fpn_layer,
                        use_postprocessing=False)
        model = torch.nn.DataParallel(model).cuda()
        model.eval()

        ckpt = torch.load(self.ckpt_path)
        model.load_state_dict(ckpt['state_dict'])
        self.logger.info('succ load model from {}'.format(self.ckpt_path))
        return model

    def preprocessing(self, image):
        w, h = image.size
        np_image = np.copy(np.asarray(image, dtype=np.uint8))
        image = image.resize([self.image_size[0], self.image_size[1]], Image.BILINEAR)
        image = np.asarray(image, dtype=np.uint8).transpose([2,0,1])
        image = np.ascontiguousarray(image)
        image = Variable(torch.ByteTensor(image), volatile=True)
        return np_image, image, w, h


    def postprocessing(self, mask):
        """ Do resize and smooth output mask"""
        if self.rotate:
            mask = imresize(mask, [320, 512], mode=Image.NEAREST,
                            is_binary=False, rotate=90).astype(np.uint8)
        else:
            mask = imresize(mask, [320, 512], mode=Image.NEAREST,
                            is_binary=False).astype(np.uint8)
        mask = smooth(mask)
        return mask

    def detect(self, image, display=False, image_id=None):
        try:
            np_image, image, w, h = self.preprocessing(image)
            image = to_cuda(image).unsqueeze(0)
            mask = to_numpy(self.model(image))[0]
            mask = self.postprocessing(mask)

            if display:
                self.visualize(np_image, mask, image_id)

            height, width = mask.shape
            ret = {
                'mask': mask,
                'width': width,
                'height': height,
                'err': 0
            }
            return ret
        except Exception as e:
            self.logger.exception('exception during segmentation {}'.format(e))
            return {'err': 1}


    def visualize(self, image, mask, image_id=None):
        mask = imresize(mask, [image.shape[1], image.shape[0]],
                        mode=Image.NEAREST,
                        is_binary=False).astype(np.uint8)
        vis_util.visualize_bbox_and_masks_on_image_array(
            image, None, mask, self.human_readable_labels)
        image = Image.fromarray(image)
        if not os.path.exists(self.visualize_path):
            os.makedirs(self.visualize_path)
        image_id = image_id if image_id else 'test'
        image.save(os.path.join(self.visualize_path, '%s.jpg' % image_id), 'JPEG')
        #import pdb
        #pdb.set_trace()


def test(model, image_dir):
    for i, file_name in enumerate(os.listdir(image_dir)):
        if i % 100 == 0:
            logging.info(f'have process {i} images')
            display = True
        image_id = file_name.split('.jpg')[0]
        display = True
        file_name = os.path.join(image_dir, file_name)
        if not os.path.exists(file_name):
            continue
        image = Image.open(file_name).convert('RGB')
        model.detect(image, display=display, image_id=image_id)
        display = False


if __name__ == '__main__':
    logging.basicConfig(level='INFO',
                    format='[%(asctime)s-%(levelname)s-%(name)s-%(lineno)s]:%(message)s')
    config_file = sys.argv[1]
    image_dir = sys.argv[2]
    logging.info('model config file: {}'.format(config_file))
    logging.info('detect image dir: {}'.format(image_dir))

    model = Segmentor(config_file)
    test(model, image_dir)

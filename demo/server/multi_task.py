#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: zhangkai
Email: zhangkai@cmcm.com
Last modified: 2017-12-20 16:16:13
'''

import yaml
import time
import torch
import logging
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision.models import resnet50
from modules.models import MultiTask
from modules.libs import transforms
from util import Logger


class MultiTaskDetector(object):

    def __init__(self):
        self.model = MultiTask(pretrained=False, backend='resnext101_32d')
        self.model.cuda().eval()
        self.model.load_state_dict(torch.load('ckpt/multi_task/multi_task.pth'))
        self.category_mapping = yaml.load(open('config/category_mapping.yml'))
        self.category_trans = yaml.load(open('config/category_trans.yml'))
        self.attr_mapping = yaml.load(open('config/attr_mapping.yml'))
        self.attr_trans = yaml.load(open('config/attr_trans.yml'))
        self.image_size = (224, 224)
        self.logger = Logger()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            #    transforms.Scale(self.image_size),
            #    transforms.ThreeCrop(self.image_size),
            #    transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops]))
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            normalize,
        ])

    def preprocessing(self, image):
        return Variable(self.transform(image).unsqueeze(0), volatile=True).cuda()

        image = image.resize(self.image_size)
        image = np.asarray(image, dtype=np.uint8)
        image = torch.ByteTensor(image)
        image = image.float().div(255)
        image[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        image[:, :, 1] = (image[:, :, 1] - 0.456) / 0.224
        image[:, :, 2] = (image[:, :, 2] - 0.406) / 0.225
        image = image.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        return Variable(image, volatile=True).cuda()

    def detect(self, image, cls_threshold=0.3, attr_threshold=0.3):
        start = time.time()
        inputs = self.preprocessing(image)
        self.logger.debug('preprocessing: %s', time.time() - start)
        start = time.time()
        #bs, ncrops, c, h, w = inputs.size()
        #logits = self.model(inputs.view(-1, c, h, w))
        #logits = logits.view(bs, ncrops, -1).mean(dim=1)
        cls_out, attr_out = self.model(inputs)
        self.logger.debug('forward: %s', time.time() - start)
        cls_preds = F.softmax(cls_out)
        cls_probs, cls_labels = cls_preds.topk(1, 1)
        cls_label = cls_labels.data.cpu().tolist()[0][0]
        cls_prob = cls_probs.data.cpu().tolist()[0][0]

        attr_preds = (F.sigmoid(attr_out)).view(1, -1).data.cpu().tolist()[0]
        attrs = [(self.attr_mapping.get(x[0], x[0]), x[1]) for x in enumerate(attr_preds)]
        attrs = [(self.attr_trans.get(x[0], x[0]), x[1]) for x in attrs]
        attrs = sorted(filter(lambda x: x[1] >= attr_threshold, attrs), key=lambda x: x[1], reverse=True)

        if cls_prob > cls_threshold:
            cls_name = self.category_mapping.get(cls_label)
            category = (self.category_trans.get(cls_name, cls_name), cls_prob)
            return category, attrs
        else:
            return None, attrs


if __name__ == '__main__':
    model = MultiTaskDetector()
    for _ in range(10):
        start = time.time()
        im = Image.open('image/camera.jpg').convert('RGB')
        im = im.rotate(-90, expand=True)
        print(model.detect(im), time.time() - start)

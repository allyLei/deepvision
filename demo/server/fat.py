#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: zhangkai
Email: zhangkai@cmcm.com
Last modified: 2017-12-20 16:16:13
'''

import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from modules.models import resnet50
from modules.libs import transforms
from util import Logger


class FatDetector(object):

    def __init__(self):
        self.model = resnet50(num_classes=2, pretrained=False)
        self.model.cuda().eval()
        self.model.load_state_dict(torch.load('ckpt/fat/fat.pth'))
        self.image_size = [(672, 224), (448, 224), (224, 224)]
        self.logger = Logger()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ScalePad(self.image_size),
            transforms.ToTensor(),
            normalize
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

    def detect(self, image):
        start = time.time()
        inputs = self.preprocessing(image)
        self.logger.debug('preprocessing: %s', time.time() - start)
        start = time.time()
        #bs, ncrops, c, h, w = inputs.size()
        #logits = self.model(inputs.view(-1, c, h, w))
        #logits = logits.view(bs, ncrops, -1).mean(dim=1)
        logits = self.model(inputs)
        self.logger.debug('forward: %s', time.time() - start)
        preds = F.softmax(logits)
        return preds.data.cpu().numpy()[0][1]
        # probs, labels = preds.topk(1, 1)
        # return probs, labels


if __name__ == '__main__':
    fat = FatDetector()
    for _ in range(10):
        start = time.time()
        im = Image.open('image/camera.jpg').convert('RGB')
        print(fat.detect(im), time.time() - start)

#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-09 11:29:11
'''

import visdom
import torch
import numpy as np
from PIL import Image

class Visualizer():

    def __init__(self, port=8097):
        self.vis = visdom.Visdom(port=port)
        self.idx = 0
        self.data = {}

    def gen_idx(self):
        self.idx += 1
        return self.idx

    def convert(self, image):
        '''Convert image into numpy array'''
        if isinstance(image, list):
            return [self.convert(x) for x in image]
        elif isinstance(image, np.ndarray):
            return image.transpose([2,0,1])
        elif isinstance(image, torch.ByteTensor):
            return image.numpy().copy().transpose([2,0,1])
        elif isinstance(image, Image.Image):
            return np.array(image).transpose([2,0,1])
        else:
            raise TypeError('{} type not supported'.format(type(image)))

    def concat(self, images):
        '''Concat a list of different sizes images together'''
        height = 0
        width = 0
        for x in images:
            height = max(height, x.shape[1])
            width = max(width, x.shape[2])
        ret = np.ones([len(images), 3, height, width], dtype=np.uint8) * 255
        for i, x in enumerate(images):
            _, h, w = x.shape
            ret[i,:,:h,:w] = x
        return ret

    def image(self, im, title, idx=None):
        if idx is None:
            idx = self.gen_idx()
        im = self.convert(im)
        if isinstance(im, list):
            im = self.concat(im)
            nrow = im.shape[0]
            width, height = nrow * im.shape[3], im.shape[2]
            self.vis.images(im, nrow=nrow, opts=dict(title=title,
                                padding=10,
                                width=width,
                                height=height), win=idx)
        else:
            width, height = im.shape[2], im.shape[1]
            self.vis.image(im, opts=dict(title=title, width=width,
                                         height=height), win=idx)
        return idx

    def line(self, x, y, legend, title='line', xlabel='x', ylabel='y', idx=None):
        if idx is None:
            idx = self.gen_idx()
        if idx not in self.data:
            self.data[idx] = {'X': [], 'Y': []}

        if isinstance(y, list) and len(y) == 1:
            y = y[0]
        self.data[idx]['X'].append(x)
        self.data[idx]['Y'].append(y)
        self.vis.line(
            X=np.array(self.data[idx]['X']),
            Y=np.array(self.data[idx]['Y']),
            opts={
                'title': title,
                'legend': legend,
                'xlabel': xlabel,
                'ylabel': ylabel
            },
            win=idx
        )
        return idx


vis = Visualizer()










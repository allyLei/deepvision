#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Mail: sunlei@conew.com
Last modified: 2017-08-12 13:50
'''

import torch.nn as nn
import torch.nn.functional as F

from .. import contrib
from ..common import preprocessing


class Model(nn.Module):
    """ This is just a wrapper class for classification models. """

    def __init__(self, backbone, **kwargs):
        super(Model, self).__init__()
        self.backbone_arch = backbone
        self.backbone = getattr(contrib, backbone)(**kwargs)

    def forward(self, x, y=None):
        x = preprocessing(x, self.backbone_arch)
        logits, _ = self.backbone(x)

        if y is None:
            probs = F.softmax(logits, dim=-1)
            return probs
        else:
            y = y.view(-1).long()
            return F.cross_entropy(logits, y)

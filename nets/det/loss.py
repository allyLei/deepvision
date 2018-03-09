#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-06 14:59:33
'''

import torch
from torch.autograd import Variable


def sigmoid_focal_loss(logits, labels, alpha=0.25, gamma=2, label_smoothing=1.0):
    probs = torch.sigmoid(logits)
    num_classes = probs.size(-1)
    ones = Variable(torch.eye(num_classes + 1).cuda())
    dense = ones.index_select(0, labels)[:, 1:]

    weight = alpha * dense + (1. - alpha) * (1. - dense)
    dense = label_smoothing * dense + (1. - label_smoothing) * (1. - dense)
    probs = probs * dense + (1 - probs) * (1 - dense)
    #probs = torch.clamp(probs, min=1e-5, max=1e5)
    probs = torch.clamp(probs, min=1e-20, max=1e5)

    focal_loss = - weight * (1. - probs) ** gamma * torch.log(probs)
    return torch.sum(focal_loss)

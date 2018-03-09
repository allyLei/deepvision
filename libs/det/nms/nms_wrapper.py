#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Mail: sunlei@conew.com
Last modified: 2017-08-02 09:47
'''

from .nms.gpu_nms import gpu_nms
from .nms.cpu_nms import cpu_nms

def nms(dets, thresh, cpu=False):
    if dets.shape[0] == 0:
        return []
    if cpu:
        return cpu_nms(dets, thresh)
    else:
        return gpu_nms(dets, thresh)


#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-04 10:20:04
'''
import logging
from easydict import EasyDict as edict
from datasets import imagenet, voc, coco

def test_imagenet():
    opt = edict()
    opt.data_dir = '/home/sunlei/workspace/datasets/imagenet'
    dataset = imagenet.Dataset(opt, True)

def test_coco():
    opt = edict()
    opt.data_dir = '/home/sunlei/workspace/detect/datasets/mscoco'
    dataset = coco.Dataset(opt, True)

def test_voc():
    opt = edict()
    opt.data_dir = '/home/sunlei/workspace/detect/datasets/voc'
    dataset = voc.Dataset(opt, True)

def test_transform():
    import numpy as np
    import datasets.transforms.transforms as transform
    import datasets.transforms.functional as F

    sampler = transform.Sampler([[300, 300], [300, 400], [400, 300]])
    tran = transform.Resize(sampler)

    tran = transform.CropAndResize(scale_sampler=sampler,
                                   valid_func=F.valid_det,
                                   use_bbox=True)
    image = np.zeros([300, 310, 3], dtype=np.uint8)
    mask = np.zeros([300, 310, 0], dtype=np.uint8)
    bbox = np.zeros([0, 5], dtype=np.float32)
    image, mask, bbox = tran(image, mask, bbox)

    from libs.det import gen_anchor_targets, gen_anchors_list

    h, w = image.shape[0], image.shape[1]
    print('h: {}, w: {}'.format(h, w))
    anchors = gen_anchors_list(w, h)
    anchors = np.vstack(anchors)

    from easydict import EasyDict as edict
    opt = edict()
    opt.rpn_pos_overlap = 0.5
    opt.rpn_low_neg_overlap = 0.2
    opt.rpn_high_neg_overlap = 0.3
    opt.crowd_overlap = 0.3
    opt.border_ratio = 0.8
    labels, targets = gen_anchor_targets(opt, w, h, bbox, anchors)

def test_np_nms():
    from libs.det.nms.nms_wrapper import nms
    import numpy as np
    bboxes = np.array([
        [1., 2., 80., 90., 0.9],
        [3., 4., 100., 90., 0.2],
        [6., 20., 69, 300, 0.3],
        [9., 50., 59., 200., 0.4],
        [45., 50., 59., 200., 0.4],
        [9., 54., 59., 250., 0.4],
        [7., 50., 23., 258., 0.4],
    ], dtype=np.float32)

    nms_thresh=0.5
    nms_indexes = nms(bboxes, nms_thresh, False)
    print(nms_indexes)



if __name__ == '__main__':
    logging.basicConfig(
        level='DEBUG',
        format='[%(asctime)s-%(levelname)s-%(name)s-%(lineno)s]:%(message)s')
    test_imagenet()


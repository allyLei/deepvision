#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-02 17:41:22
'''

import logging
import logging.config

from libs.config import Config
from libs.trainer.trainer import Trainer
from libs.evaluator.det import Evaluator
from nets.det import RetinaNet

def read_config():
    config = Config('argument parser for object detection training')
    config.add_argument('--backbone', type=str, default='resnet101')
    config.add_argument('--groups', type=int, default=1)
    config.add_argument('--dilation', type=int, nargs='+', default=1)
    config.add_argument('--conv', type=str, default='conv',
                        choices=['conv', 'seperable', 'deform'])
    config.add_argument('--fpn_layer', type=int, default=4)
    config.add_argument('--num_anchors', type=int, default=9)
    config.add_argument('--no_fix_backbone_bn', action='store_true', default=False)
    config.add_argument('--use_bn', action='store_true', default=False)
    config.add_argument('--use_residual', action='store_true', default=False)
    config.add_argument('--use_postprocessing', action='store_true', default=False)
    config.add_argument('--cls_thresh', type=float, default=0.05)
    config.add_argument('--nms_thresh', type=float, default=0.5)
    config.add_argument('--bbox_min_size', type=int, default=4,
                        help='minimum bbox size for ground truth')
    config.add_argument('--rpn_pos_overlap', type=float, default=0.5,
                        help='rpn positive overlap threshold')
    config.add_argument('--rpn_low_neg_overlap', type=float, default=0.3,
                        help='the lower bound of rpn negative overlap')
    config.add_argument('--rpn_high_neg_overlap', type=float, default=0.4,
                        help='the higher bound of rpn negative overlap')
    config.add_argument('--crowd_overlap', type=float, default=0.3,
                        help='remove anchors if they ovelap with crowds')
    config.add_argument('--border_ratio', type=float, default=0.8,
                        help='filter border anchors')
    config.add_argument('--use_mask', action='store_true',
                        help='whether to use mask targets in dataloader')
    return config.parse_args()


def gen_network(opt):
    use_postprocessing = True if opt.mode == 'test' else False
    model = RetinaNet(
        opt.num_classes, opt.backbone, opt.groups, opt.dilation, conv=opt.conv,
        fpn_layer=opt.fpn_layer, num_anchors=opt.num_anchors,
        fix_backbone_bn=not opt.no_fix_backbone_bn, use_bn=opt.use_bn,
        use_residual=opt.use_residual)
    if opt.print_arch:
        for key in model.state_dict():
            logging.info(f'{key}')

    return model


if __name__ == '__main__':
    logging.basicConfig(
        level='INFO',
        format='[%(asctime)s-%(levelname)s-%(name)s-%(lineno)s]:%(message)s')
    opt = read_config()
    logging.info(f'enter mode {opt.mode}, only [train|val|test]')
    model = gen_network(opt)
    trainer = Trainer(opt)
    trainer.run(model, Evaluator)


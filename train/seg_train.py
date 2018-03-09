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
from libs.evaluator.seg import Evaluator
from nets.seg import FairNet


def read_config():
    config = Config('argument parser for semantic segmentation training')
    config.add_argument('--backbone', type=str, default='resnet101')
    config.add_argument('--groups', type=int, default=1)
    config.add_argument('--conv', type=str, default='conv',
                        choices=['conv', 'seperable', 'deform'])
    config.add_argument('--fpn_layer', type=int, default=3)
    config.add_argument('--dilation', type=int, nargs='+', default=1)
    config.add_argument('--use_postprocessing', action='store_true', default=False)
    config.add_argument('--use_aug', action='store_true', default=False)
    return config.parse_args()


def gen_network(opt):
    use_postprocessing = True if opt.mode == 'test' else False
    model = FairNet(opt.num_classes, opt.backbone, opt.groups, opt.conv,
                    dilation=opt.dilation, fpn_layer=opt.fpn_layer,
                    use_postprocessing=use_postprocessing)
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


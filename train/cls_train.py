#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-02 17:41:22
'''

import logging
import logging.config

from datasets import gen_dataloader
from libs.config import Config
from libs.trainer.trainer import Trainer
from libs.evaluator.classify import Evaluator
from nets.cls.model import Model


def read_config():
    config = Config('argument parser for classification training')
    config.add_argument('--topk', type=int, default=5)
    config.add_argument('--backbone', type=str, default='resnet101')
    config.add_argument('--groups', type=int, default=1)
    config.add_argument('--conv', type=str, default='conv',
                        choices=['conv', 'seperable', 'deform'])
    config.add_argument('--dilation', type=int, nargs='+', default=1)

    return config.parse_args()

def gen_network(opt):
    if 'resnet' in opt.backbone:
        model = Model(opt.backbone, num_classes=opt.num_classes,
                     dilation=opt.dilation, groups=opt.groups, conv=opt.conv)
    else:
        model = Model(opt.backbone, num_classes=opt.num_classes)
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


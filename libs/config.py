#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-04 10:11:07
'''

import os
import argparse
import yaml
import pprint

class Config(object):

    def __init__(self, description):
        super(Config, self).__init__()
        self.parser = argparse.ArgumentParser(
            description=description)
        self.add_base_args()

    def add_base_args(self):
        # Basic configuration
        self.parser.add_argument('--mode', type=str, default='train',
                choices=['train', 'test', 'val'], help='mode')
        self.parser.add_argument('--config', type=str, default='',
                help='yaml config file')
        # Dataset and dataloader
        self.parser.add_argument('--num_classes', type=int, default=1000)
        self.parser.add_argument('--dataset', type=str, default='imagenet',
                help='dataset for this training')
        self.parser.add_argument('--data_dir', required=True,
                help='path of dataset')
        self.parser.add_argument('--ckpt_dir', type=str, default='./ckpt',
                help='model checkpoint output directory')
        self.parser.add_argument('--batch_size', type=int, default=64,
                help='training batch size')
        self.parser.add_argument('--num_workers', type=int, default=4,
                help='dataloader number of workers')
        self.parser.add_argument('--scales', type=int, default=[224, 224], nargs='+',
                help='image scales for training and validation')
        self.parser.add_argument('--short_side_scale', type=int, default=256,
                help='the short side scale used in resize and crop transform'
                     'i.e., first resize image according to short side, and crop'
                     'a fixed scale out')
        # Optimizer
        self.parser.add_argument('--optim', type=str, default='sgd',
                choices=['sgd', 'adam', 'rmsprop'])
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)
        self.parser.add_argument('--momentum', type=float, default=0.9)
        self.parser.add_argument('--beta1', type=float, default=0.9)
        self.parser.add_argument('--params', type=str, nargs='+', default=[],
                help='params groups')
        self.parser.add_argument('--lr_mults', type=float, nargs='+', default=[],
                help='lr multipliers for different params groups')
        self.parser.add_argument('--double_bias', action='store_true',
                help='whether to double bias during training')
        # LR strategy
        self.parser.add_argument('--lr', type=float, default=0.01,
                help='init lr value')
        self.parser.add_argument('--lr_policy', type=str, default='exp',
                choices=['exp','step','poly','lamb','cosine','plateau', 'lambda'])
        self.parser.add_argument('--decay_ratio', type=float, nargs='+', default=0.1,
                help='lr decay ratio for [exp|step|poly]')
        self.parser.add_argument('--decay_epoch', type=int, nargs='+', default=10,
                help='lr decay epoch for [exp|step|poly]')
        self.parser.add_argument('--lr_power', type=float, default=0.9,
                help='poly lr power')
        self.parser.add_argument('--min_lr', type=float, default=1e-7)
        # Training
        self.parser.add_argument('--step', type=int, default=0,
                help='init training step')
        self.parser.add_argument('--pretrained', type=str, default='')
        self.parser.add_argument('--pretrained_module', type=str, default='',
                help='pretrained module to restore')
        self.parser.add_argument('--exclude', type=str, nargs='+', default=[],
                help='exclude params when restore parameters')
        self.parser.add_argument('--resume', action='store_true')
        self.parser.add_argument('--max_epochs', type=int, default=200,
                help='max epoches to train')
        self.parser.add_argument('--summary_step', type=int, default=40)
        self.parser.add_argument('--save_step', type=int, default=1000)
        self.parser.add_argument('--optim_step', type=int, default=1,
                help='mimic large batch size')
        self.parser.add_argument('--visualize_step', type=int, default=100,
                help='visualize transform or training loss step')
        self.parser.add_argument('--print_arch', action='store_true', default=False)


    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def parse_args(self):
        opt = self.parser.parse_args()
        if opt.config:
            self.load(opt)

        # Change opt.scales
        opt.scales = [[opt.scales[i], opt.scales[i+1]] for i in range(0,len(opt.scales), 2)]
        # Change decay ratio and decay ratio
        if isinstance(opt.decay_ratio, list) and len(opt.decay_ratio) == 1:
            opt.decay_ratio = opt.decay_ratio[0]
        if isinstance(opt.decay_epoch, list) and len(opt.decay_epoch) == 1:
            opt.decay_epoch = opt.decay_epoch[0]
        pprint.pprint(opt)
        return opt

    def load(self, opt):
        doc = yaml.load(open(opt.config))
        for key, value in doc.items():
            opt.key = value

    def save(self, opt):
        file_name = os.path.join(opt.ckpt_dir, 'config.yml')
        yaml.dump(opt, open(file_name, 'w'))





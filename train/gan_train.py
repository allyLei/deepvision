#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Mail: sunlei@conew.com
Last modified: 2017-07-28 10:16
'''

import logging
import argparse

from libs.config import Config
from libs.trainer.gan import Trainer
from nets.gan import PixelDiscriminator, NLayerDiscriminator
from nets.gan import UnetGenerator, ResnetGenerator


def read_config():
    config = Config('argument parser for gan training')
    config.add_argument('--input_nc', type=int, default=3,
            help='# of input image channels')
    config.add_argument('--output_nc', type=int, default=3,
            help='# of output image channels')
    config.add_argument('--ngf', type=int, default=64,
            help='# of gen filters in first conv layer')
    config.add_argument('--ndf', type=int, default=64,
            help='# of discrim filters in first conv layer')
    config.add_argument('--netD_arch', type=str, default='nlayers',
            help='selects model to use for netD')
    config.add_argument('--netG_arch', type=str, default='resnet_9blocks',
            help='selects model to use for netG')
    config.add_argument('--n_layers_D', type=int, default=3,
            help='only used if netD_arch==n_layers')
    config.add_argument('--dataset_mode', type=str, default='unaligned',
            help='chooses how datasets are loaded. [unaligned '
            '| aligned | single]')
    config.add_argument('--model', type=str, default='cycle_gan',
            help='chooses which model to use. cycle_gan, pix2pix, test')
    config.add_argument('--reverse_AB', action='store_true',
            help='AtoB or BtoA')
    config.add_argument('--norm', type=str, default='instance',
            help='instance normalization or batch normalization')
    config.add_argument('--serial_batches', action='store_true',
            help='if true, takes images in order to make batches, '
            'otherwise takes them randomly')
    config.add_argument('--no_dropout', action='store_true',
            help='no dropout for the generator')
    config.add_argument('--max_dataset_size', type=int, default=float("inf"),
            help='Maximum number of samples allowed per dataset. '
            'If the dataset directory contains more than max_dataset_size, '
            'only a subset is loaded.')
    config.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
            help='scaling and cropping of images at load time '
            '[resize_and_crop|crop|scale_width|scale_width_and_crop]')
    config.add_argument('--no_flip', action='store_true',
            help='if specified, do not flip the images for data augmentation')
    config.add_argument('--init_type', type=str, default='normal',
            help='network initialization [normal|xavier|kaiming|orthogonal]')
    config.add_argument('--display_step', type=int, default=100,
            help='frequency of showing training results on screen')
    config.add_argument('--niter', type=int, default=100,
            help='# of iter at starting learning rate')
    config.add_argument('--niter_decay', type=int, default=100,
            help='# of iter to linearly decay learning rate to zero')
    config.add_argument('--no_lsgan', action='store_true',
            help='do *not* use least square GAN, if false, use vanilla GAN')
    config.add_argument('--pool_size', type=int, default=50,
            help='the size of image buffer that stores previously generated images')
    config.add_argument('--lambda_A', type=float, default=10.0,
            help='weight for cycle loss (A -> B -> A)')
    config.add_argument('--lambda_B', type=float, default=10.0,
            help='weight for cycle loss (B -> A -> B)')
    config.add_argument('--lr_decay_iters', type=int, default=50,
            help='multiply by a gamma every lr_decay_iters iterations')
    config.add_argument('--identity', type=float, default=0.5,
            help='use identity mapping. Setting identity other than 1 has an effect of '
            'scaling the weight of the identity mapping loss. For example, if the weight '
            'of the identity loss should be 10 times smaller than the weight of the '
            'reconstruction loss, please set optidentity = 0.1')
    return config.parse_args()


def gen_network(opt):
    if 'resnet' in opt.netG_arch:
        n_blocks = 9 if '9blocks' in opt.netG_arch else 6
        netG = ResnetGenerator(opt.input_nc, opt.output_nc, opt.ngf, norm=opt.norm,
                use_dropout=not opt.no_dropout, n_blocks=n_blocks, init_type=opt.init_type)
    elif 'unet' in opt.netG_arch:
        n_blocks = 7 if '128' in opt.netG_arch else 8
        netG = UnetGenerator(opt.input_nc, opt.output_nc, n_blocks, opt.ngf,
                norm=opt.norm, use_dropout=not opt.no_dropout, init_type=opt.init_type)
    else:
        raise NotImplementedError('Generator arch {} not supported!'.format(opt.netG_arch))

    if 'nlayer' in opt.netD_arch:
        netD = NLayerDiscriminator(opt.input_nc+opt.output_nc, opt.ndf, opt.n_layers_D,
                norm=opt.norm, use_sigmoid=opt.no_lsgan, init_type=opt.init_type)
    elif 'pixel' in opt.netD_arch:
        netD = PixelDiscriminator(opt.input_nc+opt.output_nc, opt.ndf, norm=opt.norm,
                use_sigmoid=opt.no_lsgan, init_type=opt.init_type)
    else:
        raise NotImplementedError('Discriminator arch {} not supported!'.format(opt.netD_arch))
    return netG, netD


if __name__ == '__main__':
    logging.basicConfig(
        level='INFO',
        format='[%(asctime)s-%(levelname)s-%(name)s-%(lineno)s]:%(message)s')
    opt = read_config()
    logging.info(f'enter mode {opt.mode}, only [train|val|test]')

    netG, netD = gen_network(opt)
    trainer = Trainer(opt)
    trainer.run(netG, netD)


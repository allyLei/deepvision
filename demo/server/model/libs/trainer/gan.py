#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-12-27 12:26:58
'''
import os
import logging
import time

import torch
from torch.optim import lr_scheduler

from datasets import gen_dataloader
from nets.gan.loss import GANLoss
from .utils import to_variable, gen_optimizer

class Trainer(object):

    def __init__(self, opt):
        self.logger = logging.getLogger('trainer')
        self.opt = opt

    def init(self, netG, netD, dataloader):
        self.dataloader = dataloader
        netG = torch.nn.DataParallel(netG).cuda()
        netD = torch.nn.DataParallel(netD).cuda()
        # Generate optimizer
        self.opt.steps_per_epoch = len(self.dataloader)
        self.optimizer_G = gen_optimizer(self.opt, netG)
        self.optimizer_D = gen_optimizer(self.opt, netD)
        self.scheduler_G = self.gen_lr_scheduler(self.optimizer_G)
        self.scheduler_D = self.gen_lr_scheduler(self.optimizer_D)
        # Resume training
        if self.opt.resume:
            self.resume(netG, netD)
        return netG, netD

    def resume(self, netG, netD):
        conf_file = os.path.join(opt.ckpt_dir, 'config.yml')
        doc = yaml.load(open(conf_file))
        # Resume training step, ckpt, optimizer
        opt.latest = doc.latest
        opt.step = doc.step * doc.batch_size // opt.batch_size
        if not os.path.exists(opt.latest):
            raise ValueError(f'ckpt {opt.latest} does not exists')
        state = torch.load(opt.latest)
        netG.load_state_dict(state['state_dict_G'])
        netD.load_state_dict(state['state_dict_D'])
        self.logger.info(f'resume ckpt from {opt.latest}')
        if 'optim_G' in state and 'optim_D' in state and \
                doc.optimizer == opt.optimizer:
            self.optimizer_G.load_state_dict(state['optim_G'])
            self.optimizer_D.load_state_dict(state['optim_D'])
            self.logger.info('resume optimizer succ')

    def save(self, netG, netD):
        if self.opt.step % self.opt.save_step == 0:
            if not os.path.exists(opt.ckpt_dir):
                os.makedirs(opt.ckpt_dir)
            state = {'state_dict_G': netG.state_dict(),
                     'state_dict_D': netD.state_dict(),
                     'optim_G': self.optimizer_G.state_dict(),
                     'optim_D': self.optimizer_D.state_dict()}
            # Save model and optimizer
            ckpt_file = os.path.join(self.opt.ckpt_dir, f'model.pth-{self.opt.step}')
            torch.save(state, ckpt_file)
            # Save configurations
            opt.latest = ckpt_file
            conf_file = os.path.join(opt.ckpt_dir, 'config.yml')
            yaml.dump(opt, open(conf_file, 'w'))

    def gen_criterion(self):
        self.criterionGAN = GANLoss(use_lsgan=not self.opt.no_lsgan)
        self.criterionL1 = torch.nn.L1Loss()

    def gen_lr_scheduler(self, optimizer):
        if self.opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1. - max(0, epoch + 1 + self.opt.epoch_count - self.opt.niter) / \
                    float(self.opt.niter_decay)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        elif self.opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.opt.lr_decay_iters, gamma=0.1)
        elif self.opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
                                                       threshold=0.01, patience=5)
        else:
            raise NotImplementedError(f'lr policy {self.opt.lr_policy} not implemented')
        return scheduler

    def adjust_learning_rate(self):
        self.scheduler_D.step()
        self.scheduler_G.step()
        return self.optimizer_G.param_groups[0]['lr']


    def summary(self):
        if self.train_step % self.opt.display_step == 0:
            duration = time.time() - start_time
            examples_per_sec = self.opt.batch_size / duration
            self.logger.info('[{}|{}|{}] lr: {}, loss_D_fake: {}, loss_D_real: {}, '
                    'loss_G_Gan: {}, loss_G_G1: {}, examples_per_sec: {}'.format(
                    epoch, self.train_step, self.opt.batch_size, lr, loss_D_fake.data[0],
                    loss_D_real.data[0], loss_G_Gan.data[0], loss_G_L1.data[0], examples_per_sec))

    def summary(self, epoch, lr, loss_D_fake, loss_D_real, loss_G_GAN, loss_G_L1):
        if self.opt.step % self.opt.summary_step != 0:
            return
        self.logger.info(f'[{epoch}|{self.opt.max_epochs}|{self.opt.step}]: '
                         f'lr {lr:.7f}, loss_D_fake: {loss_D_fake.data[0]}, '
                         f'loss_D_real: {loss_D_real.data[0]}, loss_G_GAN: '
                         f'{loss_G_GAN.data[0]}, loss_G_L1: {loss_G_L1.data[0]}')

    def train(self, netD, netG):
        model.train()
        init_epoch = int(self.opt.step / self.opt.steps_per_epoch)
        for epoch in range(init_epoch, self.opt.max_epochs):
            for real_a, real_b in self.dataloader:
                lr = self.adjust_learning_rate()
                real_a, real_b = to_variable(real_a), to_variable(real_b)
                if self.opt.reverse_ab:
                    real_b, real_a = real_a, real_b

                fake_b = netG(real_a)
                fake_ab = torch.cat([real_a, fake_b], dim=1)
                real_ab = torch.cat([real_a, real_b], dim=1)

                # Optimize D
                self.optimizer_D.zero_grad()
                fake_ab_D = self.shuffle(fake_ab)
                pred_fake = netD(fake_ab_D.detach())
                loss_D_fake = self.criterionGAN(pred_fake, False)
                pred_real = netD(real_ab)
                loss_D_real = self.criterionGAN(pred_real, True)
                loss_D = (loss_D_fake + loss_D_real) * 0.5
                loss_D.backward()
                self.optimizer_D.step()
                # Optimize G
                self.optimizer_G.zero_grad()
                pred_fake = netD(fake_ab)
                loss_D_GAN = self.criterionGAN(pred_fake, True)
                loss_G_L1 = self.criterionL1(fake_b, real_a) * self.opt.lambda_a
                loss_G = loss_G_GAN + loss_G_L1
                loss_G.backward()
                self.optimizer_G.step()

                self.opt.step += 1
                self.summary(epoch, lr, loss_D_fake, loss_D_real,
                             loss_G_GAN, loss_G_L1)
                self.save(netG, netD)

    def run(self, netG, netD):
        dataloader = gen_dataloader(self.opt, True)
        netG, netD = self.init(netG, netD, dataloader)
        self.train(netG, netD)


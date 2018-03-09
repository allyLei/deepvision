#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-04 23:46:04
'''

import time
import logging

import torch
from torch.autograd import Variable

from .utils import gen_optimizer, gen_lr_scheduler
from .utils import resume, save, load, to_variable

from libs.visualizer import vis, vis_util
from datasets import gen_dataloader

class Trainer(object):

    def __init__(self, opt):
        self.logger = logging.getLogger('trainer')
        self.opt = opt
        self.vis_loss_idx = vis.gen_idx()
        self.vis_lr_idx = vis.gen_idx()
        self.vis_val_idx = vis.gen_idx()

    def init(self, model, dataloader):
        self.dataloader = dataloader
        # Model parallel
        model = torch.nn.DataParallel(model).cuda()
        # Generate optimizer
        self.opt.steps_per_epoch = len(self.dataloader)
        self.optimizer = gen_optimizer(self.opt, model)
        self.lr_sched = gen_lr_scheduler(self.opt, self.optimizer)
        # Resume or restore pretrained models
        if self.opt.resume:
            resume(self.opt, model, self.optimizer)
        elif self.opt.pretrained:
            module = model
            if self.opt.pretrained_module:
                module = getattr(module.module, self.opt.pretrained_module)
            load(module, self.opt.pretrained, self.opt.exclude)
        return model

    def aggregate_grads(self):
        if self.opt.optim_step == 1:
            return True
        if not hasattr(self, 'param_groups'):
            # Initialization
            self.optim_step = 0
            self.param_groups = []
            for param_group in self.optimizer.param_groups:
                group = {}
                for p in param_group['params']:
                    if p.grad is not None:
                        data = p.grad.data
                        group[id(p)] = Variable(
                            data.new().resize_as_(data).zero_())
                self.param_groups.append(group)
        # Aggregate its grads
        for group, param_group in zip(self.param_groups,
                                      self.optimizer.param_groups):
            for p in param_group['params']:
                if p.grad is not None:
                    if id(p) in group.keys():
                        group[id(p)].add_(p.grad)
                    else:
                        data = p.grad.data
                        group[id(p)] = Variable(
                            data.new().resize_as_(data).zero_())
        self.optim_step += 1
        if self.optim_step % self.opt.optim_step == 0:
            self.zero_grad()
            self.optim_step = 0
            return True
        else:
            return False

    def zero_grad(self):
        for group, param_group in zip(self.param_groups,
                                      self.optimizer.param_groups):
            for p in param_group['params']:
                if p.grad is not None:
                    p.grad.data = group[id(p)].data / self.opt.optim_step
                    group[id(p)].data.zero_()

    def summary(self, epoch, lr, losses):
        losses = [loss[0].data[0] for loss in losses]
        if hasattr(self, 'total_loss'):
            self.total_loss += losses[0]
        else:
            self.total_loss = 0.0
            self.start = time.time()

        if self.opt.step % self.opt.summary_step != 0:
            return
        average_loss = self.total_loss / self.opt.summary_step
        losses[0] = average_loss
        duration = time.time() - self.start
        speed = self.opt.batch_size * self.opt.summary_step / duration
        self.total_loss = 0.0
        self.start = time.time()

        self.logger.info(f'[{epoch}|{self.opt.max_epochs}|{self.opt.step}]: '
                         f'lr {lr:.7f}, loss: {average_loss:.4f}, batch_size: '
                         f'{self.opt.batch_size}, speed: {speed:.2f}')

        self.visualize(epoch, lr, losses)

    def visualize(self, epoch, lr, losses):
        x = self.opt.step / self.opt.steps_per_epoch
        title = f'[{epoch}|{self.opt.max_epochs}]|{self.opt.batch_size}'
        legend =['average_loss'] + [f'loss_{i}' for i in range(len(losses)-1)]
        vis.line(x, losses, legend, xlabel='epoch',
                 ylabel='loss', title=title, idx=self.vis_loss_idx)
        vis.line(x, lr, ['lr'], xlabel='epoch',
                 ylabel='lr', title=title, idx=self.vis_lr_idx)


    def save(self, model, evaluator=None):
        if self.opt.step % self.opt.save_step == 0:
            metric = evaluator.evaluate(model) if evaluator else None
            model.train()
            save(self.opt, model, self.optimizer, metric)
            # Visualize the validation metric
            if metric:
                x = self.opt.epoch / self.opt.steps_per_epoch
                vis.line(x, metric, ['val'], title='validation',
                         xlabel='epoch', ylabel='metric', idx=self.vis_val_idx)


    def train(self, model, evaluator=None):
        model.train()
        init_epoch = int(self.opt.step / self.opt.steps_per_epoch)
        for epoch in range(init_epoch, self.opt.max_epochs):
            for item in self.dataloader:
                lr = self.lr_sched.step(self.opt.step)

                item = to_variable(item)
                losses = model(*item)
                if isinstance(losses, tuple) or isinstance(losses, list):
                    losses = [torch.mean(x) for x in losses]
                    loss = losses[0]
                else:
                    loss = torch.mean(losses)
                    losses = [loss]
                self.optimizer.zero_grad()
                loss.backward()

                if self.aggregate_grads():
                    self.optimizer.step()

                self.opt.step += 1
                self.summary(epoch, lr, losses)
                self.save(model, evaluator)


    def run(self, model, Evaluator=None):
        if Evaluator is not None:
            eval_dataloader = gen_dataloader(self.opt, train=False)
            evaluator = Evaluator(self.opt, eval_dataloader)
        else:
            evaluator = None

        if self.opt.mode == 'val':
            assert evalutor is not None, 'set evaluator when in model val'
            model = evaluator.init(model)
            evaluator.evaluate(model)
        elif self.opt.mode == 'train':
            dataloader = gen_dataloader(self.opt, train=True)
            model = self.init(model, dataloader)
            self.train(model, evaluator)
        else:
            raise ValueError(f'{self.opt.mode} not support!!!')



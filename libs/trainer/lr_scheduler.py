#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-05 10:22:52
'''

from bisect import bisect_right
import math

class LRScheduler(object):

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lrs = list(map(lambda group: group['initial_lr'],
                                 optimizer.param_groups))

    def get_lr(self, step):
        raise NotImplementedError

    def step(self, cur_step):
        for param_group, lr in zip(
                self.optimizer.param_groups, self.get_lr(cur_step)):
            param_group['lr'] = lr
        return lr


class StepLR(LRScheduler):

    def __init__(self, optimizer, ratios, dsteps, min_lr=1e-7):
        super(StepLR, self).__init__(optimizer)
        if isinstance(dsteps, int):
            dsteps = [dsteps]
        if not isinstance(ratios, list):
            ratios = [ratios] * len(dsteps)
        elif len(ratios) != len(dsteps):
            raise ValueError('decay ratios {} does not match steps {}'
                             .format(ratios, dsteps))
        self.ratios = [1.0]
        for r in ratios:
            self.ratios.append(self.ratios[-1] * r)
        self.dsteps = dsteps
        self.min_lr = min_lr

    def get_lr(self, step):
        return [max(base_lr * self.ratios[bisect_right(self.dsteps, step)],
                    self.min_lr) for base_lr in self.base_lrs]


class ExponentialLR(LRScheduler):

    def __init__(self, optimizer, ratio, dstep, min_lr=1e-7):
        super(ExponentialLR, self).__init__(optimizer)
        self.ratio = ratio
        self.dstep = dstep
        self.min_lr = min_lr

    def get_lr(self, step):
        return [max(base_lr * self.ratio ** (step * 1.0 / self.dstep),
                    self.min_lr) for base_lr in self.base_lrs]


class PolyLR(LRScheduler):

    def __init__(self, optimizer, power, dstep, min_lr=1e-7):
        super(PolyLR, self).__init__(optimizer)
        self.power = power
        self.dstep = dstep
        self.min_lr = min_lr

    def get_lr(self, step):
        return [max(base_lr * max(0., 1. - step * 1. / self.dstep) **
                    self.power, self.min_lr) for base_lr in self.base_lrs]


class CosineAnnealingLR(LRScheduler):

    def __init__(self, optimizer, dstep, min_lr=1e-7):
        super(CosineAnnealingLR, self).__init__(optimizer)
        self.dstep = dstep
        self.min_lr = min_lr

    def get_lr(self, step):
        return [self.min_lr + (base_lr - self.min_lr) *
                (1 + math.cos(step * 1. / self.dstep * math.pi)) / 2
                for base_lr in self.base_lrs]


class ReduceLROnPlateau(object):

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='val',
                 cooldown=0, min_lr=0):
        # TODO
        pass

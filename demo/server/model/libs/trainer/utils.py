#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2018-01-04 23:51:17
'''
import os
import logging
import yaml
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import numpy as np

from . import lr_scheduler


# =================================
# Optimizer and lr scheduler
# =================================

def gen_param_groups(opt, model):
    if not isinstance(opt.params, list) or not isinstance(opt.lr_mults, list) \
            or len(opt.params) != len(opt.lr_mults):
        raise ValueError(f'params {opt.params} or lr mults {opt.lr_mults} value error')
    param_groups = []
    if len(opt.params) == 0:
        opt.params, opt.lr_mults = ['module'], [1.0]
    model_params = list(
        filter(
            lambda x: x[1].requires_grad,
            model.named_parameters()))
    for name, lr_mult in zip(opt.params, opt.lr_mults):
        lr = opt.lr * lr_mult
        params = list(filter(lambda x: name in x[0], model_params))
        # Fix parameters if lr == 0
        if lr <= 0:
            for name, param in params:
                logging.warning(f'fix param {name}, {param.size()}')
                param.requires_grad = False
                continue
        if opt.double_bias:
            no_bias = list(map(lambda x: x[1], filter(
                lambda x: 'bias' not in x[0], params)))
            bias = list(map(lambda x: x[1], filter(
                lambda x: 'bias' in x[0], params)))
            param_groups.append(
                {'params': no_bias, 'initial_lr': lr, 'lr': lr})
            param_groups.append(
                {'params': bias, 'initial_lr': 2*lr, 'lr': 2*lr})
        else:
            params = list(map(lambda x: x[1], params))
            param_groups.append({'params': params, 'initial_lr': lr, 'lr': lr})
    # Filter invalid param groups
    param_groups = list(filter(lambda x: len(x['params']) > 0, param_groups))
    count = sum(map(lambda x: len(x['params']), param_groups))
    for i, param_group in enumerate(param_groups):
        logging.info('{}th param_group, {} params, init lr {}'
                     .format(i, len(param_group['params']), param_group['lr']))
    logging.info(f'total {len(param_groups)} param groups, {count} params')
    logging.info('total {} params, {} are fixed'.format(
        len(list(model.named_parameters())),
        len(list(model.named_parameters())) - len(model_params)))
    return param_groups


def gen_lr_scheduler(opt, optimizer):
    # Compute decay steps
    if isinstance(opt.decay_epoch, list):
        decay_step = [int(opt.steps_per_epoch * epoch) for epoch in opt.decay_epoch]
    else:
        decay_step = int(opt.steps_per_epoch * opt.decay_epoch)

    logging.info(f'steps per epoch: {opt.steps_per_epoch:.2f}')
    logging.info(f'lr policy {opt.lr_policy}, decay step {decay_step}')
    if opt.lr_policy == 'step':
        sched = lr_scheduler.StepLR(
            optimizer, opt.decay_ratio, decay_step, opt.min_lr)
    elif opt.lr_policy == 'exp':
        sched = lr_scheduler.ExponentialLR(
            optimizer, opt.decay_ratio, decay_step, opt.min_lr)
    elif opt.lr_policy == 'poly':
        sched = lr_scheduler.PolyLR(
            optimizer, opt.lr_power, decay_step, opt.min_lr)
    elif opt.lr_policy == 'cosine':
        sched = lr_scheduler.CosineAnnealingLR(
            optimizer, decay_step, opt.min_lr)
    else:
        raise NotImplementedError('{} not supported', opt.lr_policy)
    return sched


def gen_optimizer(opt, model):
    param_groups = gen_param_groups(opt, model)
    if opt.optim == 'adam':
        logging.info(f'optim adam, beta1 {opt.beta1} weight decay {opt.weight_decay}')
        optimizer = torch.optim.Adam(param_groups, opt.lr,
                                     betas=(opt.beta1, 0.999),
                                     weight_decay=opt.weight_decay)
    elif opt.optim == 'sgd':
        logging.info(f'optim sgd, momentum {opt.momentum} weight decay {opt.weight_decay}')
        optimizer = torch.optim.SGD(param_groups, opt.lr,
                                    momentum=opt.momentum,
                                    weight_decay=opt.weight_decay)
    elif opt.optim == 'rmsprop':
        logging.info(f'optim rmsprop, momentum {opt.momentum} weight decay {opt.weight_decay}')
        optimizer = torch.optim.RMSprop(param_groups, opt.lr,
                                        momentum=opt.momentum,
                                        weight_decay=opt.weight_decay)
    else:
        raise ValueError(f'no implementation of {opt.optim} optimizer')
    return optimizer


# ====================================
# Save and load model
# ====================================

def save(opt, model, optimizer=None, metric=None):
    if not os.path.exists(opt.ckpt_dir):
        os.makedirs(opt.ckpt_dir)

    # Save model and optimizer
    state = {'state_dict': model.state_dict()}
    if optimizer:
        state.update({'optim': optimizer.state_dict()})
    ckpt_file = os.path.join(opt.ckpt_dir, 'model.pth-{}'.format(opt.step))
    torch.save(state, ckpt_file)
    # Save configurations
    opt.latest = ckpt_file
    conf_file = os.path.join(opt.ckpt_dir, 'config.yml')
    yaml.dump(opt, open(conf_file, 'w'))

    if metric is None:
        return

    is_best = False
    if not hasattr(opt, 'best_metric'):
        opt.best_metric = metric
        is_best = True
    elif metric > opt.best_metric:
        opt.best_metric = metric
        is_best = True

    # Update the best
    if is_best:
        dir_name = os.path.join(opt.ckpt_dir, 'best')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        ckpt_file = os.path.join(dir_name, 'model-pth.best')
        torch.save(state, ckpt_file)
        opt.best_ckpt = ckpt_file
        conf_file = os.path.join(dir_name, 'config.yml')
        yaml.dump(opt, open(conf_file, 'w'))


def resume(opt, model, optimizer=None):
    conf_file = os.path.join(opt.ckpt_dir, 'config.yml')
    doc = yaml.load(open(conf_file))
    # Resume training step, ckpt, optimizer
    opt.latest = doc.latest
    opt.step = doc.step * doc.batch_size // opt.batch_size

    if not os.path.exists(opt.latest):
        raise ValueError(f'ckpt {opt.latest} does not exists')
    state = torch.load(opt.latest)
    model.load_state_dict(state['state_dict'])
    logging.info(f'resume ckpt from {opt.latest}')
    if optimizer and 'optim' in state and doc.optim == opt.optim:
        optimizer.load_state_dict(state['optim'])
        logging.info('resume optimizer succ')


def load(model, ckpt_file, exclude=''):
    ckpt = torch.load(ckpt_file)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    # Load state dict
    succ = 0
    model_state = model.state_dict()
    for name, param in ckpt.items():
        if name in model_state.keys():
            if isinstance(param, Parameter):
                param = param.data
            if exclude and name.startswith(exclude):
                logging.warning(f'skip param {name} in {exclude}')
                continue
            model_state[name].copy_(param)
            succ += 1

    logging.info('succ load {} from {} params from {}, {} params not loaded'
                 .format(succ, len(ckpt), ckpt_file, len(model_state) - succ))

# ==================================
# Torch utils
# ==================================

def to_variable(x):
    if isinstance(x, np.ndarray):
        return Variable(torch.from_numpy(x), requires_grad=False)
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.LongTensor) or\
            isinstance(x, torch.IntTensor) or isinstance(x, torch.DoubleTensor) \
            or isinstance(x, torch.ByteTensor):
        return Variable(x)
    elif isinstance(x, Variable):
        return x
    elif isinstance(x, list) or isinstance(x, tuple):
        y = []
        for i, e in enumerate(x):
            y.append(to_variable(e))
        return y
    elif isinstance(x, dict):
        y = dict(map(lambda _x: (_x[0], to_variable(_x[1])), x.items()))
        return y
    else:
        raise TypeError


def to_cuda(x):
    if isinstance(x, np.ndarray):
        return Variable(
            torch.from_numpy(x),
            requires_grad=False).cuda(
            async=True)
    elif isinstance(x, torch.FloatTensor) or isinstance(x, torch.LongTensor) or\
            isinstance(x, torch.IntTensor) or isinstance(x, torch.DoubleTensor) \
            or isinstance(x, torch.ByteTensor):
        return Variable(x).cuda(async=True)
    elif isinstance(x, Variable):
        return x.cuda(async=True)
    elif isinstance(x, list) or isinstance(x, tuple):
        y = []
        for i, e in enumerate(x):
            y.append(to_cuda(e))
        return y
    elif isinstance(x, dict):
        y = dict(map(lambda _x: (_x[0], to_cuda(_x[1])), x.items()))
        return y
    else:
        raise TypeError


def to_numpy(x):
    if isinstance(x, torch.FloatTensor) or \
          isinstance(x, torch.IntTensor) or \
          isinstance(x, torch.DoubleTensor) or \
          isinstance(x, torch.LongTensor) or \
          isinstance(x, torch.ByteTensor):
        return x.numpy().copy()
    if isinstance(x, Variable):
        if x.is_cuda:
            return x.cpu().data.numpy()
        return x.data.numpy()
    elif isinstance(x, list) or isinstance(x, tuple):
        y = []
        for i, e in enumerate(x):
            y.append(to_numpy(e))
        return y
    else:
        raise TypeError


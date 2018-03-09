#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: sunlei
Email: sunlei@cmcm.com
Last modified: 2017-09-14 11:55:36
'''



def resnet_preprocessing(images):
    images = images.float() / 255.
    images[:, 0, :, :] = (images[:, 0, :, :] - 0.485) / 0.229
    images[:, 1, :, :] = (images[:, 1, :, :] - 0.456) / 0.224
    images[:, 2, :, :] = (images[:, 2, :, :] - 0.406) / 0.225
    return images


def inception_preprocessing(images):
    images = images.float() / 255.
    images[:, 0, :, :] = (images[:, 0, :, :] - 0.5) / 0.5
    images[:, 1, :, :] = (images[:, 1, :, :] - 0.5) / 0.5
    images[:, 2, :, :] = (images[:, 2, :, :] - 0.5) / 0.5
    return images

def preprocessing(images, backbone):
    if 'resnet' in backbone or 'mobile' in backbone:
        return resnet_preprocessing(images)
    elif 'inception' in backbone or 'nasnet' in backbone:
        return inception_preprocessing(images)
    else:
        raise NotImplementedError('backbone {} not supported', backbone)





#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import division
import os
import math
import random
import types
from PIL import Image
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import torchvision

from . import functional as F


# -----------------------------------
# Transform operations
# -----------------------------------

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *item):
        for t in self.transforms:
            item = t(*item)
        return item


class RandomChoice(object):

    def __init__(self, transforms, weights=None):
        self.transforms = transforms
        self.acc = weights
        if isinstance(weights, list):
            self.acc = [0.0] * len(weights)
            for i, weight in enumerate(weights):
                self.acc[i] = weight if i == 0 else (self.acc[i-1] + weight)

    def __call__(self, *item):
        if self.acc is None:
            return random.choice(self.transforms)(*item)
        else:
            w = random.uniform(0.0, self.acc[-1])
            for i, acc_w in enumerate(self.acc):
                if w <= acc_w:
                    return self.transforms[i](*item)


class Select(object):

    def __init__(self, transform, index):
        self.transform = transform
        if isinstance(index, int):
            index = [index]
        self.index = index
        self.mapping = dict(
            map(lambda i: (self.index[i], i), range(len(self.index))))

    def __call__(self, *item):
        if len(self.index) == 1:
            convert = [self.transform(item[self.index[0]])]
        else:
            convert = self.transform([item[i] for i in self.index])

        ret = [convert[self.mapping[i]] if i in self.index else item[i]
               for i in range(len(item))]
        return ret


class Lambda(object):

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)


# ------------------------------
# Image operations
# ------------------------------


class Pad(object):

    def __init__(self, padding, fill=0):
        self.fill = fill
        assert isinstance(padding, int) or \
            (isinstance(padding, list) and len(padding) in [2, 4])
        if isinstance(padding, int):
            # Padding the same for left, top, right, bottom
            self.x1, self.y1, self.pw, self.ph = padding, padding, 2*padding, 2*padding
        elif len(padding) == 2:
            # Padding left and right only
            self.x1, self.y1, self.pw, self.ph = padding[0], 0, padding[0]+padding[1], 0
        else:
            # Padding different for left, top, right, bottom
            self.x1, self.y1, self.pw, self.ph = \
                padding[0], padding[1], padding[0]+padding[2], padding[1]+padding[3]

    def __call__(self, image):
        shape = image.shape
        h, w = shape[0], shape[1]
        shape[0] += self.ph
        shape[1] += self.pw
        ret = np.ones(shape, image.dtype) * self.fill
        ret[self.y1:self.y1+h, self.x1:self.x1+w, ...] = image
        return ret


class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        transforms = []
        if brightness > 0:
            brightness_factor = np.random.uniform(
                max(0, 1 - brightness), 1 + brightness)
            transforms.append(
                Lambda(
                    lambda img: F.adjust_brightness(
                        img, brightness_factor)))

        if contrast > 0:
            contrast_factor = np.random.uniform(
                max(0, 1 - contrast), 1 + contrast)
            transforms.append(
                Lambda(
                    lambda img: F.adjust_contrast(
                        img, contrast_factor)))

        if saturation > 0:
            saturation_factor = np.random.uniform(
                max(0, 1 - saturation), 1 + saturation)
            transforms.append(
                Lambda(
                    lambda img: F.adjust_saturation(
                        img, saturation_factor)))

        if hue > 0:
            hue_factor = np.random.uniform(-hue, hue)
            transforms.append(
                Lambda(
                    lambda img: F.adjust_hue(
                        img, hue_factor)))

        np.random.shuffle(transforms)
        transform = torchvision.transforms.transforms.Compose(transforms)

        return transform

    def __call__(self, img):
        img = Image.fromarray(img)
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return np.array(transform(img))

# ----------------------------------
# Sampler
# ----------------------------------


class Sampler(object):

    def __init__(self, scales, weights=None):
        super(Sampler, self).__init__()
        self.scales = scales
        self.acc = weights
        if isinstance(weights, list):
            self.acc = [0.0] * len(weights)
            for i, weight in enumerate(weights):
                self.acc[i] = weight if i == 0 else (self.acc[i-1] + weight)

    def adaptive_scale(self, w, h):
        min_ratio = 0.0
        index = 0
        for i, scale in enumerate(self.scales):
            x, y = w * scale[1], h * scale[0]
            ratio = (x * 1.0 / y) if y > x else (y * 1.0 / x)
            if ratio > min_ratio:
                min_ratio = ratio
                index = i
        return self.scales[index]

    def sample(self, image):
        if len(self.scales) == 1:
            return self.scales[0]
        h, w = image.shape[0], image.shape[1]
        if self.acc is None:
            return self.adaptive_scale(w, h)
        else:
            w = random.uniform(0.0, self.acc[-1])
            for i, acc_w in enumerate(self.acc):
                if w <= acc_w:
                    return self.scales[i]

# -------------------------------------------------------
# Some transforms used for detection and segmentation
# -------------------------------------------------------


class Resize(object):

    def __init__(self, scale_sampler, mode=[Image.BILINEAR, Image.NEAREST]):
        self.scale_sampler = scale_sampler
        self.mode = mode

    def __call__(self, image, mask=None, bbox=None):
        h, w = image.shape[0], image.shape[1]
        scale = self.scale_sampler.sample(image)
        ratio = [scale[0]*1.0/w, scale[1]*1.0/h]
        image = F.resize(image, scale, self.mode[0])
        mask = F.resize(mask, scale, self.mode[1])
        bbox = F.resize_bbox(bbox, ratio)
        return image if mask is None else tuple(
                filter(lambda x: isinstance(x, np.ndarray),
                       [image, mask, bbox]))


class RandomHorizontalFlip(object):

    def __call__(self, image, mask=None, bbox=None):
        if random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            if mask is not None:
                mask = mask[:, ::-1, ...].copy()
            if bbox is not None:
                bbox[:, [0, 2]] = image.shape[1] - bbox[:, [2, 0]]

        return image if mask is None else tuple(
            filter(lambda x: isinstance(x, np.ndarray), [image, mask, bbox]))


class ResizeAndCrop(object):

    def __init__(self, scale=None, ratio=[0.7, 1.5], retry=10,
                 scale_sampler=None, valid_func=None, use_bbox=False,
                 fill=0, mode=[Image.BILINEAR, Image.NEAREST]):
        super(ResizeAndCrop, self).__init__()
        assert scale_sampler is not None, 'must provide one sampler'
        self.scale = scale
        self.ratio = ratio
        self.retry = retry
        self.use_bbox = use_bbox
        self.scale_sampler = scale_sampler
        self.valid_func = valid_func
        self.fill = fill
        self.mode = mode

    def __call__(self, image, mask=None, bbox=None):
        for retry in range(self.retry):
            # First scale image and mask
            if self.scale is None:
                ref_scale = min(image.shape[0], image.shape[1])
            else:
                ref_scale = self.scale
            scale = int(random.uniform(*self.ratio) * ref_scale)
            timage = F.resize(image, scale, self.mode[0], by_short_side=True)
            tmask = F.resize(mask, scale, self.mode[1], by_short_side=True)

            # Then crop one patch out
            crop_scale = self.scale_sampler.sample(timage)
            timage, crop_box = F.random_crop(timage, crop_scale)
            tmask = F.crop(tmask, crop_box, fill=self.fill)

            if self.valid_func is not None and not self.valid_func(
                    mask, tmask):
                continue
            tbbox = F.to_bbox(tmask, bbox) if self.use_bbox else None
            return timage if tmask is None else tuple(
                filter(lambda x: isinstance(x, np.ndarray),
                       [timage, tmask, tbbox]))

        # Fall back
        scale = self.scale_sampler.sample(image)
        timage = F.resize(image, scale, self.mode[0])
        tmask = F.resize(mask, scale, self.mode[1])
        tbbox = F.to_bbox(tmask, bbox) if self.use_bbox else None
        return timage if tmask is None else tuple(
                filter(lambda x: isinstance(x, np.ndarray),
                       [timage, tmask, tbbox]))


class CropAndResize(object):

    def __init__(self, area_ratio=[0.5, 1.0],
                 aspect_ratio=[3. / 4, 4. / 3],
                 retry=10, scale_sampler=None, valid_func=None,
                 fill=0, use_bbox=False, mode=[Image.BILINEAR, Image.NEAREST]):
        assert scale_sampler is not None, 'a scale sampler must provide'
        self.area_ratio = area_ratio
        self.aspect_ratio = aspect_ratio
        self.retry = retry
        self.use_bbox = use_bbox
        self.scale_sampler = scale_sampler
        self.valid_func = valid_func
        self.fill = fill
        self.mode = mode

    def __call__(self, image, mask=None, bbox=None):
        h, w = image.shape[0], image.shape[1]
        for _ in range(self.retry):
            # First obtain crop image
            area = image.shape[0] * image.shape[1]
            target_area = random.uniform(*self.area_ratio) * area
            aspect_ratio = random.uniform(*self.aspect_ratio)
            tw = int(round(math.sqrt(target_area * aspect_ratio)))
            th = int(round(math.sqrt(target_area / aspect_ratio)))
            if random.random() < 0.5:
                tw, th = th, tw
            if tw > w or th > h:
                continue
            timage, crop_box = F.random_crop(image, [tw, th])
            tmask = F.crop(mask, crop_box, fill=self.fill)
            tbbox = F.to_bbox(tmask, bbox) if self.use_bbox else None
            if self.valid_func is not None and not self.valid_func(
                    mask, tmask):
                continue
            # Then resize image and mask
            scale = self.scale_sampler.sample(timage)
            ratio = [scale[0]*1.0/tw, scale[1]*1.0/th]
            timage = F.resize(timage, scale, self.mode[0])
            tmask = F.resize(tmask, scale, self.mode[1])
            tbbox = F.resize_bbox(tbbox, ratio)
            return timage if tmask is None else tuple(
                filter(lambda x: isinstance(x, np.ndarray),
                       [timage, tmask, tbbox]))
        # Fall back
        scale = self.scale_sampler.sample(image)
        ratio = [scale[0]*1.0/w, scale[1]*1.0/h]
        timage = F.resize(image, scale, self.mode[0])
        tmask = F.resize(mask, scale, self.mode[1])
        if bbox is not None:
            tbbox = F.resize_bbox(bbox, ratio) if self.use_bbox else None
        else:
            tbbox = F.to_bbox(tmask) if self.use_bbox else None
        return timage if tmask is None else tuple(
                filter(lambda x: isinstance(x, np.ndarray),
                       [timage, tmask, tbbox]))


# -------------------------------------
# Detection transforms
# -------------------------------------

class FilterBbox(object):

    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, image, mask, bbox):
        select = np.where((bbox[:, 2]-bbox[:, 0]+1 >= self.min_size) &
                          (bbox[:, 3]-bbox[:, 1]+1 >= self.min_size))[0]
        mask = mask[:, :, select]
        bbox = bbox[select, :]
        return image, mask, bbox


# -------------------------------------
# Clothing segmentation transforms
# -------------------------------------

class Background(object):

    def __init__(self, background_dir, prob=0.5):
        self.background_dir = background_dir
        self.prob = prob

        self.bg_images = []
        for image_dir in os.listdir(self.background_dir):
            image_dir = os.path.join(self.background_dir, image_dir)
            for file_name in os.listdir(image_dir):
                file_name = os.path.join(image_dir, file_name)
                if file_name.endswith('.jpg') and os.path.exists(file_name):
                    self.bg_images.append(file_name)

    def __call__(self, image, mask):
        if random.uniform(0., 1.) >= self.prob:
            return image, mask
        # Padding image into background
        indexes = np.where(mask > 0)
        y1, y2 = min(indexes[0]), max(indexes[0])
        x1, x2 = min(indexes[1]), max(indexes[1])
        #bbox = [x1, y1, x2, y2]
        #x1, y1, x2, y2 = bbox
        #x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        foreground = image[y1:y2+1, x1:x2+1, :]
        mask = mask[y1:y2+1, x1:x2+1]
        fg_h, fg_w = foreground.shape[0], foreground.shape[1]
        # Select one background image
        for i in range(30):
            b_h = int(random.uniform(1.2, 1.6) * fg_h)
            for j in range(10):
                b_w = int(random.uniform(0.5, 0.8) * b_h)
                if fg_w < b_w:
                    break
            file_name = random.choice(self.bg_images)
            background = Image.open(file_name).convert('RGB')
            w, h = background.width, background.height
            # Resize image according to b_h
            _w = int(1.0 * w * b_h / h)
            if _w <= b_w or fg_w >= b_w:
                continue
            background = background.resize((_w, b_h), Image.BILINEAR)
            x = random.randint(0, _w-b_w)
            background = np.array(background)
            background = background[:, x:x+b_w, :]
            # Random put foreground [fg_h, fg_w, 3] into background [b_h, b_w, 3]
            x = random.randint(0, b_w-fg_w-1)
            y = random.randint(0, b_h-fg_h-1)
            select = (mask > 0)[:, :, np.newaxis]
            background[y:y+fg_h, x:x+fg_w, :] = select * foreground + \
                (1 - select) * background[y:y+fg_h, x:x+fg_w, :]
            #tbbox = [x, y, x+fg_w-1, y+fg_h-1]
            tmask = np.zeros([b_h, b_w], dtype=np.uint8)
            tmask[y:y+fg_h, x:x+fg_w] = mask
            return background, tmask



#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: zhangkai
Mail: zhangkai@conew.com
Last modified: 2017-08-09 16:42
'''

import os
import re
import time
import shutil
import logging
import tornado.httpserver
import tornado.ioloop
import tornado.autoreload
import tornado.escape
import tornado.web
import tornado.websocket

import io
import gzip
import json
import hashlib
import collections
import random
import base64
import hashlib
import numpy as np
from detect import Detector
from seg import Segmentor
from bodyheight import HeightCalculation
from fat import FatDetector
from multi_task import MultiTaskDetector
from PIL import Image, ImageOps
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor

from tornado.options import define, options
define("port", default=50000, help="run on the given port", type=int)
define("dev",  default=True, help="dev mode", type=bool)
define("model",  default="ret300,ret600", help="model", type=str)


class ConfigHandler(tornado.web.RequestHandler):

    def get(self):
        config = {
            'min_height': 600,
            'jpeg_quality': 80,
            'max_interval': 3200,
            'delay_time': 3000,
            'mobile_enabled': True,

            'update_url': 'http://update.ai.ishield.cn/static/image.apk?v=%s' % (int(time.time())),
            'update_content': '更新了绘制方式，不再进行采样',
            'update_version_code': 15,
            'update_version_name': '1.0.15',
            'force': True,
            'ignore_able': False,
        }
        self.finish(config)


class VoiceHandler(tornado.web.RequestHandler):

    def get(self):
        config = {
            'update_url': 'http://aidemo.ishield.cn/image-demo.apk?v=%s' % (int(time.time())),
            'update_content': '您有新版本需要更新',
            'update_version_code': 3,
            'update_version_name': '0.0.3',
            'ignore_able': True,
        }
        self.finish(config)


class BaseWebSocketHandler(tornado.websocket.WebSocketHandler):

    def open(self):
        logging.info('websocket opened')

    def on_close(self):
        logging.info('websocket closed')


class HttpDetectHandler(tornado.web.RequestHandler):

    def write(self, chunk):
        if isinstance(chunk, dict):
            chunk = json.dumps(chunk, ensure_ascii=False).encode('utf8')

        buf = io.BytesIO()
        gz = gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=9)
        gz.write(chunk)
        gz.close()
        super(HttpDetectHandler, self).write(buf.getvalue())

    def get_objects(self, ret):
        if not ret:
            return []
        mode = self.get_argument("model", "det300")
        if mode == 'seg512':
            ret = filter(lambda x: x['category'] == 'person' and x['score'] >= 0.5, ret)
            max_area = 0.0
            obj = None
            for item in ret:
                x1, y1, x2, y2 = item['bbox']
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    obj = item
            ret = filter(None, [obj])

        objs = []
        for item in ret:
            x1, y1, x2, y2 = list(map(lambda x: float(x), item['bbox']))
            obj = {
                'name': '%s: %.2f' % (item['category'], item['score']),
                'color': self.application.color_map[item['category']],
                #    'rect': [y1, 1-x2, y2, 1-x1],
                'rect': [x1, y1, x2, y2],
            }
            if mode.find('mask') >= 0:
                obj.update({
                    'width': item['width'],
                    'height': item['height'],
                    'mask': item['mask']
                })
            objs.append(obj)
        return objs

    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self):
        rotate = self.get_argument("rotate", "true")
        width = int(self.get_argument("width", 720))
        height = int(self.get_argument("height", 1080))
        mode = self.get_argument("model", "det300")
        carema_type = self.get_argument("carema_type", "LG-H870DS")
        logging.info('carema_type %s', carema_type)
        model = getattr(self.application, 'model_%s' % mode)

        data = self.request.body
        with open('image/camera.jpg', 'w') as fp:
            fp.write(data)

        image = Image.open(io.BytesIO(data)).convert('RGB')
        if rotate == 'true':
            image = image.rotate(-90, expand=True)

        logging.info('data size: %.3f KB', len(data)/1024.0)

        if mode == 'seg512':
            start = time.time()
            im = ImageOps.expand(image, border=(160, 0), fill=0)
            objs = self.get_objects(self.application.model_det300.detect(im))
            logging.info('detection time: %.5fms', (time.time() - start) * 1000)
            if objs:
                start = time.time()
                ret = model.detect(image)
                logging.info('segmentation time: %.5fms', (time.time() - start) * 1000)

                x1, y1, x2, y2 = objs[0]['rect']
                # y1 = (y1 * 640 - 160) / 320
                # y2 = (y2 * 640 - 160) / 320
                x1 = (x1 * 640 - 160) / 320
                x2 = (x2 * 640 - 160) / 320
                objs[0]['rect'] = [x1, y1, x2, y2]

                '''
                start = time.time()
                height_calculate = HeightCalculation(carema_type)
                person_distance, person_height = height_calculate.measure(objs[0]['rect'])
                logging.info('calc height time: %.5fms', (time.time() - start) * 1000)
                '''
                person_distance, person_height = 0, 0

                # mask = np.array(Image.fromarray(ret['mask']).rotate(90, expand=True, resample=Image.NEAREST))
                mask = ret['mask']
                valid = np.zeros(mask.shape, dtype=np.uint8)
                valid[int(y1*mask.shape[0]):int(y2*mask.shape[0])+1, int(x1*mask.shape[1]):int(x2*mask.shape[1])+1] = 1
                mask = mask * valid

                ret['mask'] = mask.reshape(-1).tolist()
                ret['color'] = self.application.color_map

                start = time.time()
                # after rotate 90
                width = int(self.get_argument('height', 1080))
                height = int(self.get_argument('width', 1920))
                original_ratio = width * 1.0 / height
                resize_ratio = im.size[0] * 1.0 / im.size[1]
                if original_ratio > resize_ratio:
                    size = im.size[0], int(1.0 * im.size[0] * height / width)
                else:
                    size = int(1.0 * im.size[1] * width / height), im.size[1]

                im = image.resize(size)
                x1, y1, x2, y2 = x1 * im.size[0], y1 * im.size[1], x2 * im.size[0], y2 * im.size[1]
                '''
                if (y2 - y1) / (x2 - x1) > 2:
                    padding = ((y2 - y1) / 2 - (x2 - x1)) / 2
                    x1 = x1 - padding
                    x2 = x2 + padding
                    if x1 < 0:
                        x2 = x2 - x1
                        x1 = 0
                    if x2 > im.size[0]:
                        x1 = x1 - (x2 - im.size[0])
                        x2 = im.size[0]
                else:
                    padding = ((x2 - x1) * 2 - (y2 - y1)) / 2
                    y1 = y1 - padding
                    y2 = y2 + padding
                    if y1 < 0:
                        y2 = y2 - y1
                        y1 = 0
                    if y2 > im.size[1]:
                        y1 = y1 - (y2 - im.size[1])
                        y2 = im.size[1]
                '''
                logging.info('crop: %.2f, %.2f, %.2f %.2f, height:width = %.2f', x1, y1, x2, y2, (y2-y1)/(x2-x1))
                im = im.crop((x1, y1, x2, y2))
                im.save('image/crop.jpg', 'JPEG')
                fat = self.application.model_fat.detect(im)
                logging.info('classification time: %.5fms', (time.time() - start) * 1000)

                categories = []
                attrs = []
                # 截取上衣，下衣和连衣裙
                for _label in [3, 6, 7]:
                    _select = np.where(mask == _label)
                    if not (_select[0].shape[0] and _select[1].shape[0]):
                        continue
                    _y1, _x1 = min(_select[0]), min(_select[1])
                    _y2, _x2 = max(_select[0]), max(_select[1])
                    _im = np.array(image)
                    _mask = np.zeros(_im.shape, dtype=np.uint8)
                    _mask[_y1:_y2, _x1:_x2, :] = 1
                    _im = _mask * _im
                    _im = Image.fromarray(_im)
                    _im = _im.crop((max(0, _x1 - 50), max(0, _y1 - 50), min(_x2 + 50, _im.size[0]),  min(_y2 + 50, _im.size[1])))
                    _im.save('image/%s.jpg' % _label)
                    _category, _attrs = self.application.model_multi_task.detect(_im, 0.3, 0.3)
                    categories.append(_category)
                    attrs.extend(_attrs)

                _category, _attrs = self.application.model_multi_task.detect(im, 0.3, 0.3)
                categories.append(_category)
                attrs.extend(_attrs)

                categories = sorted(filter(None, categories), key=lambda x: x[1], reverse=True)
                tmp = sorted(attrs, key=lambda x: x[1], reverse=True)
                attrs = []
                attr_names = []
                for t in tmp:
                    if t[0] not in attr_names:
                        attr_names.append(t[0])
                        attrs.append(t)

                doc = {
                    'err': 0,
                    'segmentation': ret,
                    'objects': objs,
                    'person_info': {
                        'gender': 'female',
                        'height': round(person_height, 2),
                        'distance': round(person_distance, 2),
                        'category': '%s: %.2f' % (categories[0][0], categories[0][1]) if categories else None,
                        'attrs': ['%s: %.2f' % (x[0], x[1]) for x in attrs],
                        'shape': round(fat, 2),
                        'skin': 0,
                        'age': 20
                    },
                    'clothing': []
                }
                self.finish(doc)
            else:
                logging.warning('no person')
                self.finish({'err': 0})
        else:
            ret = model.detect(image)
            objs = self.get_objects(ret)
            self.finish({'err': 0, 'objects': objs})


class Application(tornado.web.Application):

    def __init__(self):
        handlers = [
            (r"/config", ConfigHandler),
            (r"/voice", VoiceHandler),
            (r"/detect", HttpDetectHandler),
        ]
        settings = dict(
            debug=options.dev,
            static_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"),
            template_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
        )
        super(Application, self).__init__(handlers, **settings)

        self.data = None
        self.color_map = collections.defaultdict(lambda: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        for i in range(1, 21):
            self.color_map[str(i)] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        if options.model.find('det300') >= 0:
            self.model_det300 = Detector('conf/retina_net.conf')
        if options.model.find('det600') >= 0:
            self.model_det600 = Detector('conf/retina_net_600.conf')
        if options.model.find('mask600') >= 0:
            self.model_mask600 = MaskRetinaNet('conf/mask_600.conf')
        if options.model.find('seg512') >= 0:
            self.model_seg512 = Segmentor('conf/cloth_320.conf', rotate=False)
            self.model_det300 = Detector('conf/retina_net.conf')

        self.model_fat = FatDetector()
        self.model_multi_task = MultiTaskDetector()
        logging.info("initialize done")


def main():
    tornado.options.parse_command_line()
    sockets = tornado.netutil.bind_sockets(options.port)
    server = tornado.httpserver.HTTPServer(Application(), xheaders=True, max_buffer_size=1024*1024*1024*5)
    server.add_sockets(sockets)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    main()

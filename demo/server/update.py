#!/usr/bin/env python
# -*- coding:utf-8 -*-

'''
Author: zhangkai
Email: zhangkai@cmcm.com
Last modified: 2017-09-18 15:42:47
'''

import os
import time
import logging
import tornado.httpserver
import tornado.ioloop
import tornado.autoreload
import tornado.escape
import tornado.web
import tornado.websocket

from tornado.options import define, options
define("port", default=50000, help="run on the given port", type=int)
define("dev",  default=True, help="dev mode", type=bool)


class ImageHandler(tornado.web.RequestHandler):

    def get(self):
        config = {
            'min_height': 600,
            'jpeg_quality': 90,
            'max_interval': 3200,
            'delay_time': 30000,
            'mobile_enabled': True,

            'update_url': 'http://10.60.242.201:7001/playcamera.apk?v=%s' % (int(time.time())),
            'update_content': '',
            'update_version_code': 22,
            'update_version_name': '1.1.6',
            'force': False,
            'ignore_able': True,
        }
        self.finish(config)


class VoiceHandler(tornado.web.RequestHandler):

    def get(self):
        config = {
            'update_url': 'http://10.60.242.201:7001/voicedemo.apk?v=%s' % (int(time.time())),
            'update_content': '',
            'update_version_code': 10,
            'update_version_name': '0.0.10',
            'ignore_able': True,
        }
        self.finish(config)


class Application(tornado.web.Application):

    def __init__(self):
        handlers = [
            (r"/image", ImageHandler),
            (r"/voice", VoiceHandler),
        ]
        settings = dict(
            debug=options.dev,
            static_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static"),
            template_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates"),
        )
        super(Application, self).__init__(handlers, **settings)


def main():
    tornado.options.parse_command_line()
    sockets = tornado.netutil.bind_sockets(options.port)
    server = tornado.httpserver.HTTPServer(Application(), xheaders=True, max_buffer_size=1024*1024*1024*5)
    server.add_sockets(sockets)
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    main()

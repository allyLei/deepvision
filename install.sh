#!/usr/bin/env bash
# -*- coding:utf-8 -*-

##############################################
#
#  Author: sunlei
#  Email: sunlei@cmcm.com
#  Last modified: 2018-03-08 15:14:01
#
##############################################

echo "install dependencies"
pip install -r requirements
echo "install dependencies finished!"

echo "install pycocotools"
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cd ../..
cp cocoapi/PythonAPI/pycocotools datasets/ -r
rm cocoapi -rf
sed -i 's/import pycocotools._mask as _mask/from . import _mask/g' datasets/pycocotools/mask.py
echo "install pycocotools finished!"

echo "make nms"
cd libs/det/nms
./make.sh
echo "make nms finished!"



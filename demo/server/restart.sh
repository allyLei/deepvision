#!/usr/bin/env bash
# -*- coding:utf-8 -*-

##############################################
#
#  Author: zhangkai
#  Mail: zhangkai@conew.com
#  Last modified: 2017-08-09 17:09
#
##############################################

mkdir -p logs
set -x

# source activate py2

if [[ "$1" == "update" ]]; then
    ps -ef | grep update.py | grep 5001 | grep -v grep | awk '{print $2}' | xargs kill -9
    nohup python update.py -port=5001 -dev=False &>logs/5001.log &
elif [[ "$1" == "det" ]]; then
    export CUDA_VISIBLE_DEVICES=6
    ps -ef | grep service.py | grep 5002 | grep -v grep | awk '{print $2}' | xargs kill -9
    nohup python service.py -model=det300,det600 -port=5002 -dev=False &>logs/5002.log &
elif [[ "$1" == "mask" ]]; then
    export CUDA_VISIBLE_DEVICES=2
    ps -ef | grep service.py | grep 5003 | grep -v grep | awk '{print $2}' | xargs kill -9
    nohup python service.py -model=mask600 -port=5003 -dev=False &>logs/5003.log &
elif [[ "$1" == "seg" ]]; then
    export CUDA_VISIBLE_DEVICES=2
    ps -ef | grep service.py | grep 5004 | grep -v grep | awk '{print $2}' | xargs kill -9
    nohup python service.py -model=seg512 -port=5004 -dev=False &>logs/5004.log &
fi

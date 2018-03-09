#!/usr/bin/env bash
# -*- coding:utf-8 -*-

##############################################
#
#  Author: zhangkai
#  Mail: zhangkai@conew.com
#  Last modified: 2017-08-08 11:36
#
##############################################

set -e -x
files=`find . -name '*.java'`
for file in $files
do
    iconv -f cp936 -t utf-8 $file > 1
    mv 1 $file
done

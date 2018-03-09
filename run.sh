#!/usr/bin/env bash
# -*- coding:utf-8 -*-

##############################################
#
#  Author: sunlei
#  Email: sunlei@cmcm.com
#  Last modified: 2018-01-02 17:56:49
#
##############################################

set -x

function classify(){
    ps aux | grep cls_train.py | grep -v grep | awk '{print $2}' | xargs kill -9
    CUDA_VISIBLE_DEVICES=$1 python train/cls_train.py \
        --dataset imagenet \
        --data_dir data/imagenet \
        --scales 224 224 \
        --backbone resnet101 \
        --batch_size 32
}

function cloth(){
    ps aux | grep seg_train.py | grep -v grep | grep cloth | awk '{print $2}' | xargs kill -9
    CUDA_VISIBLE_DEVICES=$1 nohup python train/seg_train.py \
        --dataset clothing \
        --num_classes 13 \
        --data_dir data/clothing \
        --pretrained data/pretrained_ckpt/resnext101_32.pth \
        --pretrained_module backbone \
        --scales 320 512 \
        --num_workers 8 \
        --batch_size $2 \
        --fpn_layer 3 \
        --groups 32 \
        --dilation 2 3 5 \
        --optim $3 \
        --lr_policy $4 \
        --lr_power 0.9 \
        --max_epochs 200 \
        --weight_decay 0.0001 \
        --lr $5 \
        --summary_step 50 \
        --save_step 200 >> logs/clothing_$2_$3_$4_$5.log 2>&1 &
}

function seg(){
    ps aux | grep seg_train.py | grep -v grep | awk '{print $2}' | xargs kill -9
    CUDA_VISIBLE_DEVICES=$1 nohup python train/seg_train.py \
        --dataset voc \
        --use_aug \
        --num_classes 21 \
        --data_dir data/voc \
        --pretrained data/pretrained_ckpt/resnet101.pth \
        --pretrained_module backbone \
        --short_side_scale 384 \
        --scales 384 384 512 384 384 512 \
        --num_workers 8 \
        --batch_size $5 \
        --dilation 2 5 9 \
        --optim $2 \
        --lr_policy $3 \
        --max_epochs $6 \
        --decay_epoch $6 \
        --weight_decay 0.0005 \
        --summary_step 50 \
        --save_step 500 \
        --lr $4 >> logs/seg_$2_$3_$4_$5_$6.log 2>&1 &
}

function det(){
    ps aux | grep det_train | grep -v grep | awk '{print $2}' | xargs kill -9
    CUDA_VISIBLE_DEVICES=$1 nohup python train/det_train.py \
        --dataset coco \
        --data_dir data/mscoco \
        --scales 400 300 300 300 300 400\
        --num_workers 8 \
        --batch_size $2 \
        --num_classes 80 \
        --pretrained data/pretrained_ckpt/resnet101.pth \
        --pretrained_module backbone \
        --optim $3 \
        --lr $4 \
        --lr_policy $5 \
        --weight_decay 0.0001 \
        --decay_ratio 0.1 0.1 \
        --decay_epoch 40 60 \
        --max_epochs 70 \
        --save_step 2000 \
        --summary_step 50 >> logs/det_$3_$4_$5.log 2>&1 &
}

function gan(){
    ps aux | grep gan_train | grep -v grep | awk '{print $2}' | xargs kill -9
    CUDA_VISIBLE_DEVICES=$1 python train/gan_train.py \
        --dataset gan \
        --data_dir data/gan/facades \
        --scales 256 256 \
        --num_workers 4 \
        --batch_size 1 \
        --short_side_scale 286 \
        --beta1 0.5 \
        --lr_policy lambda \
        --lr 0.0002 \
        --lambda_A 100


}

mkdir -p logs
visdom=`ps aux | grep visdom | grep -v grep`
if [[ "$visdom" == "" ]]; then
    echo "start visdom sever"
    nohup python -m visdom.server &>logs/visdom.log &
fi

#det 2,3,4,5 80 sgd 0.02 step
#seg 2,3 sgd poly 0.01 32 80
cloth 2,3 16 adam poly 0.0002





#!/usr/bin/env bash
source activate py37
#python train.py --dataset MNIST --model derpnet --batch_size 256 --n_epochs 10
python train.py --dataset SOP --model resnet --model_depth 18 --batch_size 128 --n_epochs 10
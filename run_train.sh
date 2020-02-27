#!/usr/bin/env bash
source ~/.profile
source ~/.bashrc
export PATH="/home/raman/Software/anaconda3/bin:$PATH"
#conda init bash
source activate py37dl

#python train.py --dataset MNIST --model derpnet --batch_size 256 --n_epochs 10

# resnet-18
#python train.py --dataset SOP --model resnet --model_depth 18 --batch_size 128 --n_epochs 20
#python train.py --dataset Shopee --model resnet --model_depth 18 --batch_size 128 --n_epochs 20 --crop_scale 0.5 --cropped_data_ratio 0.5 --resume --resume_folder results/2019-05-07_12-46-41/saved_models/save_10.pth
#python train.py --dataset Shopee --model resnet --model_depth 18 --batch_size 128 --n_epochs 20 --crop_scale 0.5 --cropped_data_ratio 0.5 --resume --resume_folder results/2019-08-24_10-10-39/saved_models/save_20.pth

# resnet-50
#python train.py --dataset SOP --model resnet --model_depth 50 --batch_size 32 --n_epochs 20 --crop_scale 0.5 --cropped_data_ratio 0.5 --gpu_id 3
#python train.py --gpu_id 2 --dataset Shopee --model resnet --model_depth 50 --batch_size 32 --n_epochs 20 --crop_scale 0.5 --cropped_data_ratio 0.5 --resume --resume_folder results/2019-09-01_11-50-24/saved_models/save_20.pth

# pretrain on resnet doesn't work FUCK
#python train.py --dataset SOP --model resnet --model_depth 18 --pretrained --batch_size 32 --n_epochs 20 --crop_scale 0.5 --cropped_data_ratio 0.5 --gpu_id 2

# vgg-19
#python train.py --gpu_id 2 --dataset SOP --model vgg --model_depth 19 --batch_size 32 --n_epochs 20 --crop_scale 0.5 --cropped_data_ratio 0.5
#python train.py --dataset Shopee --model vgg --model_depth 19 --batch_size 32 --n_epochs 20 --crop_scale 0.5 --cropped_data_ratio 0.5 --resume --resume_folder results/2019-08-22_10-05-55/saved_models/save_20.pth
#python train.py --gpu_id 1 --dataset Shopee --model vgg --model_depth 19 --batch_size 32 --n_epochs 20 --crop_scale 0.5 --crop_position random --cropped_data_ratio 0.5 --resume --resume_folder results/2019-09-02_14-51-45/saved_models/save_20.pth
python train.py --gpu_id 2 --dataset Shopee --model vgg --model_depth 19 --batch_size 32 --n_epochs 20 --crop_scale 0.5 --crop_position center --cropped_data_ratio 0.5 --resume --resume_folder results/2019-09-02_12-45-34/saved_models/save_20.pth

# vgg_attn (19)
#python train.py --dataset SOP --model vgg_attn --model_depth 19 --batch_size 32 --n_epochs 20 --crop_scale 0.5 --cropped_data_ratio 0.5
#python train.py --dataset Shopee  --model vgg_attn --model_depth 19 --batch_size 32 --n_epochs 20 --crop_scale 0.5 --cropped_data_ratio 0.5 --resume --resume_folder results/2019-08-29_14-08-04/saved_models/save_20.pth

#python train.py --gpu_id 1 --dataset Shopee  --model vgg_attn --model_depth 19 --batch_size 32 --n_epochs 20 --crop_scale 0.5 --cropped_data_ratio 0.5 --resume --resume_folder results/2019-09-01_17-42-26/saved_models/save_20.pth


# inception
#python train.py --gpu_id 3 --dataset SOP --model inception --batch_size 16 --n_epochs 20 --crop_scale 0.5 --cropped_data_ratio 0.5
#python train.py --gpu_id 3 --dataset Shopee --model inception --batch_size 16 --n_epochs 20 --crop_scale 0.5 --cropped_data_ratio 0.5 --resume --resume_folder results/2019-09-02_11-26-29/saved_models/save_20.pth

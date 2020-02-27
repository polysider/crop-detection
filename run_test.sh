#!/usr/bin/env bash
source ~/.profile
source ~/.bashrc
export PATH="/home/raman/Software/anaconda3/bin:$PATH"
source activate py37dl

# 2019-06-24_17-51-24
# dataset	dataset_size	train_test_split	model	model_depth	resume	resume_path	batch_size	n_epochs	crop_scale	cropped_data_ratio	resume path
#Shopee		0.8	resnet	18	True	./results/2019-05-07_12-46-41/saved_models/save_10.pth	128	20	0.5	0.5	results/2019-05-07_12-46-41/saved_models/save_10.pth
# The OG Resnet-18 10/20 epochs SOP/Shopee
#python test.py --gpu_id 0 --model resnet --model_depth 18 --model_folder ./results/2019-06-24_17-51-24/saved_models/ --model_file save_30.pth
# acc: 0.9534

# 2019-07-31_06-01-03
# The OG Resnet-18 10/20 epochs SOP/Shopee
#python test.py --gpu_id 0 --model resnet --model_depth 18 --model_folder ./results/2019-07-31_06-01-03/saved_models/ --model_file save_30.pth
# acc: 0.9517

# 2019-08-23_08-09-21
# dataset	dataset_size	train_test_split	model	model_depth	resume	resume_path	batch_size	n_epochs	crop_scale	cropped_data_ratio	resume path
# Shopee		0.8	vgg	19	True	./results/2019-08-22_10-05-55/saved_models/save_20.pth	32	20	0.5	0.5	results/2019-08-22_10-05-55/saved_models/save_20.pth
# VGG19 28x28 pics lel kek (with a RandomCropRandomScale)
#python test.py --gpu_id 0 --model vgg --model_depth 19 --model_folder ./results/2019-08-23_08-09-21/saved_models/ --model_file save_40.pth
# acc: 0.9424


# 2019-08-23_13-16-17
# dataset	dataset_size	train_test_split	model	model_depth	resume	resume_path	batch_size	n_epochs	crop_scale	crop_transform	cropped_data_ratio	resume path
# Shopee		0.8	resnet	18	False	./results/saved_models/	128	20	0.5	RandomCropRandomScale	0.5	results/saved_models/
# Resnet-18 trained on Shopee from scratch lol (with a RandomCropRandomScale)
#python test.py --gpu_id 0 --model resnet --model_depth 18 --model_folder ./results/2019-08-23_13-16-17/saved_models/ --model_file save_20.pth
# acc: 0.9559695


# 2019-08-25_13-48-13
# dataset	dataset_size	train_test_split	model	model_depth	resume	resume_path	batch_size	n_epochs	crop_scale	crop_transform	cropped_data_ratio	resume path
# Shopee		0.8	resnet	18	True	./results/2019-08-24_10-10-39/saved_models/save_20.pth	128	20	0.5	RandomCropRandomScale	0.5	results/2019-08-24_10-10-39/saved_models/save_20.pth
# Resnet-18 trained on Shopee after SOP (with a RandomCropRandomScale)
#python test.py --gpu_id 0 --model resnet --model_depth 18 --model_folder ./results/2019-08-25_13-48-13/saved_models/ --model_file save_40.pth
# acc: 0.9441


# 2019-08-30_04-57-14
# dataset	dataset_size	train_test_split	model	model_depth	resume	resume_path	batch_size	n_epochs	crop_scale	crop_transform	cropped_data_ratio	resume path
# Shopee		0.8	vgg_attn	19	True	./results/2019-08-29_14-08-04/saved_models/save_20.pth	32	20	0.5	RandomCropRandomScale	0.5	results/2019-08-29_14-08-04/saved_models/save_20.pth
# VGG-19 with "attention" with 28x28 pics
#python test.py --gpu_id 0 --model vgg_attn --model_depth 19 --model_folder ./results/2019-08-30_04-57-14/saved_models/ --model_file save_40.pth
# acc: 0.5004


# 2019-09-02_03-44-21
# dataset	dataset_size	train_test_split	model	model_depth	resume	resume_path	batch_size	n_epochs	sample_size	crop_scale	crop_transform	cropped_data_ratio
# Shopee		0.8	vgg_attn	19	True	./results/2019-09-01_17-42-26/saved_models/save_20.pth	32	20	224	0.5	RandomCropRandomScale	0.5
# VGG-19 with "attention" with 224x224 pics
#python test.py --gpu_id 0 --model vgg_attn --model_depth 19 --model_folder ./results/2019-09-02_03-44-21/saved_models/ --model_file save_40.pth
# acc: 0.9568



# 2019-09-02_03-49-21
# dataset	dataset_size	train_test_split	model	model_depth	resume	resume_path	batch_size	n_epochs	sample_size	crop_scale	crop_transform	cropped_data_ratio
# Shopee		0.8	resnet	50	True	./results/2019-09-01_11-50-24/saved_models/save_20.pth	32	20	224	0.5	RandomCropRandomScale	0.5
# Resnet-50
#python test.py --gpu_id 0 --model resnet --model_depth 50 --model_folder ./results/2019-09-02_03-49-21/saved_models/ --model_file save_40.pth
# acc: 0.9263


# 2019-09-03_03-09-59
# dataset	dataset_size	train_test_split	model	model_depth	resume	resume_path	batch_size	n_epochs	sample_size	crop_scale	crop_transform	cropped_data_ratio
# Shopee		0.8	vgg	19	True	./results/2019-09-02_14-51-45/saved_models/save_20.pth	32	20	224	0.5	RandomCropRandomScale	0.5
# VGG-19 random crop and [0.3, 0.5] scale everywhere
#python test.py --gpu_id 0 --model vgg --model_depth 19 --model_folder ./results/2019-09-03_03-09-59/saved_models/ --model_file save_40.pth
# acc : 0.949

# 2019-09-03_03-10-09
# dataset	dataset_size	train_test_split	model	model_depth	resume	resume_path	batch_size	n_epochs	sample_size	crop_scale	crop_transform	cropped_data_ratio
# Shopee		0.8	vgg	19	True	./results/2019-09-02_12-45-34/saved_models/save_20.pth	32	20	224	0.5	RandomCropRandomScale	0.5
# VGG-19 Center crop and 0.5 scale on SOP, Random crop and scale on Shopee
#python test.py --gpu_id 1 --model vgg --model_depth 19 --model_folder ./results/2019-09-03_03-10-09/saved_models/ --model_file save_40.pth
# acc : 0.955

# 2019-09-03_03-12-17
# dataset	dataset_size	train_test_split	model	model_depth	resume	resume_path	batch_size	n_epochs	sample_size	crop_scale	crop_transform	cropped_data_ratio
# Shopee		0.8	inception	18	True	./results/2019-09-02_11-26-29/saved_models/save_20.pth	16	20	299	0.5	RandomCropRandomScale	0.5
# Inception
python test.py --gpu_id 3 --model inception --model_folder ./results/2019-09-03_03-12-17/saved_models/ --model_file save_40.pth
# acc : 0.9568


# 2019-09-03_05-01-44
# dataset	dataset_size	train_test_split	model	model_depth	resume	resume_path	batch_size	n_epochs	sample_size	crop_scale	crop_transform	cropped_data_ratio
# Shopee		0.8	vgg	19	True	./results/2019-09-02_12-45-34/saved_models/save_20.pth	32	20	224	0.5	MultiScaleCornerCrop	0.5
# VGG-19 Center crop on SOP and on Shopee
#python test.py --gpu_id 0 --model vgg --model_depth 19 --model_folder ./results/2019-09-03_05-01-44/saved_models/ --model_file save_40.pth
# acc: 0.95088
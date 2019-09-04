import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()

    # GPU Arguments

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')

    parser.add_argument(
        '--gpu_id',
        type=str,
        default='1',
        help='GPU no')

    parser.add_argument(
        '--seed',
        type=int,
        default=1111,
        metavar='S',
        help='random seed (default: 1)')

    # Dataset Arguments

    parser.add_argument(
        '--n_channels',
        type=int,
        default=1,
        help='Number of channels in the input images (1 for grayscale)')

    parser.add_argument(
        '--crop_position',
        default='random',
        type=str,
        help='Position of the crop window used to synthesize samples (random | center)')

    parser.add_argument(
        '--crop_scale',
        default=0.5,
        type=float,
        help='Scale for the cropping')

    parser.add_argument(
        '--cropped_data_ratio',
        default=0.5,
        type=float,
        help='Proportion of the data that is left uncropped as a class with a label 0')

    parser.add_argument(
        '--dataset_size',
        default=None,
        type=int,
        help='Total amount of data used for testing')

    parser.add_argument(
        '--train_test_split',
        default=0.8,
        type=float,
        help='Train/test data proportion')


    # Path Arguments

    parser.add_argument(
        '--root_path',
        default='./',
        type=str,
        help='Root directory path of the project')

    parser.add_argument(
        '--dataset',
        default='Shopee',
        type=str,
        help='Root directory path of data')

    parser.add_argument(
        '--data_folder',
        default='data/',
        type=str,
        help='Root directory path of data')

    parser.add_argument(
        '--model_folder',
        default='results/saved_models',
        type=str,
        help='Resume directory')

    parser.add_argument(
        '--model_file',
        default='save_10.pth',
        type=str,
        help='Model .pth file')

    parser.add_argument(
        '--result_folder',
        default='results/',
        type=str,
        help='Result directory')

    parser.add_argument(
        '--log_same_folder',
        action='store_true',
        default=True,
        help='Saves the test log in the same subfolder as the model file')
    # parser.set_defaults(log_same_folder=False)

    parser.add_argument(
        '--log_folder',
        default='logs/',
        type=str,
        help='Log directory folder')

    # Model arguments

    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(derpnet | alexnet | resnet)')

    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')

    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=False,
        help='Uses model pretrained on imagenet if true')


    # Testing Arguments

    parser.add_argument(
        '--batch_size',
        default=128,
        type=int,
        help='Batch Size')

    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=False,
        help='Shuffle or not')

    parser.add_argument(
        '--num_workers',
        default=0,
        type=int,
        help='Number of threads')

    # misc

    parser.add_argument(
        '--show_test_images',
        action='store_true',
        default=True,
        help='Visualize testing batch or not')

    parser.add_argument(
        '--plot_interval',
        default=1,
        type=int,
        help='how many batches to wait before plotting test images')


    args = parser.parse_args()
    args = post_process_args(args)

    return args


from datetime import datetime

def post_process_args(args):

    if args.log_same_folder and args.model_folder:
        args.result_folder = os.path.join(args.model_folder, os.pardir)

    if args.data_folder and args.dataset:
        args.data_path = os.path.join(args.root_path, args.data_folder, args.dataset)

    if args.model_folder:
        args.model_path = os.path.join(args.root_path, args.model_folder, args.model_file)

    if args.root_path != '':
        args.result_path = os.path.join(args.root_path, args.result_folder)
    else:
        args.result_path = os.path.join(os.path.curdir, args.result_folder)

    args.log_path = os.path.join(args.result_path, args.log_folder)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    if args.model == 'resnet':
        args.arch = '{}-{}'.format(args.model, args.model_depth)
    else:
        args.arch = '{}'.format(args.model)

    return args
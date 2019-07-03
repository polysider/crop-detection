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
        '--dataset',
        default='SOP',
        type=str,
        help='Used dataset (MNIST | SOP | Shopee)')

    parser.add_argument(
        '--crop_anomaly',
        action='store_true',
        default=False,
        help='Uses the special version of the dataset for the Binary split of cropped/non-cropped images')
    # parser.set_defaults(unique_result_folder=False)

    parser.add_argument(
        '--n_channels',
        type=int,
        default=1,
        help='Number of channels in the input images (1 for grayscale)')

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
        help='Total amount of data used for training and validation')

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
        '--data_folder',
        default='data/',
        type=str,
        help='Root directory path of data')

    parser.add_argument(
        '--pretrain_folder',
        default='pretrained_models/resnet-18-kinetics.pth',
        type=str,
        help='Location of the pretrained model file (.pth)')

    parser.add_argument(
        '--resume_folder',
        default='results/saved_models/',
        type=str,
        help='Resume directory')

    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='avoids creation of the result folders')

    parser.add_argument(
        '--result_folder',
        default='results/',
        type=str,
        help='Result directory')

    parser.add_argument(
        '--unique_result_folder',
        action='store_true',
        default=True,
        help='Stores results in a separate folder so they do not get overwritten later')
    # parser.set_defaults(unique_result_folder=False)

    parser.add_argument(
        '--log_folder',
        default='logs/',
        type=str,
        help='Log directory folder')

    parser.add_argument(
        '--save_model_folder',
        default='saved_models/',
        type=str,
        help='Save models at this directory')

    # Model Arguments

    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(derpnet | alexnet | resnet')

    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')

    parser.add_argument(
        '--n_pretrained_classes',
        default=1000,
        type=int,
        help=
        'Number of classes of the pretrained model'
    )
    parser.add_argument(
        '--n_finetune_classes',
        default=10,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.'
    )


    # Training Arguments

    parser.add_argument(
        '--batch_size',
        default=50,
        type=int,
        help='Batch Size')

    parser.add_argument(
        '--log_interval',
        default=50,
        type=int,
        help='how many batches to wait before logging training status')

    parser.add_argument(
        '--start_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )

    parser.add_argument(
        '--n_epochs',
        default=10,
        type=int,
        help='Number of total epochs to run')

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

    parser.add_argument(
        '--validation_split',
        default=0.1,
        type=float,
        help='Portion of data selected for validation')

    parser.add_argument(
        '--pretrained',
        action='store_true',
        default=False,
        help='Uses model pretrained on imagenet if true')

    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='Uses model from previous training')

    parser.add_argument(
        '--checkpoint',
        default=10,
        type=int,
        help='Trained model is saved at every this epochs.')

    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )


    # Optimizer Arguments

    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')

    parser.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        help='Momentum')

    parser.add_argument(
        '--dampening',
        default=0.9,
        type=float,
        help='dampening of SGD')

    parser.add_argument(
        '--weight_decay',
        default=1e-3,
        type=float,
        help='Weight Decay')

    parser.add_argument(
        '--nesterov',
        action='store_true',
        help='Nesterov momentum')
    parser.set_defaults(nesterov=False)


    # misc

    parser.add_argument(
        '--show_training_images',
        action='store_true',
        default=True,
        help='Visualize training batch or not')

    parser.add_argument(
        '--plot_interval',
        default=50,
        type=int,
        help='how many batches to wait before plotting test images')


    args = parser.parse_args()
    args = post_process_args(args)

    return args


from datetime import datetime

def post_process_args(args):

    if args.data_folder and args.dataset:
        args.data_path = os.path.join(args.root_path, args.data_folder, args.dataset)

    if args.resume_folder:
        args.resume_path = os.path.join(args.root_path, args.resume_folder)
    if args.pretrain_folder:
        args.pretrain_path = os.path.join(args.root_path, args.pretrain_folder)

    if args.model == 'resnet':
        args.arch = '{}-{}'.format(args.model, args.model_depth)
    elif args.model == 'vgg':
        args.arch = '{}-{}'.format(args.model, args.model_depth)
    else:
        args.arch = '{}'.format(args.model)

    if not args.debug:

        if args.root_path != '':
            args.result_path = os.path.join(args.root_path, args.result_folder)
        else:
            args.result_path = os.path.join(os.path.curdir, args.result_folder)

        if args.unique_result_folder:
            subdir = datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M-%S')
            result_dir = os.path.join(args.result_path, subdir)
            if not os.path.isdir(result_dir):  # Create the log directory if it doesn't exist
                os.makedirs(result_dir)
            args.result_path = result_dir

        args.log_path = os.path.join(args.result_path, args.log_folder)
        if not os.path.exists(args.log_path):
            os.mkdir(args.log_path)

        args.plot_path = os.path.join(args.result_path, 'plots')
        if not os.path.exists(args.plot_path):
            os.mkdir(args.plot_path)

        args.save_model_path = os.path.join(args.result_path, args.save_model_folder)
        if not os.path.exists(args.save_model_path):
            os.mkdir(args.save_model_path)

    return args
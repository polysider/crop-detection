import os
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
import random
import numpy as np
from dataset.data_loading import MNISTLoader, CroppedMNISTLoader, CroppedSOPLoader, ShopeeDataLoader

from models import losses

from utils.parser import parse_args
from utils.model_loading import get_model
from utils.transform_loading import get_train_transform, get_test_transform, get_crop_transform
from utils.training import Trainer
from utils.visualize import showgrid

from utils.logger import Logger

from utils.visualize import tensors_to_imgs_mnist, tensors_to_imgs

def main(args):

    # args.n_epochs = 10
    # args.crop_scale = 0.3
    # args.batch_size = 128
    args.normal_data_ratio = 0.9

    if args.dataset == 'MNIST':
        args.sample_size = 28
    elif args.dataset == 'SOP' or 'Shopee':
        if args.model == 'resnet':
            args.sample_size = 224
        elif args.model == 'vgg' or args.model == 'vgg_attn':
            args.sample_size = 224
        elif args.model == 'inception':
            args.sample_size = 299
        else:
            args.sample_size = 224

    spatial_transform_train = get_train_transform(args)
    crop_transform = get_crop_transform(args)

    if args.dataset == 'MNIST':
        train_data_loader = CroppedMNISTLoader(args, crop_transform=crop_transform,
                                               spatial_transform=spatial_transform_train, training=True)

    elif args.dataset == 'SOP':
        train_data_loader = CroppedSOPLoader(args, crop_transform=crop_transform,
                                               spatial_transform=spatial_transform_train, training=True)

    elif args.dataset == 'Shopee':
        train_data_loader = ShopeeDataLoader(args, crop_transform=crop_transform,
                                               spatial_transform=spatial_transform_train, training=True)

    valid_data_loader = train_data_loader.split_validation()

    args.n_channels = train_data_loader.n_channels
    args.n_classes = train_data_loader.n_classes

    model, parameters = get_model(args)
    model = model.to(device)

    criterion = losses.cross_entropy_loss()

    train_logger = Logger(
        os.path.join(args.log_path, 'train.log'),
        ['epoch', 'loss', 'acc', 'lr'])
    train_batch_logger = Logger(
        os.path.join(args.log_path, 'train_batch.log'),
        ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
    valid_logger = Logger(
        os.path.join(args.log_path, 'val.log'),
        ['epoch', 'loss', 'acc'])

    revision_logger = Logger(
        os.path.join(args.log_path, 'revision_info.log'),
        ['dataset', 'dataset_size', 'train_test_split', 'model', 'model_depth', 'resume', 'resume_path', 'batch_size',
         'n_epochs', 'sample_size', 'crop_scale', 'crop_transform', 'cropped_data_ratio'])
    revision_logger.log({
        'dataset': args.dataset,
        'dataset_size': args.dataset_size,
        'train_test_split': args.train_test_split,
        'model': args.model,
        'model_depth': args.model_depth,
        'resume': args.resume,
        'resume_path': args.resume_path,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'sample_size': args.sample_size,
        'crop_scale': args.crop_scale,
        'crop_transform': crop_transform.__class__.__name__,
        'cropped_data_ratio': args.cropped_data_ratio
    })

    if args.nesterov:
        dampening = 0
    else:
        dampening = args.dampening

    optimizer = optim.SGD(
        parameters,
        lr=args.learning_rate,
        momentum=args.momentum,
        dampening=dampening,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov)

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=args.lr_patience)

    trainer = Trainer(model, criterion, optimizer, args, device, train_data_loader, lr_scheduler=scheduler,
                      valid_data_loader=valid_data_loader, train_logger=train_logger, batch_logger=train_batch_logger, valid_logger=valid_logger)

    trainer.train()




if __name__ == '__main__':

    args = parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print("Model: {}".format(args.model))

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("use_cuda: {}".format(use_cuda))

    # random seeeding
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("Random seed: {}".format(args.seed))

    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device: {}'.format(torch.cuda.get_device_name(torch.cuda.current_device())))
    main(args)
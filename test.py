import os
import torch

import random
import numpy as np
from dataset import data_loading

from utils.parser_test import parse_args
from utils.model_loading import get_model
from utils.transform_loading import get_test_transform, get_crop_transform
from utils.logger import Logger
from utils.moving_average import AverageMeter
from metrics.accuracy import calculate_accuracy
from models.losses import cross_entropy_loss

from utils.visualize import Visualizer, TestVisualizer

def main(args):

    args.sample_size = 28
    args.validation_split = 0.0

    # data for testing
    if args.dataset == 'MNIST':
        args.sample_size = 28
    elif args.dataset == 'SOP':
        if args.model == 'resnet':
            args.sample_size = 224
        else:
            args.sample_size = 28
    elif args.dataset == 'Shopee':
        if args.model == 'resnet':
            args.sample_size = 224

    spatial_transform_test = get_test_transform(args)
    crop_transform = get_crop_transform(args)

    if args.dataset == 'MNIST':
        test_data_loader = data_loading.CroppedMNISTLoader(args, crop_transform=crop_transform,
                                               spatial_transform=spatial_transform_test, training=False)

    elif args.dataset == 'SOP':
        test_data_loader = data_loading.CroppedSOPLoader(args, crop_transform=crop_transform,
                                               spatial_transform=spatial_transform_test, training=False)

    elif args.dataset == 'Shopee':
        test_data_loader = data_loading.ShopeeDataLoader(args, crop_transform=crop_transform,
                                               spatial_transform=spatial_transform_test, training=False)

    args.n_classes = test_data_loader.n_classes

    # prepare the model for testing
    model, parameters = get_model(args)
    model = model.to(device)
    model.eval()

    test_logger = Logger(
        os.path.join(args.log_path, 'test_{}.log'.format(args.dataset)),
        ['batch', 'loss', 'acc'])

    revision_logger = Logger(
        os.path.join(args.log_path, 'test_config_{}.log'.format(args.dataset)),
        ['dataset', 'dataset_size', 'train_test_split', 'n_classes', 'model', 'model_depth',
         'test_batch_size', 'crop_scale', 'cropped_data_ratio', 'shuffle'])
    revision_logger.log({
        'dataset': args.dataset,
        'dataset_size': args.dataset_size,
        'train_test_split': args.train_test_split,
        'n_classes': args.n_classes,
        'model': args.model,
        'model_depth': args.model_depth,
        'test_batch_size': args.batch_size,
        'crop_scale': args.crop_scale,
        'cropped_data_ratio': args.cropped_data_ratio,
        'shuffle': args.shuffle
    })

    # load the trained model weights
    print('loading checkpoint {}'.format(args.model_path))
    checkpoint = torch.load(args.model_path)
    assert args.arch == checkpoint['arch']
    model.load_state_dict(checkpoint['state_dict'])

    criterion = cross_entropy_loss()

    accuracies = AverageMeter()
    losses = AverageMeter()

    visualizer = TestVisualizer(test_data_loader.classes, test_data_loader.rgb_mean, test_data_loader.rgb_std,
                            args)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            acc = calculate_accuracy(outputs, targets)
            accuracies.update(acc, inputs.size(0))

            if args.show_test_images and i % args.plot_interval == 0:
                _, predictions = outputs.topk(1, 1, True)
                predictions = predictions.squeeze().cpu().numpy()
                visualizer.showgrid(inputs, targets, predictions)

            test_logger.log({
                'batch': i+1,
                'loss': losses.avg,
                'acc': accuracies.avg
            })

            print('Batch: [{0}/{1}]\t'
                  'Loss {loss.value:.4f} (avg {loss.avg:.4f})\t'
                  'Acc {acc.value:.3f} (avg {acc.avg:.3f})'.format(
                i + 1,
                len(test_data_loader),
                loss=losses,
                acc=accuracies))

    print('Test log written to {}'.format(test_logger.log_file))



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
# this script is originally written by Eric Song

from os.path import join, isfile, isdir
import os
import sys
import argparse
import pandas as pd
from math import ceil

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms, datasets

from models import resnet
from utils.model_loading import get_model


args = None
img_size = 224


def check_args():
    if not isdir(args['data_dir']):
        print('Error: data dir does not exist')
        return False
    if not isfile(args['model_path']):
        print('Error: model dir does not exist')
        return False
    if args['batch_size'] < 1:
        print('Error: invalid batch size')
        return False
    if args['gpu_id'] < 0:
        print('Error: invalid GPU ID')
        return False
    if args['num_workers'] < 1:
        print('Error: invalid number of workers')
        return False
    # if not isfile(args['map_file']):
    #     print('Error: map file is invalid')
    #     return False
    out_dir = args['out_file'][:args['out_file'].rfind('/')+1]
    if not isdir(out_dir):
        os.mkdir(out_dir)
    return True


def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpu_id'])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print('Loading dataset')
    infer_dataset = datasets.ImageFolder(root=args['data_dir'], transform=data_transform)
    hash_list = [tpl[0][tpl[0].rfind('/')+1:-4] for tpl in infer_dataset.imgs]
    infer_dataloader = DataLoader(infer_dataset,
                                  batch_size=args['batch_size'],
                                  shuffle=False,
                                  num_workers=args['num_workers'])
    print('Loading model')
    model = resnet.resnet18(pretrained=False, num_classes=2)
    model = model.to(device)
    print('Loading checkpoint')
    checkpoint = torch.load(args['model_path'], map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    print('run network')
    pos_scores = []
    counter = 0
    num_batches = ceil(len(hash_list) / args['batch_size'])
    with torch.no_grad():
        for _, (inputs, _) in enumerate(infer_dataloader):
            counter += 1
            sys.stdout.write('\rProcessing batch {0}/{1}'.format(counter, num_batches))
            sys.stdout.flush()
            inputs = inputs.to(device)
            outputs = model(inputs)
            pred_score = nn.functional.softmax(outputs, dim=1)
            pos_score = pred_score[:, 1]
            pos_score = pos_score.cpu().tolist()
            pos_scores.extend(pos_score)
    sys.stdout.write('\n')
    sys.stdout.flush()
    print('Writing output file')
    data_pairs = list(zip(hash_list, pos_scores))
    df = pd.DataFrame(data_pairs, columns=['hash', 'score'])
    df.to_csv(args['out_file'], index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Inference Script')
    parser.add_argument('--data-dir', required=True, help='directory of the data')
    parser.add_argument('--model-path', required=True, help='model path')
    parser.add_argument('--batch-size', type=int, default=512, help='inference batch size')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID')
    parser.add_argument('--num-workers', type=int, default=40, help='num of threads')
    parser.add_argument('--out-file', required=True, help='output file')
    parser.add_argument('--map-file', required=True, help='map file')
    args = vars(parser.parse_args())
    if check_args():
        main()


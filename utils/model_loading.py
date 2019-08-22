import torch
from torch import nn

import copy

from models import derp_net, resnet, alexnet, vgg, inception

def get_model(args):

    assert args.model in [
        'derpnet', 'alexnet', 'resnet' ,'vgg', 'inception'
    ]

    if args.model == 'alexnet':
        model = alexnet.alexnet(pretrained=args.pretrained, n_channels=args.n_channels, num_classes=args.n_classes)
    elif args.model == 'inception':
        model = inception.inception_v3(pretrained=args.pretrained, progress = True, num_classes = args.n_classes)
    elif args.model == 'vgg':
        assert args.model_depth in [11, 13, 16, 19]

        if args.model_depth == 11:
            model = vgg.vgg11_bn(
                pretrained=args.pretrained,
                progress= True,
                num_classes=args.n_classes)
        if args.model_depth == 13:
            model = vgg.vgg13_bn(
                pretrained=args.pretrained,
                progress= True,
                num_classes=args.n_classes)
        if args.model_depth == 16:
            model = vgg.vgg16_bn(
                pretrained=args.pretrained,
                progress= True,
                num_classes=args.n_classes)
        if args.model_depth == 19:
            model = vgg.vgg19(
                pretrained=args.pretrained,
                progress= True,
                num_classes=args.n_classes)     

    elif args.model == 'derpnet':
        model = derp_net.Net(n_channels=args.n_channels, num_classes=args.n_classes)

    elif args.model == 'resnet':
        assert args.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if args.model_depth == 10:
            model = resnet.resnet10(
                pretrained=args.pretrained,
                num_classes=args.n_classes)
        elif args.model_depth == 18:
            model = resnet.resnet18(
                pretrained=args.pretrained,
                num_classes=args.n_classes)
        elif args.model_depth == 34:
            model = resnet.resnet34(
                pretrained=args.pretrained,
                num_classes=args.n_classes)
        elif args.model_depth == 50:
            model = resnet.resnet50(
                pretrained=args.pretrained,
                num_classes=args.n_classes)
        elif args.model_depth == 101:
            model = resnet.resnet101(
                pretrained=args.pretrained,
                num_classes=args.n_classes)
        elif args.model_depth == 152:
            model = resnet.resnet152(
                pretrained=args.pretrained,
                num_classes=args.n_classes)
        elif args.model_depth == 200:
            model = resnet.resnet200(
                pretrained=args.pretrained,
                num_classes=args.n_classes)

    if args.pretrained and args.pretrain_path and not args.model == 'alexnet' and not args.model == 'vgg' and not args.model == 'resnet':

        print('loading pretrained model {}'.format(args.pretrain_path))
        pretrain = torch.load(args.pretrain_path)
        assert args.arch == pretrain['arch']

        # here all the magic happens: need to pick the parameters which will be adjusted during training
        # the rest of the params will be frozen
        pretrain_dict = {key[7:]: value for key, value in pretrain['state_dict'].items() if key[7:9] != 'fc'}
        from collections import OrderedDict
        pretrain_dict = OrderedDict(pretrain_dict)

        # https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
        import types
        model.load_state_dict = types.MethodType(load_my_state_dict, model)

        old_dict = copy.deepcopy(model.state_dict()) # normal copy() just gives a reference
        model.load_state_dict(pretrain_dict)
        new_dict = model.state_dict()

        num_features = model.fc.in_features
        if args.model == 'densenet':
            model.classifier = nn.Linear(num_features, args.n_classes)
        else:
            #model.fc = nn.Sequential(nn.Linear(num_features, 1028), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1028, args.n_finetune_classes))
            model.fc = nn.Linear(num_features, args.n_classes)

        # parameters = get_fine_tuning_parameters(model, args.ft_begin_index)
        parameters = model.parameters()  # fine-tunining EVERYTHIIIIIANG
        # parameters = model.fc.parameters()  # fine-tunining ONLY FC layer
        return model, parameters

    return model, model.parameters()


# custom weights loading from the state dict
def load_my_state_dict(self, state_dict):

    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

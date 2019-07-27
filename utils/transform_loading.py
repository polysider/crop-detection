from utils.spatial_transforms import CenterCrop, MultiScaleCornerCrop, Scale, Compose, RandomCrop


def get_train_transform(args):

    spatial_transform = Scale([args.sample_size, args.sample_size])
    #spatial_transform = CenterCrop(args.sample_size)
    # spatial_transform = MultiScaleCornerCrop([args.scale], args.sample_size, crop_positions='c')
    return spatial_transform


def get_crop_transform(args):

    # spatial_transform = None
    spatial_transform = RandomCrop(args.sample_size)
    #spatial_transform = MultiScaleCornerCrop([args.crop_scale], args.sample_size, crop_positions='c')
    #spatial_transform = MultiScaleCornerCrop([args.crop_scale], args.sample_size, crop_positions='tl')
    #spatial_transform = MultiScaleCornerCrop([args.crop_scale], args.sample_size, crop_positions='tr')
    #spatial_transform = MultiScaleCornerCrop([args.crop_scale], args.sample_size, crop_positions='bl')
    #spatial_transform = MultiScaleCornerCrop([args.crop_scale], args.sample_size, crop_positions='br')

    return spatial_transform


def get_test_transform(args):

    spatial_transform = Scale([args.sample_size, args.sample_size])
    return spatial_transform
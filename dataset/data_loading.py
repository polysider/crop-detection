from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataset.base_data_loader import BaseDataLoader
from dataset.cropped_MNIST import CroppedMNIST
from dataset.shopee_data import ShopeeData
from dataset.cropped_stanford_online_products import CroppedStanfordOnlineProducts


class MNISTLoader(BaseDataLoader):
    """
    Standard MNIST data loader
    """

    sample_size = 28
    n_channels = 1

    def __init__(self, args, spatial_transform, training=True):

        data_dir = args.data_path
        batch_size = args.batch_size
        shuffle = args.shuffle
        split = args.validation_split
        num_workers = args.num_workers

        if spatial_transform != None:
            transform = transforms.Compose([
                spatial_transform,
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=transform)
        self.dataset.my_classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
                                   '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
        super(MNISTLoader, self).__init__(self.dataset, batch_size, shuffle, split, num_workers)


class CroppedMNISTLoader(BaseDataLoader):
    """
    This verison of MNIST has only 2 classes: non-cropped samples and cropped ones
    """

    sample_size = 28
    n_channels = 1

    rgb_mean = (0.1307,)
    rgb_std = (0.3081,)

    def __init__(self, args, crop_transform, spatial_transform=None, training=True):

        data_dir = args.data_path
        batch_size = args.batch_size
        shuffle = args.shuffle
        split = args.validation_split
        num_workers = args.num_workers

        if spatial_transform != None:
            transform = transforms.Compose([
                spatial_transform,
                transforms.ToTensor(),
                transforms.Normalize(self.rgb_mean, self.rgb_std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.rgb_mean, self.rgb_std)
            ])
        self.data_dir = data_dir
        self.dataset = CroppedMNIST(self.data_dir, train=training, download=True, transform=transform,
                                    crop_transform=crop_transform,
                                    cropped_data_ratio=args.cropped_data_ratio,
                                    dataset_size=args.dataset_size)
        super(CroppedMNISTLoader, self).__init__(self.dataset, batch_size, shuffle, split, num_workers)


class CroppedSOPLoader(BaseDataLoader):
    """
    This verison of MNIST has only 2 classes: non-cropped samples and cropped ones
    """
    sample_size = 64
    n_channels = 3

    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]

    def __init__(self, args, crop_transform, spatial_transform=None, training=True):

        data_dir = args.data_path
        batch_size = args.batch_size
        shuffle = args.shuffle
        split = args.validation_split
        num_workers = args.num_workers

        if spatial_transform != None:
            transform = transforms.Compose([
                spatial_transform,
                transforms.ToTensor(),
                transforms.Normalize(self.rgb_mean, self.rgb_std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.rgb_mean, self.rgb_std)
            ])
        self.data_dir = data_dir
        self.dataset = CroppedStanfordOnlineProducts(self.data_dir, train=training, download=True, transform=transform,
                                                     crop_transform=crop_transform,
                                                     cropped_data_ratio=args.cropped_data_ratio,
                                                     dataset_size=args.dataset_size)
        super(CroppedSOPLoader, self).__init__(self.dataset, batch_size, shuffle, split, num_workers)


class ShopeeDataLoader(BaseDataLoader):
    """
    herp derp
    """
    sample_size = 64
    n_channels = 3

    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]

    def __init__(self, args, crop_transform, spatial_transform=None, training=True):

        data_path = args.data_path
        batch_size = args.batch_size
        shuffle = args.shuffle
        split = args.validation_split
        num_workers = args.num_workers

        if spatial_transform != None:
            transform = transforms.Compose([
                spatial_transform,
                transforms.ToTensor(),
                transforms.Normalize(self.rgb_mean, self.rgb_std)
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.rgb_mean, self.rgb_std)
            ])
        self.data_path = data_path
        self.dataset = ShopeeData(self.data_path, train=training, train_test_split=args.train_test_split,
                                                     unzip=True, transform=transform,
                                                     crop_transform=crop_transform,
                                                     cropped_data_ratio=args.cropped_data_ratio,
                                                     dataset_size=args.dataset_size)
        super(ShopeeDataLoader, self).__init__(self.dataset, batch_size, shuffle, split, num_workers)
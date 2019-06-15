import os
import csv
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_url
from torchvision.datasets.folder import default_loader

from PIL import Image


def has_file_allowed_extension(filename, extensions):

    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):

    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):

    samples = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    samples.append(item)

    return samples


class ShopeeData(ImageFolder, CIFAR10):
    """
        The Shopee dataset of 5000 negative and 1000 positive samples.
        Has 0/1 labels for non-cropped/cropped samples
        Args:
            root (string): Root directory of dataset.
            train (bool, optional): If True, creates dataset from ``Ebay_train.txt``,
                otherwise from ``Ebay_test.txt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
    """
    base_folder = 'Shopee'
    filenames = ['crop_neg.tar.gz', 'crop_pos.tar.gz']
    zip_md5 = ['', '']
    my_classes = ['0 - not cropped', '1 - cropped']

    def _get_md5(self):
        rdr = csv.reader('valid.csv', delimiter=',')
        train_list = []
        for row in rdr:
            train_list.append(row)

    def _prepare_split_old(self, split, dataset_size, cropped_data_ratio):

        if dataset_size is not None and dataset_size < len(self.samples):
            self.samples = self.samples[:dataset_size]

        split_idxs = [i for i in range(len(self.samples)) if i < len(self.samples) * split]
        samples = self.samples[split_idxs]

        self.cropped = [False] * len(self.samples)

        for i in range(len(self.samples)):
            if i < len(self.samples) * cropped_data_ratio:
                self.cropped[i] = True

        return samples

    def _split_array(self, array, split):
        end = len(array)
        threshold = round(end*split)
        array_1 = array[:threshold]
        array_2 = array[threshold:end]
        return array_1, array_2

    def _prepare_split(self, train, split, dataset_size, cropped_data_ratio):

        negative_samples_idxs = [i for i, sample in enumerate(self.samples) if sample[1] == 0]
        positive_samples_idxs = [i for i, sample in enumerate(self.samples) if sample[1] == 1]

        if dataset_size is not None and dataset_size < len(self.samples):
            self.samples = self.samples[:dataset_size]

        # train_split_negative = negative_samples_idxs[:round(len(negative_samples_idxs)*split)]
        # train_split_positive = positive_samples_idxs[:round(len(positive_samples_idxs)*split)]
        #
        # test_split_negative = negative_samples_idxs[round(len(negative_samples_idxs)*split): len(negative_samples_idxs)]
        # test_split_positive = positive_samples_idxs[round(len(positive_samples_idxs)*split): len(positive_samples_idxs)]

        train_split_negative, test_split_negative = self._split_array(negative_samples_idxs, split)
        train_split_positive, test_split_positive = self._split_array(positive_samples_idxs, split)

        self.cropped = [False] * len(self.samples)

        for i, sample_index in enumerate(negative_samples_idxs):
            if i < len(negative_samples_idxs) * cropped_data_ratio:
                self.cropped[sample_index] = True

        train_split = train_split_negative + train_split_positive
        test_split = test_split_negative + test_split_positive

        # split_idxs = [i for i in range(len(self.samples)) if i < len(self.samples) * split]
        # train_samples = self.samples[train_split_positive]

        if train:
            samples = [self.samples[idx] for idx in train_split]
            to_be_cropped = [self.cropped[idx] for idx in train_split]
        else:
            samples = [self.samples[idx] for idx in test_split]
            to_be_cropped = [self.cropped[idx] for idx in test_split]

        return samples, to_be_cropped


    def __init__(self, root, train=False, train_test_split=0.8, transform=None, target_transform=None, crop_transform=None, download=False,
                 cropped_data_ratio=0.5, dataset_size=None, unzip=True, **kwargs):

        self.root = root
        if unzip:
            self.unzip()

        super().__init__(root, transform, target_transform, loader=default_loader)

        self.crop_transform = crop_transform

        # classes = self.my_classes
        # class_to_idx = {classes[i]: i for i in range(len(classes))}
        classes, class_to_idx = find_classes(root)
        extensions = ['jpg']
        self.samples = make_dataset(root, class_to_idx, extensions)
        self.train_test_split = train_test_split
        self.data_list = self._get_md5()
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use unzip=True to extract it')

        self.samples, self.cropped = self._prepare_split(train, train_test_split, dataset_size, cropped_data_ratio)


    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                tuple: (image, target) where target is index of the target class.
        """

        img, target = self._prepare_sample(index=index)

        if self.transform is not None:
            img = self.transform(img)

        # Checking here if the image is grayscale, i.e. tensor has the shape of 1,x,x instead of 3,x,x
        # and converting the grayscale image to an rgb (1,x,x tensor to 3,x,x)
        if img.shape[0] == 1:
            img = torch.stack((torch.squeeze(img),)*3)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _prepare_sample(self, index):

        img, target = self.samples[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img)

        # cropped image is getting a 1 label, and the non-cropped one is labeled as 0
        if self.crop_transform is not None:
            if self.cropped[index]:
                target = 1
                img = self.crop_transform(img)
            else:
                target = 0

        return img, target

    def _check_integrity(self):
        root = self.root
        # for fentry in (self.data_list):
        #     filename, md5 = fentry[0], fentry[1]
        #     fpath = os.path.join(root, self.base_folder, filename)
        #     if not check_integrity(fpath, md5):
        #         return False
        return True
        # return False

    def unzip(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        # download_url(self.url, root, self.filename, self.tgz_md5)

        # extract file
        cwd = os.getcwd()

        # os.chdir(root)
        for filename in self.filenames:
            tar = tarfile.open(os.path.join(root, filename), "r:gz")
            os.chdir(root)
            tar.extractall()
            tar.close()
            os.chdir(cwd)
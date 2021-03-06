import os
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_url, check_integrity
from torchvision.datasets.folder import default_loader

from PIL import Image


class CroppedStanfordOnlineProducts(ImageFolder):
    """
        A modified version of the Stanford Online Products dataset.
        Has 0/1 labels for non-cropped/cropped samples
        `Stanford Online Products <http://cvgl.stanford.edu/projects/lifted_struct/>`_ Dataset.
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
    base_folder = 'Stanford_Online_Products'
    url = 'ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip'
    filename = 'Stanford_Online_Products.zip'
    zip_md5 = '7f73d41a2f44250d4779881525aea32e'

    train_list = [
        ['bicycle_final/111265328556_0.JPG', '77420a4db9dd9284378d7287a0729edb'],
        ['chair_final/111182689872_0.JPG', 'ce78d10ed68560f4ea5fa1bec90206ba']
    ]
    test_list = [
        ['table_final/111194782300_0.JPG', '8203e079b5c134161bbfa7ee2a43a0a1'],
        ['toaster_final/111157129195_0.JPG', 'd6c24ee8c05d986cafffa6af82ae224e']
    ]
    my_classes = ['0 - not cropped', '1 - cropped']

    def make_dataset(self):


        pass

    def __init__(self, root, train=False, transform=None, target_transform=None, crop_transform=None, download=False,
                 cropped_data_ratio=0.5, dataset_size=None, **kwargs):

        super().__init__(root, transform, target_transform, loader=default_loader)

        self.crop_transform = crop_transform

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.samples = [(os.path.join(root, self.base_folder, path), int(class_id) - 1) for
                     i, (image_id, class_id, super_class_id, path) in enumerate(map(str.split, open(
                os.path.join(root, self.base_folder, 'Ebay_{}.txt'.format('train' if train else 'test'))))) if i > 1]

        if dataset_size is not None and dataset_size < len(self.samples):
            self.samples = self.samples[:dataset_size]

        self.cropped = [False] * len(self.samples)

        for i in range(len(self.samples)):
            if i < len(self.samples) * cropped_data_ratio:
                self.cropped[i] = True

    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                tuple: (image, target) where target is index of the target class.
        """

        img, target = self._prepare_sample(index=index)

        img = img.convert('RGB')

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

        # cropped image is getting a 1 label, and the non-cropped one is labeled as 0
        if self.crop_transform is not None:
            img, target = self.samples[index]
            img = Image.open(img)
            if self.cropped[index]:
                target = 1
                img = self.crop_transform(img)
            else:
                target = 0
        else:
            img, target = self.samples[index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            img = Image.open(img)

        return img, target

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import zipfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.zip_md5)

        # extract file
        cwd = os.getcwd()
        os.chdir(root)
        with zipfile.ZipFile(self.filename, "r") as zip:
            zip.extractall()
        os.chdir(cwd)

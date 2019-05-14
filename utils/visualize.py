"""
Methods for showing images
"""
import os

import numpy as np
from matplotlib import pyplot as plt
from utils.norm import UnNormalize


class Visualizer:

    def __init__(self, classes, mean, std, args, show_all=False, samples_per_class=4):
        """
        inputs:
        imgs: list of Torch cpu tensors of NumImgs x n_channels x W x H
        labels: list of class labels of length = NumImgs
        classes: list of all possible class labels
        """
        self.args = args

        self.classes = classes
        self.mean = mean
        self.std = std
        self.num_classes = len(self.classes)
        self.plot_dir = self.args.plot_path if hasattr(self.args, 'plot_path') else None
        self.show_all = show_all
        self.samples_per_class = samples_per_class
        self.grayscale = True if args.n_channels == 1 else False

        plt.figure()
        plt.rcParams['figure.figsize'] = (16.0, 12.0)  # set default size of plots
        plt.rcParams['image.interpolation'] = 'nearest'
        plt.rcParams['image.cmap'] = 'gray'
        plt.axis('off')

    def tensors_to_imgs(self, tensors, mean, std):

        # npimgs = np.asarray([img.numpy() for img in imgs])
        npimgs = []

        for tensor in tensors:
            unorm = UnNormalize(mean=mean, std=std)
            tensor = unorm(tensor)
            npimg = tensor.cpu().numpy()
            npimg = np.clip(npimg, 0, 1)
            npimg = np.transpose(npimg, (1, 2, 0))
            npimg = npimg.squeeze()
            npimgs.append(npimg)

        return np.array(npimgs)

    def showgrid(self, tensors, targets, batch):
        """
        a lot of bad code down below. Please proceed with caution
        inputs:
        tensors: list of Torch cpu tensors of NumImgs x n_channels x W x H
        targets: list of Torch cpu tensors for class labels of length = NumImgs
        batch: ordinal number of a batch I guess
        """

        npimgs = self.tensors_to_imgs(tensors, self.mean, self.std)
        labels = targets.cpu().numpy()

        title = "Few samples from the batch number {}".format(batch + 1)

        for y, cls in enumerate(self.classes):
            idxs = np.flatnonzero(labels == y)
            if not len(idxs) == 0 and not self.show_all:
                idxs = np.random.choice(idxs, self.samples_per_class, replace=True)
            self.samples_per_class = len(idxs) if len(idxs) > self.samples_per_class else self.samples_per_class

            for i, idx in enumerate(idxs):
                plt_idx = i * self.num_classes + y + 1
                plt.subplot(self.samples_per_class, self.num_classes, plt_idx)
                img = npimgs[idx]
                img = np.squeeze(img)
                if self.grayscale:
                    img = np.stack((img,) * 3)
                    img = np.transpose(img, (1, 2, 0))
                plt.imshow(img)
                plt.axis('off')
                if i == 0:
                    plt.title(cls)
        if title is not None:
            plt.suptitle(title, fontsize=16)
        if self.plot_dir is not None:
            plot_name = title if title is not None else 'pictures.png'
            img_path = os.path.join(self.plot_dir, plot_name)
            plt.savefig(img_path)
        plt.show(block=False)
        plt.draw()
        plt.pause(0.5)



class TestVisualizer(Visualizer):

    def __init__(self, classes, mean, std, args):
        """
                inputs:
                imgs: list of Torch cpu tensors of NumImgs x n_channels x W x H
                labels: list of class labels of length = NumImgs
                classes: list of all possible class labels
                """
        self.args = args

        self.classes = classes
        self.mean = mean
        self.std = std
        self.num_classes = len(self.classes)
        self.plot_dir = self.args.plot_path if hasattr(self.args, 'plot_path') else None
        self.grayscale = True if args.n_channels == 1 else False

        fig_size = [10, 10]
        plt.rcParams["figure.figsize"] = fig_size

        self.fig, self.axes = plt.subplots(5, 5)
        self.fig.subplots_adjust(hspace=0.2, wspace=0.2)


    def showgrid(self, tensors, targets, predictions=None):

        npimgs = self.tensors_to_imgs(tensors, self.mean, self.std)
        targets = targets.cpu().numpy()

        for i, ax in enumerate(self.axes.flat):
            ax.set_xticks([])
            ax.set_yticks([])

        for i, npimage in enumerate(npimgs):

            if i < 25:
                # Plot image.
                self.axes.flat[i].imshow(npimage)

                # True vs predicted labels
                if predictions is None:
                    xlabel = "True: {0}".format(targets[i])
                else:
                    xlabel = "True: {0}, Pred: {1}".format(targets[i], predictions[i])

                self.axes.flat[i].set_xlabel(xlabel)
                self.axes.flat[i].set_xticks([])
                self.axes.flat[i].set_yticks([])

        # Draw the plot
        plt.show(block=False)
        plt.draw()
        plt.pause(0.5)


# rubbish below


def tensors_to_imgs(tensors, mean, std):

    # npimgs = np.asarray([img.numpy() for img in imgs])
    npimgs = []

    for tensor in tensors:
        unorm = UnNormalize(mean=mean, std=std)
        tensor = unorm(tensor)
        npimg = tensor.numpy()
        npimg = np.clip(npimg, 0, 1)
        npimg = np.transpose(npimg, (1, 2, 0))
        npimg = npimg.squeeze()
        npimgs.append(npimg)

    return np.array(npimgs)


def showgrid(npimgs, labels, classes, grayscale=False, title=None, log_dir=None, samples_per_class=8, show_all=False):
    """
    inputs:
    imgs: list of Torch cpu tensors of NumImgs x 1 x 32 x 32
    labels: list of class labels of length = NumImgs
    classes: list of all possible class labels
    """
    plt.figure()
    plt.rcParams['figure.figsize'] = (16.0, 12.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    labels = np.asarray(labels)
    plt.axis('off')

    num_classes = len(classes)

    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(labels == y)
        if not len(idxs) == 0 and not show_all:
            idxs = np.random.choice(idxs, samples_per_class, replace=True)
        samples_per_class = len(idxs) if len(idxs) > samples_per_class else samples_per_class

        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            img = npimgs[idx]
            img = np.squeeze(img)
            if grayscale:
                img = np.stack((img,) * 3)
                img = np.transpose(img, (1, 2, 0))
            plt.imshow(img)
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    if title is not None:
        plt.suptitle(title, fontsize=16)
    if log_dir is not None:
        if title is not None:
            plot_name = title
        else:
            plot_name = 'pictures.png'
        img_path = os.path.join(log_dir, 'training_samples/', plot_name)
        plt.savefig(img_path)
    plt.show(block=False)
    plt.draw()
    plt.pause(0.5)




def tensors_to_imgs_mnist(imgs):

    # npimgs = np.asarray([img.numpy() for img in imgs])
    npimgs = []

    for img in imgs:
        npimg = img.numpy()
        mean = np.array([0.1307])
        std = np.array([0.3081])
        npimg = std * npimg + mean  # unnormalize
        npimg = np.clip(npimg, 0, 1)
        npimg = np.transpose(npimg, (1, 2, 0))
        npimg = npimg.squeeze()
        npimgs.append(npimg)

    return np.array(npimgs)
import torch
import time
import os
import numpy as np

from utils.moving_average import AverageMeter
from metrics.accuracy import calculate_accuracy
from utils.visualize import Visualizer, TestVisualizer


class Trainer:

    def __init__(self, model, criterion, optimizer, args, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, train_logger=None, batch_logger=None, valid_logger=None):

        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_logger = train_logger
        self.batch_logger = batch_logger
        self.valid_logger = valid_logger
        self.args = args
        self.resume = self.args.resume
        self.resume_path = self.args.resume_path
        self.device = device
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.start_epoch = args.start_epoch
        self.epochs = args.n_epochs
        self.visualizer = Visualizer(self.data_loader.classes, self.data_loader.rgb_mean, self.data_loader.rgb_std,
                                     self.args)
        # self.visualizer_test = TestVisualizer(self.data_loader.classes, self.data_loader.rgb_mean,
        #                                       self.data_loader.rgb_std, 6, self.args)

    def train(self):
        """
        Full training cycle
        :return:
        """

        if self.resume and self.resume_path:
            print('loading checkpoint {}'.format(self.resume_path))
            checkpoint = torch.load(self.resume_path)
            assert self.args.arch == checkpoint['arch']

            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            result = self._train_epoch(epoch)

        return result


    def _train_epoch(self, epoch):

        print('train at epoch {}'.format(epoch))

        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()
        samples_used = AverageMeter()

        end_time = time.time()

        for i, (inputs, targets) in enumerate(self.data_loader):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            data_time.update(time.time() - end_time)
            outputs = self.model(inputs)

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                _, predictions = outputs.topk(1, 1, True)

                if self.args.show_training_images and i % self.args.plot_interval == 0:
                    batch_number = (epoch - 1) * len(self.data_loader) + i
                    self.visualizer.showgrid(inputs, targets, batch_number)
                    # self.visualizer_test.make_grid(inputs, targets, predictions)
                    # self.visualizer_test.show()

            with torch.no_grad():

                losses.update(loss.item(), inputs.size(0))
                acc = calculate_accuracy(outputs, targets)
                accuracies.update(acc, inputs.size(0))

                batch_time.update(time.time() - end_time)
                end_time = time.time()
                samples_used.update(inputs.size(0))

                self.batch_logger.log({
                    'epoch': epoch,
                    'batch': i + 1,
                    'iter': (epoch - 1) * len(self.data_loader) + (i + 1),
                    'loss': losses.value,
                    'acc': accuracies.value,
                    'lr': self.optimizer.param_groups[0]['lr']
                })

                if i + 1 == len(self.data_loader):
                    print("wait a minute")

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Samples used: {3}\t'
                      'Time {batch_time.value:.3f} (avg {batch_time.avg:.3f})\t'
                      'Data {data_time.value:.3f} (avg {data_time.avg:.3f})\t'
                      'Loss {loss.value:.4f} (avg {loss.avg:.4f})\t'
                      'Acc {acc.value:.3f} (avg {acc.avg:.3f})'.format(
                          epoch,
                          i + 1,
                          len(self.data_loader),
                          samples_used.sum,
                          batch_time=batch_time,
                          data_time=data_time,
                          loss=losses,
                          acc=accuracies))


        self.train_logger.log({
            'epoch': epoch,
            'loss': losses.avg,
            'acc': accuracies.avg,
            'lr': self.optimizer.param_groups[0]['lr']
        })

        if epoch % self.args.checkpoint == 0:
            save_file_path = os.path.join(self.args.save_model_path, 'save_{}.pth'.format(epoch))
            states = {
                'epoch': epoch, # was epoch + 1
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
            torch.save(states, save_file_path)

        log = {
            'loss': losses.avg,
            'metrics': accuracies
        }

        if self.do_validation:
            val_log, validation_loss = self._validate_epoch(epoch)
            log = {**log, **val_log}

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(validation_loss)

        return log


    def _validate_epoch(self, epoch):

        print('validation at epoch {}'.format(epoch))

        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accuracies = AverageMeter()

        end_time = time.time()

        for i, (inputs, targets) in enumerate(self.valid_data_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            data_time.update(time.time() - end_time)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Validation Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.value:.3f} (avg {batch_time.avg:.3f})\t'
                  'Data {data_time.value:.3f} (avg {data_time.avg:.3f})\t'
                  'Loss {loss.value:.4f} (avg {loss.avg:.4f})\t'
                  'Acc {acc.value:.3f} (avg {acc.avg:.3f})'.format(
                epoch,
                i + 1,
                len(self.valid_data_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                acc=accuracies))

        self.valid_logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg})

        log = {
            'loss': losses.avg,
            'metrics': accuracies
        }

        return log, loss



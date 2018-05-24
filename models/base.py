import os
import sys
import time
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.models import load_model
from abc import ABCMeta, abstractmethod

from .utils import *

try:
    from .notifyier import *
except ImportError as e:
    print(e)
    print("You did not set a notifyier. Notifications will not be sent anywhere")


class BaseModel(metaclass=ABCMeta):
    '''
    Base class for non-conditional generative networks
    '''

    def __init__(self, **kwargs):
        '''
        Initialization
        '''
        if 'name' not in kwargs:
            raise Exception('Please specify model name!')
        self.current_epoch = 0

        self.run_id = kwargs.get('run_id', 0)

        self.name = "{}_r{}".format(kwargs['name'], self.run_id)

        if 'input_shape' not in kwargs:
            raise Exception('Please specify input shape!')

        self.input_shape = kwargs['input_shape']

        if 'output' not in kwargs:
            self.output = 'output'
        else:
            self.output = kwargs['output']

        self.test_mode = kwargs.get('test_mode', False)
        self.trainers = {}
        self.last_epoch = 0
        self.dataset = None
        self.g_losses, self.d_losses, self.losses_ratio = [], [], []
        self.label_smoothing = kwargs.get('label_smoothing', 0.0)
        self.input_noise = kwargs.get('input_noise', 0.0)

        self.checkpoint_every = kwargs.get('checkpoint_every', 1)
        self.notify_every = kwargs.get('notify_every', self.checkpoint_every)
        self.lr = kwargs.get('lr', 1e-4)
        self.dis_loss_control = kwargs.get('dis_loss_control', 1.0)

    def get_experiment_id(self):
        return self.name

    def _get_experiment_id(self):
        return self.get_experiment_id()
    experiment_id = property(_get_experiment_id)

    def main_loop(self, dataset, samples, samples_conditionals=None, epochs=100, batchsize=100, reporter=[], ):
        '''
        Main learning loop
        '''

        # Create output directories if not exist
        self.dataset = dataset
        out_dir = os.path.join(self.output, self.experiment_id)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        res_out_dir = os.path.join(out_dir, 'results')
        if not os.path.isdir(res_out_dir):
            os.mkdir(res_out_dir)

        wgt_out_dir = os.path.join(out_dir, 'weights')
        if not os.path.isdir(wgt_out_dir):
            os.mkdir(wgt_out_dir)

        # Start training
        print('\n\n--- START TRAINING ---\n')
        num_data = len(dataset)
        self.dataset = dataset
        self.g_losses, self.d_losses, self.losses_ratio = [], [], []
        for e in range(self.last_epoch, epochs):
            perm = np.random.permutation(num_data)
            start_time = time.time()
            self.current_epoch = e
            for b in range(0, num_data, batchsize):

                # account for division of data into batch size
                if batchsize > num_data - b:
                    continue
                bsize = batchsize
                indx = perm[b:b + bsize]

                # slice batch out of dataset
                x_batch, y_batch = self.make_batch(dataset, indx)

                # add optional input noise that decreases over time
                x_batch = add_input_noise(x_batch, curr_epoch=e, total_epochs=epochs, start_noise=self.input_noise)

                # finally, train and report status
                losses = self.train_on_batch(x_batch, y_batch=y_batch)
                print_current_progress(e, b, bsize, len(dataset), losses, elapsed_time=time.time() - start_time)

                # check for collapse scenario where G and D losses are equal
                did_collapse = self.did_collapse(losses)
                if did_collapse:
                    message = "[{}] {}. Stopped at Epoch #{}".format(self.experiment_id, did_collapse, e + 1)
                    try:
                        notify_with_message(message, experiment_id=self.experiment_id)
                    except NameError:
                        pass
                    print(message)
                    exit()

                if self.test_mode:
                    print('\nFinish testing: %s' % self.experiment_id)
                    return

            # plot samples and losses and send notification if it's checkpoint time
            is_checkpoint = ((e + 1) % self.checkpoint_every) == 0
            is_notification_checkpoint = ((e + 1) % self.notify_every) == 0
            outfile = None
            if is_checkpoint:
                outfile = os.path.join(res_out_dir, "epoch_{:04}_batch_{}".format(e + 1, b + bsize))
                self.save_images(samples, "{}_samples.png".format(outfile), conditionals_for_samples=samples_conditionals)
                self.save_model(wgt_out_dir, e + 1)
                self.plot_losses_hist("{}_losses.png".format(outfile))
            if is_notification_checkpoint:
                if not outfile:
                    outfile = os.path.join(res_out_dir, "epoch_{:04}_batch_{}".format(e + 1, b + bsize))
                    self.save_images(samples, "{}_samples.png".format(outfile), conditionals_for_samples=samples_conditionals)
                    self.plot_losses_hist("{}_losses.png".format(outfile))
                try:
                    message = "[{}] Epoch #{:04}".format(self.experiment_id, e + 1)
                    notify_with_image("{}_samples.png".format(outfile), experiment_id=self.experiment_id, message=message)
                    notify_with_image("{}_losses.png".format(outfile), experiment_id=self.experiment_id, message=message)
                except NameError as e:
                    print(e)

            elapsed_time = time.time() - start_time
            print('Took: {}s\n'.format(elapsed_time))
            self.did_train_over_an_epoch()

    def plot_losses_hist(self, outfile):
        plt.plot(self.g_losses, label='Gen')
        plt.plot(self.d_losses, label='Dis')
        plt.legend()
        plt.savefig(outfile)
        plt.close()

    def save_losses_history(self, losses):
        self.g_losses.append(losses['g_loss'])
        self.d_losses.append(losses['d_loss'])
        self.losses_ratio.append(losses['g_loss'] / losses['d_loss'])

    def make_batch(self, dataset, indx):
        '''
        Get batch from dataset
        '''
        data = dataset.images[indx]
        labels = dataset.attrs[indx]
        return data, labels

    def did_collapse(self, losses):
        return False

    def save_images(self, samples, filename, conditionals_for_samples=None):
        '''
        Save images generated from random sample numbers
        '''
        imgs = self.predict(samples)
        # imgs = np.clip(imgs * 0.5 + 0.5, 0.0, 1.0)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))

        self.save_image_as_plot(imgs, filename)

    def save_image_as_plot(self, imgs, filename):
        fig = plt.figure(figsize=(8, 8))
        grid = gridspec.GridSpec(10, 10, wspace=0.1, hspace=0.1)
        for i in range(100):
            ax = plt.Subplot(fig, grid[i])
            if imgs.ndim == 4:
                ax.imshow(imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            else:
                ax.imshow(imgs[i, :, :], cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        fig.savefig(filename, dpi=200)
        plt.close(fig)

    def save_model(self, out_dir, epoch):
        folder = os.path.join(out_dir, 'epoch_%05d' % epoch)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        for k, v in self.trainers.items():
            filename = os.path.join(folder, '%s.hdf5' % (k))
            v.save_weights(filename)

    def store_to_save(self, name):
        self.trainers[name] = getattr(self, name)

    def load_model(self, folder):
        for k, v in self.trainers.items():
            try:
                filename = os.path.join(folder, "{}.hdf5".format(k))
                getattr(self, k).load_weights(filename)
            except OSError as e:
                print(e)
                print("Couldn't load {}. Starting from scratch".format(filename))
            except ValueError as e:
                print(e)
                print("Couldn't load {}. Starting from scratch".format(filename))

        # load epoch number
        epoch = int(folder.split('_')[-1].replace('/', ''))
        self.last_epoch = epoch

    def did_train_over_an_epoch(self):
        pass

    @abstractmethod
    def predict(self, z_sample):
        '''
        Plase override "predict" method in the derived model!
        '''
        pass

    @abstractmethod
    def train_on_batch(self, x_batch, y_batch=None, compute_grad_norms=False):
        '''
        Plase override "train_on_batch" method in the derived model!
        '''
        pass

    def predict_images(self, z_sample):
        images = self.predict(z_sample)
        if images.shape[3] == 1:
            images = np.squeeze(imgs, axis=(3,))
        # images = np.clip(predictions * 0.5 + 0.5, 0.0, 1.0)
        return images

    @staticmethod
    def get_labels(batchsize, smoothing=0.0, one_sided_smoothing=True):
        if smoothing > 0.0:
            y_pos = 1. - np.random.random((batchsize, )) * smoothing
            if one_sided_smoothing:
                y_neg = np.zeros(batchsize, dtype='float32')
            else:
                y_neg = np.random.random((batchsize, )) * smoothing
        else:
            y_pos = np.ones(batchsize, dtype='float32')
            y_neg = np.zeros(batchsize, dtype='float32')

        return y_pos, y_neg

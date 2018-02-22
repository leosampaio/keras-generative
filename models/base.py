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
from .notifyier import *

# number of training images to store in memory at once
# larger datasets are split into segments of this size
SEGMENT_SIZE = 100 * 1000
GRAD_NORM_LIMIT = 100
CHECKPOINT_ITERS = 20000


def time_format(t):
    m, s = divmod(t, 60)
    m = int(m)
    s = int(s)
    if m == 0:
        return '%d sec' % s
    else:
        return '%d min %d sec' % (m, s)


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

        if 'run_id' not in kwargs:
            run_id = 0
        else:
            run_id = kwargs['run_id']

        self.name = "{}_r{}".format(kwargs['name'], run_id)

        if 'input_shape' not in kwargs:
            raise Exception('Please specify input shape!')

        self.check_input_shape(kwargs['input_shape'])
        self.input_shape = kwargs['input_shape']

        if 'output' not in kwargs:
            self.output = 'output'
        else:
            self.output = kwargs['output']

        self.test_mode = False
        self.trainers = {}

        self.last_epoch = 0

        self.dataset = None

        self.g_losses, self.d_losses, self.losses_ratio = [], [], []

        self.label_smoothing = kwargs.get('label_smoothing', 0.0)
        self.input_noise = kwargs.get('input_noise', 0.0)

    def check_input_shape(self, input_shape):
        # Check for CelebA
        if input_shape == (64, 64, 3):
            return

        # Check for MNIST (size modified)
        if input_shape == (32, 32, 1):
            return

        # Check for Cifar10, 100 etc
        if input_shape == (32, 32, 3):
            return

        errmsg = 'Input size should be 32 x 32 or 64 x 64!'
        raise Exception(errmsg)

    def main_loop(self, datasets, samples, epochs=100, batchsize=100, reporter=[]):
        '''
        Main learning loop
        '''

        global SEGMENT_SIZE

        # Create output directories if not exist
        out_dir = os.path.join(self.output, self.name)
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
        num_data = len(datasets)
        self.dataset = datasets
        self.g_losses, self.d_losses, self.losses_ratio = [], [], []
        for e in range(self.last_epoch, epochs):
            # perm = np.random.permutation(num_data)

            if SEGMENT_SIZE > len(datasets): 
                SEGMENT_SIZE = len(datasets)
            total_segments = len(datasets) // SEGMENT_SIZE
            for segment_idx in range(total_segments):
                start_time = time.time()
                perm = np.random.permutation(SEGMENT_SIZE)
                num_data = SEGMENT_SIZE
                print('\nLoading segment {}'.format(segment_idx))
                dataset = np.asarray(datasets[segment_idx * SEGMENT_SIZE:(segment_idx + 1) * SEGMENT_SIZE], 'float32')
                for b in range(0, num_data, batchsize):
                    bsize = min(batchsize, num_data - b)
                    indx = perm[b:b + bsize]

                    # every 20000 iterations save generated images and compute gradient norms
                    checkpoint = (b + bsize) % CHECKPOINT_ITERS == 0 or (b + bsize) == num_data

                    # Get batch and train on it
                    x_batch = self.make_batch(dataset, indx)
                    x_batch = BaseModel.add_input_noise(x_batch, e, segment_idx, total_segments, self.input_noise)
                    losses = self.train_on_batch(x_batch, compute_grad_norms=checkpoint)

                    # get current status
                    ratio = 100.0 * (b + bsize + segment_idx * SEGMENT_SIZE) / len(datasets)
                    status_string = "Epoch #{:04.0f} | {:06.0f} / {:06.0f} ({:6.2f}% )".format(e + 1, segment_idx * SEGMENT_SIZE + b + bsize, len(datasets), ratio)

                    for k in losses.keys():
                        status_string = "{} | {} = {:8.6f}".format(status_string, k, losses[k])
                        assert not np.isnan(losses[k])

                    # compute ETA
                    elapsed_time = time.time() - start_time
                    eta = elapsed_time / (b + bsize) * (len(datasets) - (b + bsize))
                    status_string = "{} | ETA: {}".format(status_string, time_format(eta))
                    print(status_string)
                    sys.stdout.flush()

                    # add collapse checking
                    if losses["g_loss"] == losses["d_loss"]:
                        send_telegram_message("Generator and Discriminator losses are equal. Possible collapse!")
                        print("Generator and Discriminator losses are equal. Possible collapse!")
                        exit()

                    if b % (5 * batchsize) == 0:
                        self.g_losses.append(losses['g_loss'])
                        self.d_losses.append(losses['d_loss'])
                        self.losses_ratio.append(losses['g_loss'] / losses['d_loss'])
                        self.save_losses_hist(out_dir)
                    # Save generated images
                    if checkpoint:
                        outfile = os.path.join(res_out_dir, 'epoch_%04d_batch_%d.png' % (
                            e + 1, segment_idx * SEGMENT_SIZE + b + bsize))
                        self.save_images(samples, outfile)
                        self.save_losses_hist(out_dir)
                        send_telegram_image(os.path.join(out_dir, 'loss_hist.png'))

                    if self.test_mode:
                        print('\nFinish testing: %s' % self.name)
                        return

                elapsed_time = time.time() - start_time
                print('Took: {}\n'.format(elapsed_time))

            print('')

            # Save current weights
            self.save_model(wgt_out_dir, e + 1)

    def save_losses_hist(self, out_dir):
        plt.plot(self.g_losses, label='Gen')
        plt.plot(self.d_losses, label='Dis')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'loss_hist.png'))
        plt.close()

        plt.plot(self.losses_ratio, label='G / D')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'losses_ratio.png'))
        plt.close()

    def make_batch(self, datasets, indx):
        '''
        Get batch from datasets
        '''
        return datasets[indx]
        batch = []
        for idx in indx:
            batch.append(datasets[idx])
        batch = np.array(batch, 'float32')

        return batch

    def save_images(self, samples, filename):
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
        send_telegram_image(filename)
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
            filename = os.path.join(folder, '%s.hdf5' % (k))
            getattr(self, k).load_weights(filename)

        # load epoch number
        epoch = int(folder.split('_')[-1].replace('/', ''))
        self.last_epoch = epoch

    @abstractmethod
    def predict(self, z_sample):
        '''
        Plase override "predict" method in the derived model!
        '''
        pass

    @abstractmethod
    def train_on_batch(self, x_batch, compute_grad_norms=False):
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
    def get_labels(batchsize, smoothing=0.0):
        if smoothing > 0.0:
            y_pos = 1. - np.random.random((batchsize, )) * smoothing
            y_neg = np.zeros(batchsize, dtype='float32')#np.random.random((batchsize, )) * smoothing
        else:
            y_pos = np.ones(batchsize, dtype='float32')
            y_neg = np.zeros(batchsize, dtype='float32')

        return y_pos, y_neg

    @staticmethod
    def add_input_noise(x_batch, curr_epoch, curr_segment, total_segments, start_noise):
        if start_noise == 0.0:
            return x_batch

        noise = np.random.normal(size=x_batch.shape)

        noise_factor = (curr_segment / total_segments + curr_epoch * total_segments) or 1
        noise_factor = start_noise / noise_factor
        if noise_factor < 0.02:
            return x_batch

        noised_batch = x_batch + noise * noise_factor
        return noised_batch

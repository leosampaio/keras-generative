import os
import random
from abc import ABCMeta, abstractmethod

import numpy as np

import keras
import matplotlib.gridspec as gridspec
from keras.engine.topology import Layer
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU, LocallyConnected2D,
                          Lambda)
from keras.optimizers import Adam, Adadelta
from keras import backend as K

from core.models import BaseModel

from .utils import *
from .layers import *


def simple_discriminator_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p, prediction for pairs (Gx(z), z)
    y_pred[:,1]: q, prediction for pairs (x, Gz(z))
    """
    p = K.clip(y_pred[:, 0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:, 1], K.epsilon(), 1.0 - K.epsilon())
    p_true = y_true[:, 0]
    q_true = y_true[:, 1]

    q_error = -K.mean(K.log(K.abs(q_true - q)))
    p_error = -K.mean(K.log(K.abs(p - p_true)))

    return q_error + p_error


def simple_generator_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p, prediction for pairs (Gx(z), z)
    y_pred[:,1]: q, prediction for pairs (x, Gz(z))
    """
    p = K.clip(y_pred[:, 0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:, 1], K.epsilon(), 1.0 - K.epsilon())
    p_true = y_true[:, 0]
    q_true = y_true[:, 1]

    q_error = -K.mean(K.log(K.abs(p_true - q)))
    p_error = -K.mean(K.log(K.abs(p - q_true)))

    return q_error + p_error


class TemporalCycleGAN(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=128,
                 name='temporal_alice',
                 **kwargs):
        super().__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.f_Gab = None
        self.f_Gba = None
        self.f_D = None

        self.gen_trainer = None
        self.dis_trainer = None

        self.last_d_loss = 10000000

        self.is_conditional = kwargs.get('is_conditional', False)
        self.auxiliary_classifier = kwargs.get('auxiliary_classifier', False)
        self.conditional_dims = kwargs.get('conditional_dims', 0)
        self.conditionals_for_samples = kwargs.get('conditionals_for_samples', False)

        self.last_losses = {
            'g_loss': 10.,
            'd_loss': 10.
        }

        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        batchsize = len(x_data)
        a_data, b_data = x_data, y_batch

        # perform label smoothing if applicable

        y_pos, y_neg = smooth_binary_labels(batchsize, self.label_smoothing, one_sided_smoothing=True)
        y = np.stack((y_neg, y_pos), axis=1)

        input_data = [x_data, b_data]
        label_data = [y, a_data, b_data]

        # train both networks
        _, d_loss, _, _ = self.dis_trainer.train_on_batch(input_data, label_data)
        _, g_loss, ba_cycle_loss, ab_cycle_loss = self.gen_trainer.train_on_batch(input_data, label_data)
        if self.last_losses['d_loss'] < self.dis_loss_control:
            _, g_loss, ba_cycle_loss, ab_cycle_loss = self.gen_trainer.train_on_batch(input_data, label_data)
        if self.last_losses['d_loss'] < self.dis_loss_control * 1e-5:
            for i in range(0, 5):
                _, g_loss, ba_cycle_loss, ab_cycle_loss = self.gen_trainer.train_on_batch(input_data, label_data)

        self.last_d_loss = d_loss
        losses = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'ba_cycle_loss': ba_cycle_loss,
            'ab_cycle_loss': ba_cycle_loss,
        }

        self.last_losses = losses
        return losses

    def predict(self, z_samples):
        return self.f_Gab.predict(z_samples)

    def build_trainer(self):
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)

        a_hat = self.f_Gba(input_b)
        b_hat = self.f_Gab(input_a)

        a_reconstructed = self.f_Gba(b_hat)
        b_reconstructed = self.f_Gab(a_hat)
        p = self.f_D([a_hat, input_b])
        q = self.f_D([input_a, b_hat])
        input = [input_a, input_b]

        concatenated = Concatenate(axis=-1, name="discriminator_outputs")([p, q])
        return Model(input, [concatenated, a_reconstructed, b_reconstructed], name='TCycleGAN')

    def build_model(self):

        self.f_Gab = self.build_Gx()
        self.f_Gba = self.build_Gx()
        self.f_D = self.build_D()
        self.f_Gba.summary()
        self.f_Gab.summary()
        self.f_D.summary()

        opt_d, opt_g = self.build_optmizers()
        loss_d, loss_g = self.define_loss_functions()

        # build discriminator
        self.dis_trainer = self.build_trainer()
        set_trainable(self.f_Gba, False)
        set_trainable(self.f_Gab, False)
        set_trainable(self.f_D, True)
        self.dis_trainer.compile(optimizer=opt_d,
                                 loss=[loss_d, 'mae', 'mae'],
                                 loss_weights=[1., 0., 0.])

        # build generators
        self.gen_trainer = self.build_trainer()
        set_trainable(self.f_Gba, True)
        set_trainable(self.f_Gab, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=opt_g,
                                 loss=[loss_g, 'mae', 'mae'],
                                 loss_weights=[1., 1., 1.])

        self.dis_trainer.summary()
        self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_trainer')

    def define_loss_functions(self):
        return simple_discriminator_lossfun, simple_generator_lossfun

    def make_batch(self, dataset, idx):
        a_data, b_data = dataset.get_pairs(idx)
        return a_data, b_data

    def save_model(self, out_dir, epoch):
        self.trainers['f_D'] = self.f_D
        self.trainers['f_Gba'] = self.f_Gba
        self.trainers['f_Gab'] = self.f_Gab
        super().save_model(out_dir, epoch)

    def save_images(self, samples, filename, conditionals_for_samples=None):
        '''
        Save images generated from random sample numbers
        '''
        samples_a, samples_b = self.dataset.get_some_random_samples()
        samples_ab = self.f_Gab.predict(samples_a)
        samples_ba = self.f_Gba.predict(samples_b)
        
        samples_a = self.dataset.get_original_frames_from_processed_samples(samples_a)
        samples_ab = self.dataset.get_original_frames_from_processed_samples(samples_ab)
        samples_ba = self.dataset.get_original_frames_from_processed_samples(samples_ba)
        samples_b = self.dataset.get_original_frames_from_processed_samples(samples_b)

        n = self.dataset.input_n_frames
        grid = [[np.zeros(samples_a.shape[2:])] * n * 2] * 4 * 2
        grid = np.array(grid)
        grid[0::2, 0:n] = samples_a
        grid[0::2, n:n * 2] = samples_ab
        grid[1::2, 0:n] = samples_ba
        grid[1::2, n:n * 2] = samples_b
        self.save_image_as_plot(grid, filename)

    def save_image_as_plot(self, imgs, filename):
        fig = plt.figure(figsize=(8, 8))
        grid = gridspec.GridSpec(8, self.dataset.input_n_frames * 2, wspace=0.1, hspace=0.1)
        for i, imgs_row in enumerate(imgs):
            for j, img in enumerate(imgs_row):
                ax = plt.Subplot(fig, grid[i, j])
                if img.shape[2] == 1:
                        img = np.squeeze(img, axis=(2,))
                if img.ndim == 3:
                    ax.imshow(img, interpolation='none', vmin=0.0, vmax=1.0)
                else:
                    ax.imshow(img, cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
                ax.axis('off')
                fig.add_subplot(ax)
        fig.savefig(filename, dpi=200)
        plt.close(fig)

    def did_collapse(self, losses):
        if losses["g_loss"] == losses["d_loss"]:
            return "G and D losses are equal"
        else:
            return False

    def build_Gx(self):
        a_input = Input(shape=self.input_shape)
        orig_channels = self.input_shape[2]

        x = BasicConvLayer(64, (7, 7), strides=(1, 1), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(a_input)
        x = BasicConvLayer(128, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicConvLayer(256, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = ResLayer(256, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = ResLayer(256, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = ResLayer(256, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(128, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)

        x = BasicConvLayer(32, (7, 7), strides=(1, 1), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(orig_channels, (1, 1), activation='sigmoid', bnorm=False)(x)

        return Model(a_input, x)

    def build_D(self):
        a_input = Input(shape=self.input_shape)

        a = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(a_input)
        res_a = a = BasicConvLayer(64, (3, 3), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(a)
        a = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01, residual=res_a)(a)
        res_a = a = BasicConvLayer(128, (3, 3), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(a)
        a = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01, residual=res_a)(a)
        a = Flatten()(a)

        b_input = Input(shape=self.input_shape)

        b = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(b_input)
        res_b = b = BasicConvLayer(64, (3, 3), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(b)
        b = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01, residual=res_b)(b)
        res_b = b = BasicConvLayer(128, (3, 3), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(b)
        b = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01, residual=res_b)(b)
        b = Flatten()(b)

        ab = Concatenate(axis=-1)([a, b])

        ab = Dense(512)(ab)
        ab = LeakyReLU(0.01)(ab)
        ab = Dropout(0.2)(ab)

        ab = Dense(512)(ab)
        ab = LeakyReLU(0.01)(ab)
        ab = Dropout(0.2)(ab)

        ab = Dense(1)(ab)
        ab = Activation('sigmoid')(ab)

        return Model([a_input, b_input], ab)

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr)
        opt_g = Adam(lr=self.lr)
        return opt_d, opt_g

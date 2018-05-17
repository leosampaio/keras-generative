import os
import random
from abc import ABCMeta, abstractmethod

import numpy as np

import keras
from keras.engine.topology import Layer
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU, LocallyConnected2D,
                          Lambda)
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
import tensorflow as tf

from .base import BaseModel
import models

from .utils import *
from .layers import *
from .alice import generator_lossfun, discriminator_lossfun, simple_generator_lossfun, simple_discriminator_lossfun
from .triplet_alice_lcc_ds import TripletALICEwithLCCandDS, triplet_lossfun_creator

def dmae_pi_loss(y_true, y_pred):
    """
        min_PI ||(K.PI)^t - (L.PI^t)||2 
    """

    K_PI, L_PI_t = y_pred[..., 0], y_pred[..., 1]
    return K.sum(K.square(K.transpose(K_PI) - L_PI_t))

def dmae_theta_loss(y_true, y_pred):
    """
        min_PI trace(K.PI.L.PI^t)
    """

    K_PI, L_PI_t = y_pred[..., 0], y_pred[..., 1]
    return -tf.trace(K.dot(K_PI, L_PI_t))

class DMAEwithExplicitALICE(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 submodels=['ealice_shared', 'ealice_shared'],
                 permutation_matrix_shape=None,
                 *args,
                 **kwargs):
        kwargs['name'] = 'dmae_ealice'
        super().__init__(*args, **kwargs)

        assert permutation_matrix_shape is not None
        self.permutation_matrix_shape = permutation_matrix_shape

        self.alice_d1 = models.models[submodels[0]](*args, **kwargs)
        self.alice_d2 = models.models[submodels[1]](*args, **kwargs)

        # create local references to ease model saving and loading
        self.d1_f_D = self.alice_d1.f_D
        self.d1_f_Gz = self.alice_d1.f_Gz
        self.d1_f_Gx = self.alice_d1.f_Gx

        self.d2_f_D = self.alice_d2.f_D
        self.d2_f_Gz = self.alice_d2.f_Gz
        self.d2_f_Gx = self.alice_d2.f_Gx

        self.z_dims = kwargs.get('z_dims', 128)
        self.submodels_weights = kwargs.get('submodels_weights', None)

        self.last_losses = {
            'g_loss': 10.,
            'd_loss': 10.,
            'domain1_g_loss': 10.,
            'domain1_d_loss': 10.,
            'domain2_g_loss': 10.,
            'domain2_d_loss': 10.,
            'dmae_theta_loss': 10.,
            'dmae_pi_loss': 10.,
        }

        self.dmae_losses = []

        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        a_x, b_x = x_data
        a_y, b_y = y_batch

        batchsize = len(a_x)

        # perform label smoothing if applicable
        y_pos, y_neg = smooth_binary_labels(batchsize, self.label_smoothing, one_sided_smoothing=True)
        y = np.stack((y_neg, y_pos), axis=1)

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        input_data = [a_x, b_x, a_y, b_y, z_latent_dis]
        input_labels = [y, a_x, y, b_x, y]

        # train both networks
        d_loss = self.dis_trainer.train_on_batch(input_data, input_labels)
        g_loss = self.gen_trainer.train_on_batch(input_data, input_labels)
        dmae_loss = self.dmae_trainer.train_on_batch([a_x, b_x, a_y, b_y], y)
        if self.last_losses['domain1_d_loss'] < self.dis_loss_control or self.last_losses['domain2_d_loss'] < self.dis_loss_control:
            g_loss = self.gen_trainer.train_on_batch(input_data, input_labels)

        self.last_losses = {
            'g_loss': g_loss[1] + g_loss[3],
            'd_loss': d_loss[0],
            'domain1_g_loss': g_loss[1],
            'domain1_d_loss': d_loss[1],
            'domain1_c_loss': g_loss[2],
            'domain2_g_loss': g_loss[3],
            'domain2_d_loss': d_loss[3],
            'domain2_c_loss': g_loss[4],
            'dmae_theta_loss': g_loss[5],
            'dmae_pi_loss': dmae_loss
        }

        return self.last_losses

    def build_trainer(self):

        input_a_x = Input(shape=self.input_shape)
        input_b_x = Input(shape=self.input_shape)
        input_a_i = Input(shape=(1,), dtype='int64')
        input_b_i = Input(shape=(1,), dtype='int64')
        input_z = Input(shape=(self.z_dims, ))

        d1_z_hat = self.alice_d1.f_Gz(input_a_x)
        d2_z_hat = self.alice_d2.f_Gz(input_b_x)

        # build ALICE for Domain 1 (anchor)
        d1_x_hat = self.alice_d1.f_Gx(input_z)
        d1_x_reconstructed = Activation('linear', name='d1_cycled')(self.alice_d1.f_Gx(d1_z_hat))
        d1_p = self.alice_d1.f_D([d1_x_hat, input_z])
        d1_q = self.alice_d1.f_D([input_a_x, d1_z_hat])

        # build ALICE for Domain 2 (using the positive samples)
        d2_x_hat = self.alice_d2.f_Gx(input_z)
        d2_x_reconstructed = Activation('linear', name='d2_cycled')(self.alice_d2.f_Gx(d2_z_hat))
        d2_p = self.alice_d2.f_D([d2_x_hat, input_z])
        d2_q = self.alice_d2.f_D([input_b_x, d2_z_hat])

        concatenated_d1 = Concatenate(axis=-1, name="d1_discriminator")([d1_p, d1_q])
        concatenated_d2 = Concatenate(axis=-1, name="d2_discriminator")([d2_p, d2_q])

        Kgram = GramMatrixLayer()(d1_z_hat)
        Lgram = GramMatrixLayer()(d2_z_hat)
        K_PI, L_PI_t = self.permutation_matrix([Kgram, Lgram, input_a_i, input_b_i])
        exp_dim_layer = Lambda(lambda x: K.expand_dims(x))
        concatenated_dmae_matrices = Concatenate(axis=-1, name="dmae_matrices")([exp_dim_layer(K_PI), exp_dim_layer(L_PI_t)])

        input_data = [input_a_x, input_b_x, input_a_i, input_b_i, input_z]
        output_data = [concatenated_d1, d1_x_reconstructed, concatenated_d2, d2_x_reconstructed, concatenated_dmae_matrices]

        return Model(
            input_data,
            output_data,
            name='dmae'
        )

    def build_dmae_trainer(self):

        input_a_x = Input(shape=self.input_shape)
        input_b_x = Input(shape=self.input_shape)
        input_a_i = Input(shape=(1,), dtype='int64')
        input_b_i = Input(shape=(1,), dtype='int64')
        n, m = self.permutation_matrix_shape

        d1_z_hat = self.alice_d1.f_Gz(input_a_x)
        d2_z_hat = self.alice_d2.f_Gz(input_b_x)

        # DMAE permutation matrix PI
        self.permutation_matrix = PermutationMatrixPiLayer(n, m, restriction_weight=1.0)
        Kgram = GramMatrixLayer()(d1_z_hat)
        Lgram = GramMatrixLayer()(d2_z_hat)
        K_PI, L_PI_t = self.permutation_matrix([Kgram, Lgram, input_a_i, input_b_i])
        exp_dim_layer = Lambda(lambda x: K.expand_dims(x))
        concatenated_dmae_matrices = Concatenate(axis=-1, name="dmae_matrices")([exp_dim_layer(K_PI), exp_dim_layer(L_PI_t)])

        return Model(
            [input_a_x, input_b_x, input_a_i, input_b_i],
            concatenated_dmae_matrices,
            name='dmae'
        )

    def build_model(self):

        # get loss functions and optmizers
        loss_d, loss_g = self.define_loss_functions()
        opt_d, opt_g = self.build_optmizers()

        # build the DMAE trainer (PI matrix optmizer)
        self.dmae_trainer = self.build_dmae_trainer()
        self.permutation_matrix.trainable = True
        set_trainable([self.alice_d1.f_Gx, self.alice_d1.f_Gz,
                       self.alice_d2.f_Gx, self.alice_d2.f_Gz,
                       self.alice_d1.f_D, self.alice_d2.f_D], False)
        self.dmae_trainer.compile(optimizer=opt_g,
                                  loss=dmae_pi_loss)

        # build the discriminators trainer
        self.dis_trainer = self.build_trainer()
        self.permutation_matrix.trainable = False
        set_trainable(
            [self.alice_d1.f_Gx, self.alice_d1.f_Gz,
             self.alice_d2.f_Gx, self.alice_d2.f_Gz], False)
        set_trainable([self.alice_d1.f_D, self.alice_d2.f_D], True)
        self.dis_trainer.compile(optimizer=opt_d,
                                 loss={
                                     "d1_discriminator": loss_d,
                                     "d1_cycled": "mae",
                                     "d2_discriminator": loss_d,
                                     "d2_cycled": "mae",
                                     "dmae_matrices": dmae_theta_loss
                                 },
                                 loss_weights=[1., 0., 1., 0., 0.])

        # build the generators trainer
        self.gen_trainer = self.build_trainer()
        self.permutation_matrix.trainable = True
        set_trainable(
            [self.alice_d1.f_Gx, self.alice_d1.f_Gz,
             self.alice_d2.f_Gx, self.alice_d2.f_Gz], True)
        set_trainable([self.alice_d1.f_D, self.alice_d2.f_D], False)
        self.gen_trainer.compile(optimizer=opt_g,
                                 loss={
                                     "d1_discriminator": loss_g,
                                     "d1_cycled": "mae",
                                     "d2_discriminator": loss_g,
                                     "d2_cycled": "mae",
                                     "dmae_matrices": dmae_theta_loss
                                 },
                                 loss_weights=[1., 1., 1., 1., 1.])

        self.dis_trainer.summary()
        self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_trainer')
        self.store_to_save('d1_f_D')
        self.store_to_save('d1_f_Gz')
        self.store_to_save('d1_f_Gx')
        self.store_to_save('d2_f_D')
        self.store_to_save('d2_f_Gz')
        self.store_to_save('d2_f_Gx')

    def define_loss_functions(self):
        return simple_discriminator_lossfun, simple_generator_lossfun

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr, clipnorm=5.)
        opt_g = Adam(lr=self.lr, clipnorm=5.)
        return opt_d, opt_g

    def make_batch(self, dataset, indx):
        data, labels = dataset.get_unlalabeled_pairs(indx)
        return data, labels

    def plot_losses_hist(self, out_dir):
        plt.plot(self.g_losses, label='Gen')
        plt.plot(self.d_losses, label='Dis')
        plt.plot(self.dmae_losses[0], label='DMAE_theta')
        plt.plot(self.dmae_losses[1], label='DMAE_pi')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'loss_hist.png'))
        plt.close()

    def save_losses_history(self, losses):
        self.g_losses.append(losses['g_loss'])
        self.d_losses.append(losses['d_loss'])
        self.dmae_losses.append((losses["dmae_theta_loss"], losses["dmae_pi_loss"]))
        self.losses_ratio.append(losses['g_loss'] / losses['d_loss'])

    predict = TripletALICEwithLCCandDS.__dict__['predict']
    save_images = TripletALICEwithLCCandDS.__dict__['save_images']
    predict_images = TripletALICEwithLCCandDS.__dict__['predict_images']
    did_collapse = TripletALICEwithLCCandDS.__dict__['did_collapse']
    load_model = TripletALICEwithLCCandDS.__dict__['load_model']

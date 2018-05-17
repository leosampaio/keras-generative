import os
import random
from abc import ABCMeta, abstractmethod

import numpy as np

import keras
from keras.engine.topology import Layer
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU, LocallyConnected2D,
                          Lambda, Add)
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K

from .base import BaseModel
import models

from .utils import *
from .layers import *
from .alice import generator_lossfun, discriminator_lossfun, simple_generator_lossfun, simple_discriminator_lossfun
from .triplet_alice_lcc_ds import TripletALICEwithLCCandDS, triplet_lossfun_creator


def latent_cycle_mae_loss(y_true, y_pred):

    a, b = y_pred[..., :y_pred.shape[-1] // 2], y_pred[..., (y_pred.shape[-1] // 2):]
    return K.mean(K.abs(a - b), axis=-1)

def gram_matrix(x):
    s = K.shape(x)
    x = K.batch_flatten(x)
    x = K.reshape(x, (-1, s[1]*s[2], s[3]))
    return K.batch_dot(x, K.permute_dimensions(x, (0, 2, 1)))

def frobenius_norm(x):
    a, b = x[0], x[1]
    a = gram_matrix(a)
    b = gram_matrix(b)
    s = K.cast(K.shape(a), dtype='float32')
    M = s[1] * s[2] * 3
    return K.sum(K.square(a - b)) / (4 * K.square(M))

def min_loss(y_true, y_pred):
    return K.sum(y_pred)


class TripletExplicitALICEwithExplicitLCCandDSandStyleLoss(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 submodels=['ealice_shared', 'ealice_shared'],
                 *args,
                 **kwargs):
        kwargs['name'] = 'triplet_ealice_elcc_ds_stylel'
        super().__init__(*args, **kwargs)

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
        self.is_conditional = kwargs.get('is_conditional', False)
        self.auxiliary_classifier = kwargs.get('auxiliary_classifier', False)
        self.conditional_dims = kwargs.get('conditional_dims', 0)
        self.conditionals_for_samples = kwargs.get('conditionals_for_samples', False)
        self.triplet_margin = kwargs.get('triplet_margin', 1.0)
        self.triplet_weight = kwargs.get('triplet_weight', 1.0)
        self.submodels_weights = kwargs.get('submodels_weights', None)

        self.triplet_losses = []

        self.last_losses = {
            'g_loss': 10.,
            'd_loss': 10.,
            'domain1_g_loss': 10.,
            'domain1_d_loss': 10.,
            'domain2_g_loss': 10.,
            'domain2_d_loss': 10.,
            'lc12_loss': 10.,
            'lc21_loss': 10.,
            'triplet_loss': 10.,
        }

        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        a_x, p_x, n_x = x_data
        a_y, p_y, n_y = y_batch

        batchsize = len(a_x)

        # perform label smoothing if applicable
        y_pos, y_neg = smooth_binary_labels(batchsize, self.label_smoothing, one_sided_smoothing=True)
        y = np.stack((y_neg, y_pos), axis=1)
        y_dumb = y.reshape((-1, 2, 1))

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        input_data = [a_x, p_x, n_x, z_latent_dis]
        input_labels = [y, a_x, y, p_x, y, y, y_dumb, y_dumb, y]

        # train both networks
        d_loss = self.dis_trainer.train_on_batch(input_data, input_labels)
        g_loss = self.gen_trainer.train_on_batch(input_data, input_labels)
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
            'lc12_loss': g_loss[5],
            'lc21_loss': g_loss[6],
            'd1_cstyle_loss': g_loss[7],
            'd2_cstyle_loss': g_loss[8],
            'triplet_loss': g_loss[9],
        }

        return self.last_losses

    def build_trainer(self):

        input_a_x = Input(shape=self.input_shape)
        input_p_x = Input(shape=self.input_shape)
        input_n_x = Input(shape=self.input_shape)
        input_z = Input(shape=(self.z_dims, ))

        d1_z_hat = self.alice_d1.f_Gz(input_a_x)
        d2_z_hat = self.alice_d2.f_Gz(input_p_x)

        # build ALICE for Domain 1 (anchor)
        d1_x_hat = self.alice_d1.f_Gx(input_z)
        d1_x_reconstructed = Activation('linear', name='d1_cycled')(self.alice_d1.f_Gx(d1_z_hat))
        d1_p, *_ = self.alice_d1.f_D([d1_x_hat, input_z])
        d1_q, *d1_original_grams = self.alice_d1.f_D([input_a_x, d1_z_hat])

        # build ALICE for Domain 2 (using the positive samples)
        d2_x_hat = self.alice_d2.f_Gx(input_z)
        d2_x_reconstructed = Activation('linear', name='d2_cycled')(self.alice_d2.f_Gx(d2_z_hat))
        d2_p, *_ = self.alice_d2.f_D([d2_x_hat, input_z])
        d2_q, *d2_original_grams = self.alice_d2.f_D([input_p_x, d2_z_hat])

        # produce cross cycled x samples
        reconstructed_x_12 = self.alice_d2.f_Gx(d1_z_hat)
        reconstructed_x_21 = self.alice_d1.f_Gx(d2_z_hat)

        # style cross-domain consistency (or some other cool name)
        _, *cycled_12_grams = self.alice_d1.f_D([reconstructed_x_12, input_z])
        _, *cycled_21_grams = self.alice_d1.f_D([reconstructed_x_21, input_z])
        frobenius_norm_layer = Lambda(frobenius_norm, output_shape=(None, 1))
        norm_btw_grams_d1 = [frobenius_norm_layer([a, b]) for a, b in zip(d1_original_grams, cycled_21_grams)]
        norm_btw_grams_d2 = [frobenius_norm_layer([a, b]) for a, b in zip(d2_original_grams, cycled_12_grams)]
        d1_cross_style_loss = Add(name="d1_cross_style_loss")(norm_btw_grams_d1)
        d2_cross_style_loss = Add(name="d2_cross_style_loss")(norm_btw_grams_d2)

        input = [input_a_x, input_p_x, input_n_x, input_z]

        # get reconstructed latent variables for latent cycle consistency
        slice_g_lambda = Lambda(lambda x: x[:, :self.z_dims // 2], output_shape=(self.z_dims // 2, ))
        latent_cycle_12 = slice_g_lambda(self.alice_d2.f_Gz(reconstructed_x_12))
        latent_cycle_21 = slice_g_lambda(self.alice_d1.f_Gz(reconstructed_x_21))
        sliced_d1_z_hat = slice_g_lambda(d1_z_hat)
        sliced_d2_z_hat = slice_g_lambda(d2_z_hat)

        # get only encoding for Domain 2 negative samples
        d2_z_n_hat = self.alice_d2.f_Gz(input_n_x)

        concatenated_d1 = Concatenate(axis=-1, name="d1_discriminator")([d1_p, d1_q])
        concatenated_d2 = Concatenate(axis=-1, name="d2_discriminator")([d2_p, d2_q])
        concatenated_lat_cycle_12 = Concatenate(axis=-1, name="lat_cycle_12")([latent_cycle_12, sliced_d1_z_hat])
        concatenated_lat_cycle_21 = Concatenate(axis=-1, name="lat_cycle_21")([latent_cycle_21, sliced_d2_z_hat])
        concatenated_triplet_enc = Concatenate(axis=-1, name="triplet_encoding")([d1_z_hat, d2_z_hat, d2_z_n_hat])
        return Model(
            input,
            [concatenated_d1, d1_x_reconstructed, concatenated_d2,
             d2_x_reconstructed, concatenated_lat_cycle_12,
             concatenated_lat_cycle_21, d1_cross_style_loss,
             d2_cross_style_loss, concatenated_triplet_enc],
            name='triplet_ali'
        )

    def build_model(self):

        # change original discriminators for versions that output gram matrices
        self.d1_f_D = self.alice_d1.f_D = self.build_style_D()
        self.d2_f_D = self.alice_d2.f_D = self.build_style_D()

        # get loss functions and optmizers
        loss_d, loss_g, loss_triplet = self.define_loss_functions()
        opt_d, opt_g = self.build_optmizers()

        # build the discriminators trainer
        self.dis_trainer = self.build_trainer()
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
                                     "lat_cycle_12": latent_cycle_mae_loss,
                                     "lat_cycle_21": latent_cycle_mae_loss,
                                     "d1_cross_style_loss": min_loss,
                                     "d2_cross_style_loss": min_loss,
                                     "triplet_encoding": loss_triplet
                                 },
                                 loss_weights=[1., 0., 1., 0., 0., 0., 0., 0., 0.])

        # build the generators trainer
        self.gen_trainer = self.build_trainer()
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
                                     "lat_cycle_12": latent_cycle_mae_loss,
                                     "lat_cycle_21": latent_cycle_mae_loss,
                                     "d1_cross_style_loss": min_loss,
                                     "d2_cross_style_loss": min_loss,
                                     "triplet_encoding": loss_triplet
                                 },
                                 loss_weights=[1., 1., 1., 1., 1., 1., 1., 1., self.triplet_weight])

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
        return simple_discriminator_lossfun, simple_generator_lossfun, triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.z_dims)

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr, clipnorm=5.)
        opt_g = Adam(lr=self.lr, clipnorm=5.)
        return opt_d, opt_g

    def build_style_D(self):
        x_input = Input(shape=self.input_shape)

        l1 = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_input)
        l2 = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(l1)
        l3 = ResLayer(64, (4, 4), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(l2)
        l4 = ResLayer(64, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(l3)
        l5 = ResLayer(64, (4, 4), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(l4)
        l6 = BasicConvLayer(64, (4, 4), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(l5)
        x = Flatten()(l6)

        z_input = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_input)
        z = BasicConvLayer(128, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
        z = ResLayer(128, (1, 1), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
        z = ResLayer(128, (1, 1), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
        z = BasicConvLayer(128, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
        z = Flatten()(z)

        xz = Concatenate(axis=-1)([x, z])

        xz = Dense(1024)(xz)
        xz = LeakyReLU(0.01)(xz)
        xz = Dropout(0.2)(xz)

        xz = Dense(1024)(xz)
        xz = LeakyReLU(0.01)(xz)
        xz = Dropout(0.2)(xz)

        xz = Dense(1)(xz)
        xz = Activation('sigmoid')(xz)

        # gram matrices from all x layers
        g1 = GramMatrixLayer()(l1)
        g2 = GramMatrixLayer()(l2)
        g3 = GramMatrixLayer()(l3)
        g4 = GramMatrixLayer()(l4)
        g5 = GramMatrixLayer()(l5)
        # g6 = GramMatrixLayer()(l6)

        return Model([x_input, z_input], [xz, l1, l2, l3, l4, l5])

    predict = TripletALICEwithLCCandDS.__dict__['predict']
    make_batch = TripletALICEwithLCCandDS.__dict__['make_batch']
    save_images = TripletALICEwithLCCandDS.__dict__['save_images']
    predict_images = TripletALICEwithLCCandDS.__dict__['predict_images']
    did_collapse = TripletALICEwithLCCandDS.__dict__['did_collapse']
    plot_losses_hist = TripletALICEwithLCCandDS.__dict__['plot_losses_hist']
    save_losses_history = TripletALICEwithLCCandDS.__dict__['save_losses_history']
    load_model = TripletALICEwithLCCandDS.__dict__['load_model']

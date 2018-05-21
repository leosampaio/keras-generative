import os
import random
from abc import ABCMeta, abstractmethod
from pprint import pprint

import numpy as np
from sklearn.svm import LinearSVC

import keras
from keras.engine.topology import Layer
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU,
                          LocallyConnected2D, Add,
                          Lambda, AveragePooling1D, GlobalAveragePooling2D)
from keras.optimizers import Adam, Adadelta
from keras import initializers
from keras import backend as K
from keras.applications.mobilenet import MobileNet

from .base import BaseModel

from .utils import *
from .layers import *


def triplet_lossfun_creator(margin=1., zdims=256, inverted=False):
    def triplet_lossfun(_, y_pred):

        m = K.constant(margin)
        zero = K.constant(0.)
        a, p, n = [y_pred[..., i:i + zdims] for i in range(0, y_pred.shape[-1], zdims)]
        if inverted:
            return K.maximum(zero, m + K.sqrt(K.sum(K.square(a - n))))
        else:
            return K.maximum(zero, m + K.sqrt(K.sum(K.square(a - p))) - K.sqrt(K.sum(K.square(a - n))))

    return triplet_lossfun


def discriminator_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p, prediction for pairs (Gx(z), z)
    y_pred[:,1]: q, prediction for pairs (x, Gz(z))
    y_pred[:,2]: p_cycle, prediction for pairs (x, x)
    y_pred[:,3]: q_cycle, prediction for pairs (x, Gx(x))
    """
    p = K.clip(y_pred[:, 0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:, 1], K.epsilon(), 1.0 - K.epsilon())
    p_true = y_true[:, 0]
    q_true = y_true[:, 1]

    q_error = -K.mean(K.log(K.abs(q_true - q)))
    p_error = -K.mean(K.log(K.abs(p - p_true)))

    return q_error + p_error


def generator_lossfun(y_true, y_pred):
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


class TOPGAN(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=128,
                 name='topgan',
                 **kwargs):
        super().__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.f_Gx = None
        self.f_D = None

        self.gen_trainer = None
        self.dis_trainer = None

        self.is_conditional = kwargs.get('is_conditional', False)
        self.auxiliary_classifier = kwargs.get('auxiliary_classifier', False)
        self.conditional_dims = kwargs.get('conditional_dims', 0)
        self.conditionals_for_samples = kwargs.get('conditionals_for_samples', False)

        self.triplet_margin = kwargs.get('triplet_margin', 1.0)
        self.triplet_weight = kwargs.get('triplet_weight', 1.0)

        self.last_losses = {
            'g_loss': 10.,
            'd_loss': 10.
        }

        self.svm = LinearSVC()
        self.svgan_type = 'epoch_svm'  # or 'batch_svm'
        self.did_train_svm_for_the_first_time = False

        self.embedding_size = 1024

        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        batchsize = len(x_data)

        # perform label smoothing if applicable
        y_pos, y_neg = smooth_binary_labels(batchsize, self.label_smoothing, one_sided_smoothing=False)
        y = np.stack((y_neg, y_pos), axis=1)

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        x_permutation = np.array(np.random.permutation(batchsize), dtype='int64')
        input_data = [x_data, x_permutation, z_latent_dis]
        label_data = [y, y]

        # train both networks
        _, d_loss, d_triplet = self.dis_trainer.train_on_batch(input_data, label_data)
        _, g_loss, g_triplet = self.gen_trainer.train_on_batch(input_data, label_data)
        if self.last_losses['d_loss'] < self.dis_loss_control:
            _, g_loss, g_triplet = self.gen_trainer.train_on_batch(input_data, label_data)
        if self.last_losses['d_loss'] < self.dis_loss_control * 1e-5:
            for i in range(0, 5):
                _, g_loss, g_triplet = self.gen_trainer.train_on_batch(input_data, label_data)

        losses = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'd_triplet': d_triplet,
            'g_triplet': g_triplet
        }

        self.last_losses = losses
        return losses

    def predict(self, z_samples):
        return self.f_Gx.predict(z_samples)

    def build_trainer(self):
        input_x = Input(shape=self.input_shape)
        input_x_perm = Input(shape=(1,), dtype='int64')
        input_z = Input(shape=(self.z_dims,))

        assert self.f_D is not None

        x_hat = self.f_Gx(input_z)
        negative_embedding, p = self.f_D(x_hat)
        anchor_embedding, q = self.f_D(input_x)
        positive_embedding = Lambda(lambda x: K.squeeze(K.gather(anchor_embedding, input_x_perm), 1))(anchor_embedding)

        input = [input_x, input_x_perm, input_z]

        concatenated_dis = Concatenate(axis=-1, name="dis_classification")([p, q])
        concatenated_triplet = Concatenate(axis=-1, name="triplet")([anchor_embedding, positive_embedding, negative_embedding])
        output = [concatenated_dis, concatenated_triplet]
        return Model(input, output, name='svgan')

    def build_model(self):

        self.f_Gx = self.build_Gx()  # Moriarty, the encoder
        self.f_D = self.build_D()   # Sherlock, the detective
        self.f_Gx.summary()
        self.f_D.summary()

        opt_d, opt_g = self.build_optmizers()
        loss_d, loss_g, triplet_d_loss, triplet_g_loss = self.define_loss_functions()

        # build discriminator
        self.dis_trainer = self.build_trainer()
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, True)
        self.dis_trainer.compile(optimizer=opt_d,
                                 loss=[loss_d, triplet_d_loss],
                                 loss_weights=[1., 1.])

        # build generators
        self.gen_trainer = self.build_trainer()
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=opt_g,
                                 loss=[loss_g, triplet_g_loss],
                                 loss_weights=[1., self.triplet_weight])

        self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('gen_trainer')

    def save_model(self, out_dir, epoch):
        self.trainers['f_Gx'] = self.f_Gx
        self.trainers['f_D'] = self.f_D
        super().save_model(out_dir, epoch)

    def define_loss_functions(self):
        return (discriminator_lossfun, generator_lossfun,
                triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size),
                triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size, inverted=True))

    def save_images(self, samples, filename, conditionals_for_samples=None):
        '''
        Save images generated from random sample numbers
        '''
        imgs = self.predict(samples)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        self.save_image_as_plot(imgs, filename)

    def did_collapse(self, losses):
        return False

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))

        x = Dense(256, input_dim=self.z_dims, kernel_initializer=initializers.RandomNormal(stddev=0.02))(z_input)
        x = LeakyReLU(0.2)(x)
        x = Dense(512)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(1024)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(self.input_shape[0] * self.input_shape[1] * self.input_shape[2], activation='sigmoid')(x)
        x = Reshape((self.input_shape[0], self.input_shape[1], self.input_shape[2]))(x)

        return Model(z_input, x)

    def build_D(self):
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.01)(x_input)
        x = BasicConvLayer(64, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        # x = ResLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        # x = ResLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        # x = ResLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(256, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x_embedding = Flatten()(x)

        x_embedding = Dense(self.embedding_size)(x_embedding)
        x_embedding = LeakyReLU(0.01)(x_embedding)
        x_embedding = Dropout(0.2)(x_embedding)

        fc_x = Dense(1024)(x_embedding)
        fc_x = LeakyReLU(0.01)(fc_x)
        fc_x = Dropout(0.2)(fc_x)

        fc_x = Dense(1)(fc_x)
        fc_x = Activation('sigmoid')(fc_x)

        return Model(x_input, [x_embedding, fc_x])

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr)
        opt_g = Adam(lr=self.lr)
        return opt_d, opt_g


class TOPGANbasedonInfoGAN(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=128,
                 name='topgan_binfogan',
                 **kwargs):
        super().__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.f_Gx = None
        self.f_D = None

        self.gen_trainer = None
        self.dis_trainer = None

        self.is_conditional = kwargs.get('is_conditional', False)
        self.auxiliary_classifier = kwargs.get('auxiliary_classifier', False)
        self.conditional_dims = kwargs.get('conditional_dims', 0)
        self.conditionals_for_samples = kwargs.get('conditionals_for_samples', False)

        self.triplet_margin = kwargs.get('triplet_margin', 1.0)
        self.triplet_weight = kwargs.get('triplet_weight', 1.0)

        self.last_losses = {
            'g_loss': 10.,
            'd_loss': 10.
        }

        self.svm = LinearSVC()
        self.svgan_type = 'epoch_svm'  # or 'batch_svm'
        self.did_train_svm_for_the_first_time = False

        self.embedding_size = 1024

        self.build_model()

    train_on_batch = TOPGAN.__dict__['train_on_batch']
    predict = TOPGAN.__dict__['predict']
    build_trainer = TOPGAN.__dict__['build_trainer']
    build_model = TOPGAN.__dict__['build_model']
    define_loss_functions = TOPGAN.__dict__['define_loss_functions']
    save_images = TOPGAN.__dict__['save_images']
    did_collapse = TOPGAN.__dict__['did_collapse']
    build_optmizers = TOPGAN.__dict__['build_optmizers']

    def save_model(self, out_dir, epoch):
        self.trainers['f_Gx'] = self.f_Gx
        self.trainers['f_D'] = self.f_D
        super().save_model(out_dir, epoch)

    def build_Gx(self):
        """
        Network Architecture based on the one presented in infoGAN
        """
        z_input = Input(shape=(self.z_dims,))

        x = Dense(1024, input_dim=self.z_dims)(z_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(128 * (self.input_shape[0] // 4) * (self.input_shape[1] // 4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((self.input_shape[0] // 4, self.input_shape[1] // 4, 128))(x)

        x = BasicDeconvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = BasicDeconvLayer(self.input_shape[2], (4, 4), strides=(2, 2), bnorm=False, padding='same', activation='sigmoid')(x)

        return Model(z_input, x)

    def build_D(self):
        """
        Network Architecture based on the one presented in infoGAN
        """
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.2)(x_input)
        x = BasicConvLayer(128, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2)(x)
        x_embedding = Flatten()(x)

        x_embedding = Dense(self.embedding_size)(x_embedding)
        x_embedding = BatchNormalization()(x_embedding)
        x_embedding = LeakyReLU(0.2)(x_embedding)

        fc_x = Dense(1)(x_embedding)
        fc_x = Activation('sigmoid')(fc_x)

        return Model(x_input, [x_embedding, fc_x])


class TOPGANwithAE(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=128,
                 name='topgan_ae_ebgan',
                 aux_clas_weight=1.,
                 ae_weight=1.,
                 embedding_dim=256,
                 g_triplet_weight=0.,
                 d_triplet_weight=1.,
                 isolate_d_classifier=False,
                 **kwargs):
        super().__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.f_Gx = None
        self.f_D = None

        self.gen_trainer = None
        self.dis_trainer = None

        self.is_conditional = kwargs.get('is_conditional', False)
        self.auxiliary_classifier = kwargs.get('auxiliary_classifier', False)
        self.conditional_dims = kwargs.get('conditional_dims', 0)
        self.conditionals_for_samples = kwargs.get('conditionals_for_samples', False)

        self.triplet_margin = kwargs.get('triplet_margin', 1.0)
        self.triplet_weight = kwargs.get('triplet_weight', 1.0)

        self.last_losses = {
            'g_loss': 10.,
            'd_loss': 10.
        }

        self.all_losses = {
            'g_loss': [],
            'd_loss': [],
            'd_triplet': [],
            'g_triplet': [],
            'ae_loss': [],
        }

        self.svm = LinearSVC()
        self.svgan_type = 'epoch_svm'  # or 'batch_svm'
        self.did_train_svm_for_the_first_time = False

        self.embedding_size = embedding_dim
        self.aux_clas_weight = aux_clas_weight
        self.ae_weight = ae_weight
        self.d_triplet_weight = d_triplet_weight
        self.g_triplet_weight = g_triplet_weight
        self.isolate_d_classifier = isolate_d_classifier

        pprint(vars(self))

        self.build_model()

    train_on_batch = TOPGAN.__dict__['train_on_batch']
    predict = TOPGAN.__dict__['predict']
    build_trainer = TOPGAN.__dict__['build_trainer']
    build_Gx = TOPGAN.__dict__['build_Gx']
    define_loss_functions = TOPGAN.__dict__['define_loss_functions']
    did_collapse = TOPGAN.__dict__['did_collapse']
    build_optmizers = TOPGAN.__dict__['build_optmizers']

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        batchsize = len(x_data)

        # perform label smoothing if applicable
        y_pos, y_neg = smooth_binary_labels(batchsize, self.label_smoothing, one_sided_smoothing=False)
        y = np.stack((y_neg, y_pos), axis=1)

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        noise = np.random.normal(scale=0.1, size=x_data.shape)

        x_permutation = np.array(np.random.permutation(batchsize), dtype='int64')
        input_data = [x_data, noise, x_permutation, z_latent_dis]
        label_data = [y, y, x_data]

        # train both networks
        _, _, d_triplet, ae_loss = self.ae_triplet_trainer.train_on_batch(input_data, label_data)
        _, d_loss, d_triplet, ae_loss = self.dis_trainer.train_on_batch(input_data, label_data)
        _, g_loss, g_triplet, _ = self.gen_trainer.train_on_batch(input_data, label_data)
        if self.last_losses['d_loss'] < self.dis_loss_control:
            _, g_loss, g_triplet, _ = self.gen_trainer.train_on_batch(input_data, label_data)
        if self.last_losses['d_loss'] < self.dis_loss_control * 1e-2:
            for i in range(0, 5):
                _, g_loss, g_triplet, _ = self.gen_trainer.train_on_batch(input_data, label_data)

        losses = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'd_triplet': d_triplet,
            'g_triplet': g_triplet,
            'ae_loss': ae_loss
        }

        self.last_losses = losses
        for k, v in losses.items():
            self.all_losses[k].append(v)
        return losses

    def build_trainer(self):
        input_x = Input(shape=self.input_shape)
        input_noise = Input(shape=self.input_shape)
        input_x_perm = Input(shape=(1,), dtype='int64')
        input_z = Input(shape=(self.z_dims,))

        assert self.f_D is not None

        x_noisy = Add()([input_x, input_noise])
        x_hat = self.f_Gx(input_z)
        x_hat_noisy = Add()([x_hat, input_noise])

        negative_embedding, p, _ = self.f_D(x_hat_noisy)
        anchor_embedding, q, x_reconstructed = self.f_D(x_noisy)
        positive_embedding = Lambda(lambda x: K.squeeze(K.gather(anchor_embedding, input_x_perm), 1))(anchor_embedding)

        input = [input_x, input_noise, input_x_perm, input_z]

        concatenated_dis = Concatenate(axis=-1, name="dis_classification")([p, q])
        concatenated_triplet = Concatenate(axis=-1, name="triplet")([anchor_embedding, positive_embedding, negative_embedding])
        output = [concatenated_dis, concatenated_triplet, x_reconstructed]
        return Model(input, output, name='svgan')

    def build_model(self):

        self.f_Gx = self.build_Gx()  # Moriarty, the encoder
        self.f_D = self.build_D()   # Sherlock, the detective
        self.f_Gx.summary()
        self.f_D.summary()
        self.encoder.summary()
        self.decoder.summary()
        self.aux_classifier.summary()

        opt_d, opt_g = self.build_optmizers()
        loss_d, loss_g, triplet_d_loss, triplet_g_loss = self.define_loss_functions()

        # build discriminator
        self.dis_trainer = self.build_trainer()
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, not self.isolate_d_classifier)
        set_trainable(self.aux_classifier, True)
        self.dis_trainer.compile(optimizer=opt_d,
                                 loss=[loss_d, triplet_d_loss, 'mae'],
                                 loss_weights=[self.aux_clas_weight, 0., 0.])

        # build autoencoder+triplet
        self.ae_triplet_trainer = self.build_trainer()
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, True)
        set_trainable(self.aux_classifier, not self.isolate_d_classifier)
        self.ae_triplet_trainer.compile(optimizer=opt_d,
                                        loss=[loss_d, triplet_d_loss, 'mae'],
                                        loss_weights=[0., self.d_triplet_weight, self.ae_weight])

        # build generators
        self.gen_trainer = self.build_trainer()
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=opt_g,
                                 loss=[loss_g, triplet_g_loss, 'mae'],
                                 loss_weights=[self.aux_clas_weight, self.g_triplet_weight, 0.])

        self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('gen_trainer')
        self.store_to_save('f_Gx')
        self.store_to_save('f_D')

    def save_model(self, out_dir, epoch):
        self.trainers['f_Gx'] = self.f_Gx
        self.trainers['f_D'] = self.f_D
        super().save_model(out_dir, epoch)

    def plot_losses_hist(self, out_dir):
        plot_metrics(out_dir,
                     metrics_list=[(self.all_losses['g_loss'], self.all_losses['d_loss']),
                                   self.all_losses['g_triplet'],
                                   self.all_losses['d_triplet'],
                                   self.all_losses['ae_loss']],
                     iterations_list=list(range(len(self.all_losses['ae_loss']))),
                     metric_names=[('g_loss', 'd_loss'), 'g_triplet', 'd_triplet', 'ae_loss'],
                     legend=[True, True, True, True],
                     figsize=(16, 16),
                     wspace=0.4)

    def save_images(self, samples, filename, conditionals_for_samples=None):
        '''
        Save images generated from random sample numbers
        '''
        imgs = self.predict(samples)
        np.random.seed(14)
        perm = np.random.permutation(len(self.dataset))
        imgs_from_dataset = self.dataset.images[perm[:10]]
        noise = np.random.normal(scale=0.1, size=imgs_from_dataset.shape)
        imgs_from_dataset += noise
        np.random.seed()

        imgs[80:90] = imgs_from_dataset
        encoding = self.encoder.predict(imgs_from_dataset)
        x_hat = self.decoder.predict(encoding)
        imgs[90:] = x_hat
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        self.save_image_as_plot(imgs, filename)

    def build_encoder(self):
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.2)(x_input)

        x_embedding = Flatten()(x)
        x_embedding = Dense(self.embedding_size)(x_embedding)
        x_embedding = BatchNormalization()(x_embedding)
        x_embedding = Activation('relu')(x_embedding)

        return Model(x_input, x_embedding)

    def build_decoder(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x_hat = Dense(64 * self.input_shape[0] // 2 * self.input_shape[1] // 2)(embedding_input)
        x_hat = BatchNormalization()(x_hat)
        x_hat = Activation('relu')(x_hat)

        x_hat = Reshape((self.input_shape[0] // 2, self.input_shape[1] // 2, 64))(x_hat)

        x_hat = BasicDeconvLayer(self.input_shape[2], (4, 4), strides=(2, 2), bnorm=False, activation='sigmoid', padding='same')(x_hat)

        return Model(embedding_input, x_hat)

    def build_aux_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Dense(256)(embedding_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(embedding_input, x)

    def build_D(self):
        """
        Network Architecture based on the one presented in infoGAN
        """
        x_input = Input(shape=self.input_shape)

        self.encoder = self.build_encoder()
        x_embedding = self.encoder(x_input)

        self.aux_classifier = self.build_aux_classifier()
        discriminator = self.aux_classifier(x_embedding)

        self.decoder = self.build_decoder()
        x_hat = self.decoder(x_embedding)

        return Model(x_input, [x_embedding, discriminator, x_hat])


class TOPGANwithAEfromEBGAN(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=128,
                 name='topgan_ae_ebgan',
                 aux_clas_weight=1.,
                 ae_weight=1.,
                 embedding_dim=256,
                 g_triplet_weight=0.,
                 d_triplet_weight=1.,
                 isolate_d_classifier=False,
                 **kwargs):
        super().__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.f_Gx = None
        self.f_D = None

        self.gen_trainer = None
        self.dis_trainer = None

        self.last_losses = {
            'g_loss': 10.,
            'd_loss': 10.
        }

        self.all_losses = {
            'g_loss': [],
            'd_loss': [],
            'd_triplet': [],
            'g_triplet': [],
            'ae_loss': [],
        }

        self.triplet_margin = kwargs.get('triplet_margin', 1.0)
        self.triplet_weight = kwargs.get('triplet_weight', 1.0)

        self.embedding_size = embedding_dim
        self.aux_clas_weight = aux_clas_weight
        self.ae_weight = ae_weight
        self.d_triplet_weight = d_triplet_weight
        self.g_triplet_weight = g_triplet_weight
        self.isolate_d_classifier = isolate_d_classifier

        pprint(vars(self))

        self.build_model()

    train_on_batch = TOPGANwithAE.__dict__['train_on_batch']
    predict = TOPGANwithAE.__dict__['predict']
    build_trainer = TOPGANwithAE.__dict__['build_trainer']
    build_Gx = TOPGANwithAE.__dict__['build_Gx']
    define_loss_functions = TOPGANwithAE.__dict__['define_loss_functions']
    save_images = TOPGANwithAE.__dict__['save_images']
    did_collapse = TOPGANwithAE.__dict__['did_collapse']
    build_optmizers = TOPGANwithAE.__dict__['build_optmizers']
    build_D = TOPGANwithAE.__dict__['build_D']
    train_on_batch = TOPGANwithAE.__dict__['train_on_batch']
    build_model = TOPGANwithAE.__dict__['build_model']
    plot_losses_hist = TOPGANwithAE.__dict__['plot_losses_hist']

    def save_model(self, out_dir, epoch):
        self.trainers['f_Gx'] = self.f_Gx
        self.trainers['f_D'] = self.f_D
        super().save_model(out_dir, epoch)

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(inputs)
        x = BasicConvLayer(128, (4, 4), strides=(2, 2), bnorm=True, activation='relu')(x)
        x = BasicConvLayer(256, (4, 4), strides=(2, 2), bnorm=True, activation='relu')(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(self.embedding_size)(x)
        x = Activation('linear')(x)

        return Model(inputs, x)

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))
        orig_channels = self.input_shape[2]

        w = self.input_shape[0] // 2**2
        x = Dense(w * w * 128)(z_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape((w, w, 128))(x)

        x = BasicDeconvLayer(256, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(128, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(32, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)

        x = BasicConvLayer(32, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(orig_channels, (1, 1), activation='sigmoid', bnorm=False)(x)

        return Model(z_input, x)

    def build_aux_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Dense(256)(embedding_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(embedding_input, x)

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
        orig_channels = self.input_shape[2]

        w = self.input_shape[0] // 2**2
        x = Dense(w * w * 128)(z_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape((w, w, 128))(x)

        x = BasicDeconvLayer(256, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(128, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(32, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)

        x = BasicConvLayer(32, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(orig_channels, (1, 1), activation='sigmoid', bnorm=False)(x)

        return Model(z_input, x)

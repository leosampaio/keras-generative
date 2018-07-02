import os
import random
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.svm import LinearSVC

import keras
from keras.engine.topology import Layer
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU, LocallyConnected2D,
                          Lambda, AveragePooling1D, GlobalAveragePooling2D)
from keras.optimizers import Adam, Adadelta, RMSprop
from keras import initializers
from keras import backend as K
from keras.applications.mobilenet import MobileNet

from core.models import BaseModel

from .utils import *
from .layers import *


def generator_lossfun(y_true, y_pred):
    """
    the more positive predictions we get for generator samples, 
    the worse it is, hence the sign inversion
    """
    return -K.mean(y_pred)


class SupportVectorWGAN(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=128,
                 name='sv_wgan',
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

        self.last_losses = {
            'g_loss': 10.,
            'd_loss': 10.
        }

        self.svm = LinearSVC()
        self.svgan_type = 'epoch_svm'  # or 'batch_svm'
        self.did_train_svm_for_the_first_time = False

        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        if not self.did_train_svm_for_the_first_time:
            self.did_train_svm_for_the_first_time = True
            self.did_train_over_an_epoch()

        batchsize = len(x_data)

        # perform label smoothing if applicable

        y_pos = np.ones((batchsize, 1), dtype='float32')
        y_neg = y_pos * -1
        y_combined = np.stack((y_neg, y_pos), axis=1)

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        if self.svgan_type == 'batch_svm':
            x_feats = self.f_preprocessing.predict(x_data)
            x_hat_feats = self.f_preprocessing.predict(
                self.f_Gx.predict(z_latent_dis))
            x_concat = np.stack((x_hat_feats, x_feats), axis=1)
            self.svm.fit(np.reshape(x_concat, (batchsize * 2, -1)), np.reshape(y, (batchsize * 2)))

        svm_coef = np.reshape(np.repeat(self.svm.coef_, batchsize), (batchsize, -1))
        svm_intercept = np.reshape(np.repeat(self.svm.intercept_, batchsize), (batchsize, 1))

        input_data = [x_data, z_latent_dis, svm_coef, svm_intercept]

        # train
        g_loss = self.gen_trainer.train_on_batch(input_data, y_neg)
        losses = {
            'g_loss': g_loss,
            'd_loss': -g_loss
        }

        self.last_losses = losses
        return losses

    def did_train_over_an_epoch(self):
        if self.svgan_type == 'epoch_svm':

            # retrain the svm over the entire dataset
            dataset_size = len(self.dataset)

            y_pos = np.ones(dataset_size, dtype='float32')
            y_neg = -y_pos
            y = np.stack((y_neg, y_pos), axis=1)
            z_latent_dis = np.random.normal(size=(dataset_size, self.z_dims))
            x_feats = self.f_preprocessing.predict(self.dataset.images)
            x_hat_feats = self.f_preprocessing.predict(
                self.f_Gx.predict(z_latent_dis))
            x_concat = np.stack((x_hat_feats, x_feats), axis=1)

            self.svm.fit(np.reshape(x_concat, (dataset_size * 2, -1)), np.reshape(y, (dataset_size * 2)))

    def predict(self, z_samples):
        return self.f_Gx.predict(z_samples)

    def build_trainer(self):
        input_x = Input(shape=self.input_shape)
        input_z = Input(shape=(self.z_dims, ))
        svm_coeficients = Input(shape=(1024,))
        svm_b = Input(shape=(1,))

        assert self.f_D is not None

        input_x_feats = self.f_preprocessing(input_x)
        x_hat = self.f_preprocessing(self.f_Gx(input_z))
        p = self.f_D([x_hat, svm_coeficients, svm_b])
        # q = self.f_D([input_x_feats, svm_coeficients, svm_b])
        input = [input_x, input_z, svm_coeficients, svm_b]

        # concatenated = Concatenate(axis=-1)([p, q])
        return Model(input, p, name='svgan')

    def build_model(self):

        self.f_Gx = self.build_Gx()  # Moriarty, the encoder
        self.f_D = self.build_D()   # Sherlock, the detective
        self.f_preprocessing = self.build_f_preprocessing()
        self.f_Gx.summary()
        self.f_D.summary()

        opt_g = Adam(lr=self.lr)
        loss_g = generator_lossfun

        # build generators
        self.gen_trainer = self.build_trainer()
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_preprocessing, False)
        self.gen_trainer.compile(optimizer=opt_g, loss=loss_g)

        self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('gen_trainer')

    def save_model(self, out_dir, epoch):
        self.trainers['f_Gx'] = self.f_Gx
        super().save_model(out_dir, epoch)
        del self.trainers['f_Gx']

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
        x = Dense(32 * 32 * 3, activation='sigmoid')(x)
        x = Reshape((32, 32, 3))(x)

        return Model(z_input, x)

    def build_D(self):

        input_features = Input(shape=(1024,))
        svm_coeficients = Input(shape=(1024,))
        svm_b = Input(shape=(1,))

        def svm_prediction(inputs):
            x, coef, b = inputs[0], inputs[1], inputs[2]
            x = K.expand_dims(x, axis=-1)
            coef = K.expand_dims(coef, axis=-1)
            pred = K.batch_dot(x, coef, axes=[1, 1])
            pred = K.reshape(pred, (-1, 1))
            return pred + b

        svm_layer = Lambda(svm_prediction, output_shape=(1,))
        output = svm_layer([input_features, svm_coeficients, svm_b])
        return Model([input_features, svm_coeficients, svm_b], output)

    def build_f_preprocessing(self):

        input_x = Input(shape=self.input_shape)

        # load pretrained model
        base_model = MobileNet(input_shape=(128, 128, 3), weights='imagenet', include_top=False)
        pret_model = Model(inputs=base_model.input, outputs=base_model.layers[-4].output)

        # upsample to match input
        x = UpSampling2D(size=(128 // self.input_shape[0], 128 // self.input_shape[0]))(input_x)
        x = pret_model(x)

        # add pooling to shrink output size
        x = GlobalAveragePooling2D()(x)

        model = Model(input_x, x)
        return model

    def build_optmizers(self):
        pass


class SupportVectorWGANwithMSE(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=128,
                 name='sv_wgan_mse',
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

        self.last_losses = {
            'g_loss': 10.,
            'd_loss': 10.
        }

        self.svm = LinearSVC()
        self.svgan_type = 'epoch_svm'  # or 'batch_svm'
        self.did_train_svm_for_the_first_time = False

        self.build_model()

    did_train_over_an_epoch = SupportVectorWGAN.__dict__['did_train_over_an_epoch']
    predict = SupportVectorWGAN.__dict__['predict']
    save_images = SupportVectorWGAN.__dict__['save_images']
    did_collapse = SupportVectorWGAN.__dict__['did_collapse']
    build_Gx = SupportVectorWGAN.__dict__['build_Gx']
    build_D = SupportVectorWGAN.__dict__['build_D']
    build_f_preprocessing = SupportVectorWGAN.__dict__['build_f_preprocessing']
    build_optmizers = SupportVectorWGAN.__dict__['build_optmizers']

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        if not self.did_train_svm_for_the_first_time:
            self.did_train_svm_for_the_first_time = True
            self.did_train_over_an_epoch()

        batchsize = len(x_data)

        # perform label smoothing if applicable

        y_pos = np.ones((batchsize, 1), dtype='float32')
        y_neg = y_pos * -1
        y_combined = np.stack((y_neg, y_pos), axis=1)

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        if self.svgan_type == 'batch_svm':
            x_feats = self.f_preprocessing.predict(x_data)
            x_hat_feats = self.f_preprocessing.predict(
                self.f_Gx.predict(z_latent_dis))
            x_concat = np.stack((x_hat_feats, x_feats), axis=1)
            self.svm.fit(np.reshape(x_concat, (batchsize * 2, -1)), np.reshape(y, (batchsize * 2)))

        svm_coef = np.reshape(np.repeat(self.svm.coef_, batchsize), (batchsize, -1))
        svm_intercept = np.reshape(np.repeat(self.svm.intercept_, batchsize), (batchsize, 1))

        input_data = [x_data, z_latent_dis, svm_coef, svm_intercept]

        # train
        g_loss = self.gen_trainer.train_on_batch(input_data, [y_neg, x_data])
        losses = {
            'g_loss': g_loss[1],
            'd_loss': -g_loss[1],
            'content_consistency_loss': g_loss[2]
        }

        self.last_losses = losses
        return losses

    def build_trainer(self):
        input_x = Input(shape=self.input_shape)
        input_z = Input(shape=(self.z_dims, ))
        svm_coeficients = Input(shape=(1024,))
        svm_b = Input(shape=(1,))

        assert self.f_D is not None

        input_x_feats = self.f_preprocessing(input_x)
        x_hat = self.f_Gx(input_z)
        x_hat_feats = self.f_preprocessing(x_hat)
        p = self.f_D([x_hat_feats, svm_coeficients, svm_b])
        # q = self.f_D([input_x_feats, svm_coeficients, svm_b])
        input = [input_x, input_z, svm_coeficients, svm_b]

        # concatenated = Concatenate(axis=-1)([p, q])
        return Model(input, [p, x_hat], name='svgan')

    def build_model(self):

        self.f_Gx = self.build_Gx()  # Moriarty, the encoder
        self.f_D = self.build_D()   # Sherlock, the detective
        self.f_preprocessing = self.build_f_preprocessing()
        self.f_Gx.summary()
        self.f_D.summary()

        opt_g = Adam(lr=self.lr)
        loss_g = generator_lossfun

        # build generators
        self.gen_trainer = self.build_trainer()
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_preprocessing, False)
        self.gen_trainer.compile(optimizer=opt_g, loss=[loss_g, 'mse'],
                                 loss_weights=[1., 0.1])

        self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('gen_trainer')

    def save_model(self, out_dir, epoch):
        self.trainers['f_Gx'] = self.f_Gx
        super().save_model(out_dir, epoch)
        del self.trainers['f_Gx']
import os
import random
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.svm import LinearSVC

import keras
from keras.engine.topology import Layer
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU,
                          Lambda, AveragePooling1D, GlobalAveragePooling2D)
from keras.optimizers import Adam, Adadelta
from keras import initializers
from keras import backend as K
from keras.applications.mobilenet import MobileNet

from core.models import BaseModel

from .utils import *
from .layers import *


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


class SupportVectorGAN(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=128,
                 name='sv_gan',
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

        y_pos, y_neg = smooth_binary_labels(batchsize, self.label_smoothing, one_sided_smoothing=False)
        y = np.stack((y_neg, y_pos), axis=1)

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        x_feats = self.f_preprocessing.predict(x_data)
        x_hat_feats = self.f_preprocessing.predict(
            self.f_Gx.predict(z_latent_dis))
        x_concat = np.stack((x_hat_feats, x_feats), axis=1)

        if self.svgan_type == 'batch_svm':
            self.svm.fit(np.reshape(x_concat, (batchsize * 2, -1)), np.reshape(y, (batchsize * 2, -1)))

        svm_score = self.svm.score(np.reshape(x_concat, (batchsize * 2, -1)), np.reshape(y, (batchsize * 2, -1)))
        svm_coef = np.reshape(np.repeat(self.svm.coef_, batchsize), (batchsize, -1))
        svm_intercept = np.reshape(np.repeat(self.svm.intercept_, batchsize), (batchsize, 1))
        input_data = [x_feats, z_latent_dis, svm_coef, svm_intercept]

        # train both networks
        g_loss = self.gen_trainer.train_on_batch(input_data, y)
        losses = {
            'g_loss': g_loss,
            'd_loss': 1.0 - svm_score
        }

        self.last_losses = losses
        return losses

    def did_train_over_an_epoch(self):
        if self.svgan_type == 'epoch_svm':

            # retrain the svm over the entire dataset
            dataset_size = len(self.dataset)

            y_pos, y_neg = smooth_binary_labels(dataset_size, self.label_smoothing, one_sided_smoothing=False)
            y = np.stack((y_neg, y_pos), axis=1)
            z_latent_dis = np.random.normal(size=(dataset_size, self.z_dims))
            x_feats = self.f_preprocessing.predict(self.dataset.images)
            x_hat_feats = self.f_preprocessing.predict(
                self.f_Gx.predict(z_latent_dis))
            x_concat = np.stack((x_hat_feats, x_feats), axis=1)

            self.svm.fit(np.reshape(x_concat, (dataset_size * 2, -1)), np.reshape(y, (dataset_size * 2, -1)))

    def predict(self, z_samples):
        return self.f_Gx.predict(z_samples)

    def build_trainer(self):
        input_x_feats = Input(shape=(1024,))
        input_z = Input(shape=(self.z_dims, ))
        svm_coeficients = Input(shape=(1024,))
        svm_b = Input(shape=(1,))

        assert self.f_D is not None

        x_hat = self.f_preprocessing(self.f_Gx(input_z))
        p = self.f_D([x_hat, svm_coeficients, svm_b])
        q = self.f_D([input_x_feats, svm_coeficients, svm_b])
        input = [input_x_feats, input_z, svm_coeficients, svm_b]

        concatenated = Concatenate(axis=-1)([p, q])
        return Model(input, concatenated, name='svgan')

    def build_model(self):

        self.f_Gx = self.build_Gx()  # Moriarty, the encoder
        self.f_D = self.build_D()   # Sherlock, the detective
        self.f_preprocessing = self.build_f_preprocessing()
        self.f_Gx.summary()
        self.f_D.summary()

        opt_d, opt_g = self.build_optmizers()
        loss_d, loss_g = self.define_loss_functions()

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

    def define_loss_functions(self):
        return discriminator_lossfun, generator_lossfun

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
        opt_d = Adam(lr=self.lr)
        opt_g = Adam(lr=self.lr)
        return opt_d, opt_g



# def build(self):

#     input_x = Input(shape=self.input_shape)

#     # load pretrained model
#     model = MobileNet(input_shape=(128, 128, 3), weights='imagenet', include_top=False)

#     # include top

#     model = Model(input_x, x)
#     set_trainable([model], False)
#     for layer in model.layers[:7:-1]
#         layer.trainable = True
#     model.compile(optimizer=opt_g, loss=loss_g)

#     for e in epochs:
#         for b in batches:
#             x = get_batch()

#         model.train_on_batch(input_data, y)

#     model.save_weights('arquivo.h5')
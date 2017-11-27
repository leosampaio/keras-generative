import os
import numpy as np

import keras
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Lambda, Reshape, Concatenate
from keras.layers import Activation, LeakyReLU, ELU
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Dropout
from keras.optimizers import Adam, Adadelta
from keras import backend as K
from matplotlib import gridspec

from .base import BaseModel

from .utils import *
from .layers import *


class DiscriminatorLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(DiscriminatorLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_fake)

        loss_real = keras.metrics.binary_crossentropy(y_pos, y_real)
        loss_fake = keras.metrics.binary_crossentropy(y_neg, y_fake)

        return K.mean(loss_real + loss_fake)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake = inputs[1]
        loss = self.lossfun(y_real, y_fake)
        self.add_loss(loss, inputs=inputs)

        return y_real


def discriminator_accuracy(y_real, y_fake):
    y_pos = K.ones_like(y_real)
    y_neg = K.zeros_like(y_fake)
    acc_real = keras.metrics.binary_accuracy(y_pos, y_real)
    acc_fake = keras.metrics.binary_accuracy(y_neg, y_fake)
    return 0.5 * K.mean(acc_real + acc_fake)


class EncoderLossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(EncoderLossLayer, self).__init__(**kwargs)

    def lossfun(self, y_real, y_fake):
        y_pos = K.ones_like(y_real)
        y_neg = K.zeros_like(y_fake)

        loss_fake = keras.metrics.binary_crossentropy(y_pos, y_fake)
        loss_real = keras.metrics.binary_crossentropy(y_neg, y_real)

        return K.mean(loss_real + loss_fake)

    def call(self, inputs):
        y_real = inputs[0]
        y_fake = inputs[1]
        loss = self.lossfun(y_real, y_fake)
        self.add_loss(loss, inputs=inputs)

        return y_real


def encoder_accuracy(y_real, y_fake):
    y_pos = K.ones_like(y_fake)
    y_neg = K.zeros_like(y_real)
    acc_fake = keras.metrics.binary_accuracy(y_pos, y_fake)
    acc_real = keras.metrics.binary_accuracy(y_neg, y_real)
    return 0.5 * K.mean(acc_real + acc_fake)


class AAE(BaseModel):
    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=256,
                 name='aae',
                 decoder_loss='mse',
                 **kwargs
                 ):
        super(AAE, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.encoder = None
        self.decoder = None
        self.discriminator = None

        self.enc_trainer = None
        self.ae_trainer = None
        self.dis_trainer = None

        self.decoder_loss = decoder_loss

        self.build_model()

    def train_on_batch(self, x_real):
        batchsize = len(x_real)
        y_pos = np.ones(batchsize, dtype='float32')
        y_neg = np.zeros(batchsize, dtype='float32')

        # # Reconstruction Phase - train the autoencoder part
        # ae_loss = self.ae_trainer.train_on_batch(x_real, x_real)

        # Regularization Phase - train the discriminator and the encoder
        z_real = np.random.normal(size=(batchsize, self.z_dims)).astype('float32')
        z_fake = self.encoder.predict(x_real)
        half_batch = batchsize // 2

        dis_loss_real, dis_acc_real = self.dis_trainer.train_on_batch(z_real[:half_batch], y_pos[:half_batch])
        dis_loss_fake, dis_acc_fake = self.dis_trainer.train_on_batch(z_fake[:half_batch], y_neg[:half_batch])
        dis_loss, dis_acc = 0.5 * np.add(dis_loss_real, dis_loss_fake), 0.5 * np.add(dis_acc_real, dis_acc_fake)

        # enc_loss, enc_acc = self.enc_trainer.train_on_batch(x_real, y_pos)
        i = 0
        while True:
            ae_loss = self.ae_trainer.train_on_batch(x_real, [x_real, y_pos])
            if ae_loss[0] < 4 or i > 10:
                break
            i += 1
            print('Retrained ae {} times'.format(i))

        losses = {
            'ae_loss': ae_loss[0],
            'enc_loss': ae_loss[1],
            'dis_loss': dis_loss,
            'dis_acc': dis_acc,
            # 'enc_loss': enc_loss,
            # 'enc_acc': enc_acc,
        }
        return losses

    def predict(self, z_samples):
        return self.decoder.predict(z_samples)

    def save_images(self, samples, filename):
        generated = self.decoder.predict(samples[:40])
        generated = generated * 0.5 + 0.5
        generated = np.clip(generated, 0.0, 1.0)
        # plot reconstructions
        originals = self.dataset[:30]
        encodings = self.encoder.predict(originals)
        reconstructions = self.decoder.predict(encodings) * 0.5 + 0.5
        reconstructions = np.clip(reconstructions, 0.0, 1.0)
        originals = originals * 0.5 + 0.5
        originals = np.clip(originals, 0.0, 1.0)

        fig = plt.figure(figsize=(8, 8))
        grid = gridspec.GridSpec(10, 10, wspace=0.1, hspace=0.1)
        for i in range(30):
            # plot original images
            ax = plt.Subplot(fig, grid[2 * i])
            ax.imshow(originals[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)
            # plot reconstructions
            ax = plt.Subplot(fig, grid[2 * i + 1])
            ax.imshow(reconstructions[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)
        for i in range(60, 100):
            # plot generated images
            ax = plt.Subplot(fig, grid[i])
            ax.imshow(generated[i - 60, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
            ax.axis('off')
            fig.add_subplot(ax)

        fig.savefig(filename, dpi=200)
        plt.close(fig)

    def build_model(self):
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.discriminator = self.build_discriminator()

        self.encoder.summary()
        self.decoder.summary()
        self.discriminator.summary()

        # Compile discriminator
        set_trainable(self.encoder, False)
        set_trainable(self.decoder, False)
        set_trainable(self.discriminator, True)

        x_real = Input(shape=self.input_shape)

        z_fake = self.encoder(x_real)
        x_reconstructed = self.decoder(z_fake)

        y_fake = self.discriminator(z_fake)

        self.dis_trainer = self.discriminator
        self.dis_trainer.compile(loss='binary_crossentropy',
                                 optimizer=Adam(lr=1.0e-5, beta_1=0.1),
                                 metrics=['accuracy'])
        self.dis_trainer.summary()

        # Compile autoencoder
        set_trainable(self.encoder, True)
        set_trainable(self.decoder, True)
        set_trainable(self.discriminator, False)

        self.ae_trainer = Model(x_real, [x_reconstructed, y_fake])
        self.ae_trainer.compile(loss=[self.decoder_loss, 'binary_crossentropy'],
                                # loss_weights=[0.7, 0.3],
                                optimizer=Adam(lr=1.0e-4, beta_1=0.5), )
        self.ae_trainer.summary()

        # # Compile encoder
        #
        # self.enc_trainer = Model(x_real, y_fake)
        # self.enc_trainer.compile(loss='binary_crossentropy',
        #                          optimizer=Adam(),
        #                          metrics=['accuracy'])
        # self.enc_trainer.summary()
        #
        # Store trainers
        self.store_to_save('dis_trainer')
        self.store_to_save('ae_trainer')
        # self.store_to_save('enc_trainer')

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=64, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=128, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024, activation='relu')(x)

        z = Dense(self.z_dims, )(x)

        return Model(inputs, z)

    def build_decoder(self):
        inputs = Input(shape=(self.z_dims,))
        w = self.input_shape[0] // (2 ** 3)
        x = Dense(w * w * 256)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape((w, w, 256))(x)

        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2))(x)

        img_size, img_channels = self.input_shape[0], self.input_shape[2]
        x = BasicDeconvLayer(filters=img_size, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=img_channels, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model(inputs, x)

    def build_discriminator(self):
        z_inputs = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_inputs)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = Flatten()(z)

        z = Dropout(0.2)(z)
        z = Dense(1024)(z)
        z = LeakyReLU(0.1)(z)

        z = Dropout(0.2)(z)
        z = Dense(1)(z)
        z = Activation('sigmoid')(z)

        return Model(z_inputs, z)


class AAE2(AAE):
    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=256,
                 name='aae2',
                 decoder_loss='mse',
                 use_dropout=False,
                 **kwargs
                 ):
        super().__init__(input_shape=input_shape, z_dims=z_dims, name=name, decoder_loss=decoder_loss, **kwargs)
        self.use_dropout = use_dropout

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(inputs)
        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=512, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)

        x = Flatten()(x)
        # x = Dense(1024, activation='relu')(x)

        z = Dense(self.z_dims, )(x)

        return Model(inputs, z)

    def build_decoder(self):
        inputs = Input(shape=(self.z_dims,))
        w = self.input_shape[0] // (2 ** 4)
        x = Dense(w * w * 512)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 512))(x)

        x = BasicDeconvLayer(filters=512, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        if self.use_dropout:
            x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        if self.use_dropout:
            x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicDeconvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, kernel_size=(3, 3), activation='tanh')(x)

        return Model(inputs, x)


class DrAAE2(AAE2):
    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=256,
                 name='draae2',
                 decoder_loss='mse',
                 **kwargs
                 ):
        super().__init__(input_shape=input_shape, z_dims=z_dims, name=name, decoder_loss=decoder_loss, use_dropout=True,
                         **kwargs)


class BinAAE(AAE):
    """
    Uses binary cross-entropy as the loss function for the Autoencoder's reconstructions
    """

    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=256,
                 name='baae',
                 decoder_loss='binary_crossentropy',
                 **kwargs
                 ):
        super().__init__(input_shape=input_shape, z_dims=z_dims, name=name, decoder_loss=decoder_loss, **kwargs)

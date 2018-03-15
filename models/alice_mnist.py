import keras.backend as K
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU, LocallyConnected2D,
                          Lambda)
from keras.optimizers import Adam, SGD, RMSprop
import numpy as np

from models import ALICE
from models.layers import BasicConvLayer, BasicDeconvLayer, SampleNormal
from models.utils import set_trainable, zero_loss


class ALICEforMNIST(ALICE):

    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'alice_for_mnist'
        super().__init__(*args, **kwargs)

    def build_Gz(self):
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x_input)
        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(256, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(512, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)

        x = Flatten()(x)

        # the output is an average (mu) and std variation (sigma)
        # describing the distribution that better describes the input
        mu = Dense(self.z_dims)(x)
        mu = Activation('linear')(mu)
        sigma = Dense(self.z_dims)(x)
        sigma = Activation('linear')(sigma)

        # use the generated values to sample random z from the latent space
        concatenated = Concatenate(axis=-1)([mu, sigma])
        output = Lambda(
            function=lambda x: x[:, :self.z_dims] + (K.exp(x[:, self.z_dims:]) * (K.random_normal(shape=K.shape(x[:, self.z_dims:])))),
            output_shape=(self.z_dims, )
        )(concatenated)

        return Model(x_input, output)

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
        orig_channels = self.input_shape[2]

        x = Reshape((1, 1, -1))(z_input)

        x = BasicDeconvLayer(128, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(64, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(32, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)

        x = BasicConvLayer(32, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(orig_channels, (1, 1), activation='sigmoid', bnorm=False)(x)

        return Model(z_input, x)

    def build_D(self):
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_input)
        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(128, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = Flatten()(x)

        z_input = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_input)
        z = BasicConvLayer(128, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
        z = BasicConvLayer(128, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
        z = Flatten()(z)

        xz = Concatenate(axis=-1)([x, z])

        xz = Dense(128)(xz)
        xz = LeakyReLU(0.01)(xz)
        xz = Dropout(0.2)(xz)

        xz = Dense(1)(xz)
        xz = Activation('sigmoid')(xz)

        return Model([x_input, z_input], xz)

    def build_D_cycle(self):
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_input)
        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(128, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = Flatten()(x)

        x_hat_input = Input(shape=self.input_shape)
        x_hat = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_hat_input)
        x_hat = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_hat)
        x_hat = BasicConvLayer(128, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_hat)
        x_hat = Flatten()(x_hat)

        x_x_hat = Concatenate(axis=-1)([x, x_hat])

        x_x_hat = Dense(128)(x_x_hat)
        x_x_hat = LeakyReLU(0.01)(x_x_hat)
        x_x_hat = Dropout(0.2)(x_x_hat)

        x_x_hat = Dense(1)(x_x_hat)
        x_x_hat = Activation('sigmoid')(x_x_hat)

        return Model([x_input, x_hat_input], x_x_hat)

    def build_optmizers(self):
        opt_d = RMSprop(lr=1e-4)
        opt_g = RMSprop(lr=1e-4)
        return opt_d, opt_g

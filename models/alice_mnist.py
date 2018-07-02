import keras.backend as K
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU,
                          Lambda)
from keras.optimizers import Adam, RMSprop
import numpy as np

from models.alice import ALICE, ExplicitALICE
from models.layers import BasicConvLayer, BasicDeconvLayer
from models.ali_mnist import ALIforSharedExp


class ALICEforMNIST(ALICE):

    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'alice_for_mnist'
        super().__init__(*args, **kwargs)

    def build_Gz(self):
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x_input)
        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(128, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(256, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(512, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)

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

        x = BasicDeconvLayer(256, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicDeconvLayer(128, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicDeconvLayer(64, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicDeconvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicDeconvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)

        x = BasicConvLayer(32, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(32, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(orig_channels, (1, 1), activation='sigmoid', bnorm=False)(x)

        return Model(z_input, x)

    def build_D(self):
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.1)(x_input)
        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(128, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = Flatten()(x)

        z_input = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_input)
        z = BasicConvLayer(128, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.1)(z)
        z = BasicConvLayer(128, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.1)(z)
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

        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.1)(x_input)
        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(128, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = Flatten()(x)

        x_hat_input = Input(shape=self.input_shape)
        x_hat = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.1)(x_hat_input)
        x_hat = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.1)(x_hat)
        x_hat = BasicConvLayer(128, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.1)(x_hat)
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


class ALICEwithDSforMNIST(ALICE):

    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'alice_ds_for_mnist'
        super().__init__(*args, **kwargs)

    def build_Gz(self):
        x_input = Input(shape=self.input_shape)

        res_x = x = BasicConvLayer(64, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x_input)
        x = BasicConvLayer(64, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        res_x = x = BasicConvLayer(64, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x)
        x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        res_x = x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x)
        x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        res_x = x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x)
        x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        res_x = x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x)
        x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x)

        res_x = x_g = BasicConvLayer(128, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x_g = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x_g)
        res_x = x_g = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x_g)
        x_g = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x_g)
        res_x = x_g = BasicConvLayer(128, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x_g)

        res_x = x_d = BasicConvLayer(128, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x_d = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x_d)
        res_x = x_d = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x_d)
        x_d = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x_d)
        res_x = x_d = BasicConvLayer(128, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x_d)

        x_g = Flatten()(x_g)
        x_d = Flatten()(x_d)

        mu_g = Dense(self.z_dims // 2)(x_g)
        mu_g = Activation('linear')(mu_g)
        sigma_g = Dense(self.z_dims // 2)(x_g)
        sigma_g = Activation('linear')(sigma_g)

        mu_d = Dense(self.z_dims // 2)(x_d)
        mu_d = Activation('linear')(mu_d)
        sigma_d = Dense(self.z_dims // 2)(x_d)
        sigma_d = Activation('linear')(sigma_d)

        # use the generated values to sample random z from the latent space
        concatenated_g = Concatenate(axis=-1)([mu_g, sigma_g])
        concatenated_d = Concatenate(axis=-1)([mu_d, sigma_d])
        output_g = Lambda(
            function=lambda x: x[:, :self.z_dims // 2] + (K.exp(x[:, self.z_dims // 2:]) * (K.random_normal(shape=K.shape(x[:, self.z_dims // 2:])))),
            output_shape=(self.z_dims // 2, )
        )(concatenated_g)
        output_d = Lambda(
            function=lambda x: x[:, :self.z_dims // 2] + (K.exp(x[:, self.z_dims // 2:]) * (K.random_normal(shape=K.shape(x[:, self.z_dims // 2:])))),
            output_shape=(self.z_dims // 2, )
        )(concatenated_d)

        concatenated = Concatenate(axis=-1)([output_g, output_d])
        return Model(x_input, concatenated)

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
        orig_channels = self.input_shape[2]

        x = Dense(64, activation='relu')(z_input)
        x = Dense(128, activation='relu')(x)

        x = Reshape((4, 4, 8))(x)

        res_x = x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same', residual=res_x)(x)
        res_x = x = BasicDeconvLayer(64, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same', residual=res_x)(x)
        res_x = x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same', residual=res_x)(x)
        res_x = x = BasicDeconvLayer(64, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same', residual=res_x)(x)
        res_x = x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same', residual=res_x)(x)
        res_x = x = BasicDeconvLayer(64, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same', residual=res_x)(x)
        res_x = x = BasicDeconvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same')(x)
        x = BasicDeconvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01, padding='same', residual=res_x)(x)

        x = BasicConvLayer(32, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(orig_channels, (1, 1), activation='sigmoid', bnorm=False)(x)

        return Model(z_input, x)

    build_D = ALICEforMNIST.__dict__['build_D']
    build_D_cycle = ALICEforMNIST.__dict__['build_D_cycle']
    build_optmizers = ALICEforMNIST.__dict__['build_optmizers']


class ALICEforSharedExp(ALICE):

    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'alice_shared_exp'
        super().__init__(*args, **kwargs)

    build_Gz = ALIforSharedExp.__dict__['build_Gz']
    build_Gx = ALIforSharedExp.__dict__['build_Gx']

    def build_D(self):
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_input)
        res_x = x = BasicConvLayer(64, (3, 3), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01, residual=res_x)(x)
        res_x = x = BasicConvLayer(128, (3, 3), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01, residual=res_x)(x)
        x = Flatten()(x)

        z_input = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_input)
        res_z = z = BasicConvLayer(128, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
        z = BasicConvLayer(128, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01, residual=res_z)(z)
        res_z = z = BasicConvLayer(128, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
        z = BasicConvLayer(128, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01, residual=res_z)(z)
        z = Flatten()(z)

        xz = Concatenate(axis=-1)([x, z])

        xz = Dense(512)(xz)
        xz = LeakyReLU(0.01)(xz)
        xz = Dropout(0.2)(xz)

        xz = Dense(512)(xz)
        xz = LeakyReLU(0.01)(xz)
        xz = Dropout(0.2)(xz)

        xz = Dense(1)(xz)
        xz = Activation('sigmoid')(xz)

        return Model([x_input, z_input], xz)

    def build_D_cycle(self):
        x_input = Input(shape=self.input_shape)
        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_input)
        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)

        x_hat_input = Input(shape=self.input_shape)
        x_hat = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_hat_input)
        x_hat = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_hat)

        x_x_hat = Concatenate(axis=-1)([x, x_hat])
        x_x_hat = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_x_hat)
        x_res = x_x_hat = BasicConvLayer(64, (3, 3), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_x_hat)
        x_x_hat = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01, residual=x_res)(x_x_hat)
        x_res = x_x_hat = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_x_hat)
        x_x_hat = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01, residual=x_res)(x_x_hat)

        x_x_hat = Flatten()(x_x_hat)
        x_x_hat = Dense(512)(x_x_hat)
        x_x_hat = LeakyReLU(0.01)(x_x_hat)
        x_x_hat = Dropout(0.2)(x_x_hat)

        x_x_hat = Dense(512)(x_x_hat)
        x_x_hat = LeakyReLU(0.01)(x_x_hat)
        x_x_hat = Dropout(0.2)(x_x_hat)

        x_x_hat = Dense(1)(x_x_hat)
        x_x_hat = Activation('sigmoid')(x_x_hat)

        return Model([x_input, x_hat_input], x_x_hat)

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr, clipnorm=5.)
        opt_g = Adam(lr=self.lr, clipnorm=5.)
        return opt_d, opt_g

class ExplicitALICEforSharedExp(ExplicitALICE):

    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'ealice_shared'
        super().__init__(*args, **kwargs)

    build_D = ALIforSharedExp.__dict__['build_D']

    def build_Gz(self):
        x_input = Input(shape=self.input_shape)

        res_x = x = BasicConvLayer(64, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x_input)
        x = BasicConvLayer(64, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        res_x = x = BasicConvLayer(64, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x)
        x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        res_x = x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x)
        x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        res_x = x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x)
        x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        res_x = x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x)
        x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x)

        res_x = x_g = BasicConvLayer(128, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x_g = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x_g)
        res_x = x_g = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x_g)
        x_g = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x_g)
        res_x = x_g = BasicConvLayer(128, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x_g)

        res_x = x_d = BasicConvLayer(128, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x_d = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x_d)
        res_x = x_d = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x_d)
        x_d = BasicConvLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x_d)
        res_x = x_d = BasicConvLayer(128, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, residual=res_x)(x_d)

        x_g = Flatten()(x_g)
        x_d = Flatten()(x_d)

        mu_g = Dense(self.z_dims // 2)(x_g)
        mu_g = Activation('linear')(mu_g)
        sigma_g = Dense(self.z_dims // 2)(x_g)
        sigma_g = Activation('linear')(sigma_g)

        mu_d = Dense(self.z_dims // 2)(x_d)
        mu_d = Activation('linear')(mu_d)
        sigma_d = Dense(self.z_dims // 2)(x_d)
        sigma_d = Activation('linear')(sigma_d)

        # use the generated values to sample random z from the latent space
        concatenated_g = Concatenate(axis=-1)([mu_g, sigma_g])
        concatenated_d = Concatenate(axis=-1)([mu_d, sigma_d])
        output_g = Lambda(
            function=lambda x: x[:, :self.z_dims // 2] + (K.exp(x[:, self.z_dims // 2:]) * (K.random_normal(shape=K.shape(x[:, self.z_dims // 2:])))),
            output_shape=(self.z_dims // 2, )
        )(concatenated_g)
        output_d = Lambda(
            function=lambda x: x[:, :self.z_dims // 2] + (K.exp(x[:, self.z_dims // 2:]) * (K.random_normal(shape=K.shape(x[:, self.z_dims // 2:])))),
            output_shape=(self.z_dims // 2, )
        )(concatenated_d)

        concatenated = Concatenate(axis=-1)([output_g, output_d])
        return Model(x_input, concatenated)

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
        orig_channels = self.input_shape[2]

        x = Dense(512)(z_input); x = LeakyReLU(0.1)(x)
        x = Dense(512)(x); x = LeakyReLU(0.1)(x)

        x = Reshape((4, 4, 32))(x)

        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        res_x = x = BasicDeconvLayer(64, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same', residual=res_x)(x)
        res_x = x = BasicDeconvLayer(64, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same', residual=res_x)(x)
        res_x = x = BasicDeconvLayer(64, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same', residual=res_x)(x)
        res_x = x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same', residual=res_x)(x)
        res_x = x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same', residual=res_x)(x)
        res_x = x = BasicDeconvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same', residual=res_x)(x)

        x = BasicConvLayer(64, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(orig_channels, (1, 1), activation='sigmoid', bnorm=False)(x)

        return Model(z_input, x)

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr, clipnorm=5.)
        opt_g = Adam(lr=self.lr, clipnorm=5.)
        return opt_d, opt_g

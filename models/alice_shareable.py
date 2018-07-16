from collections import namedtuple
import keras.backend as K
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU,
                          Lambda)
from keras.optimizers import Adam

from models.alice import ExplicitALICE
from models.layers import BasicConvLayer, BasicDeconvLayer, ResLayer, ResDeconvLayer
from models.ali_mnist import ALIforSharedExp


class ShareableExplicitALICEforSharedExp(ExplicitALICE):

    def __init__(self, share_with=None, n_layers_to_share=0, *args, **kwargs):
        kwargs['name'] = 'ealice_shareable'
        ShareableLayers = namedtuple('ShareableLayers', ['common_branch', 'cross_domain_branch'])
        self.s_layers = ShareableLayers(common_branch=[], cross_domain_branch=[])
        self.share_with = share_with
        self.n_layers_to_share = n_layers_to_share
        super().__init__(*args, **kwargs)

    build_D = ALIforSharedExp.__dict__['build_D']

    def build_Gz(self):
        x_input = Input(shape=self.input_shape)

        # save all shareable layers
        self.s_layers[0].append(BasicConvLayer(64, (5, 5), strides=(1, 1), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.1))
        self.s_layers[0].append(ResLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1))

        self.s_layers[1].append(ResLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1))
        self.s_layers[1].append(ResLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1))
        self.s_layers[1].append(BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1))
        self.s_layers[1].append(BasicConvLayer(64, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1))

        # if there is a reference model (pre-defined layer weights)
        if self.share_with is not None and self.n_layers_to_share != (0, 0):
            self.s_layers[0][-self.n_layers_to_share[0]:] = self.share_with.s_layers[0][-self.n_layers_to_share[0]:]
            self.s_layers[1][-self.n_layers_to_share[1]:] = self.share_with.s_layers[1][-self.n_layers_to_share[1]:]

        # apply all layers to input
        x = x_input
        for layer in self.s_layers[0]:
            x = layer(x)
        x_d = self.s_layers[1][0](x)
        for layer in self.s_layers[1][1:]:
            x_d = layer(x_d)

        x_g = ResLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x_g = ResLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x_g)
        x_g = BasicConvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x_g)
        x_d = BasicConvLayer(64, (1, 1), strides=(1, 1), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.1)(x_d)

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

        x = Dense(512)(z_input)
        x = LeakyReLU(0.1)(x)
        x = Dense(512)(x)
        x = LeakyReLU(0.1)(x)

        x = Reshape((4, 4, 32))(x)

        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = ResDeconvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = ResDeconvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = ResDeconvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(64, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)

        x = BasicConvLayer(32, (1, 1), strides=(1, 1), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(orig_channels, (1, 1), activation='sigmoid', bnorm=False)(x)

        return Model(z_input, x)

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr, clipnorm=5.)
        opt_g = Adam(lr=self.lr, clipnorm=5.)
        return opt_d, opt_g

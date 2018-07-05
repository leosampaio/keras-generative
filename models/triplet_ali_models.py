import keras.backend as K
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU,
                          Lambda)
from keras.optimizers import RMSprop

from models.triplet_ali import TripletALI
from models.layers import BasicConvLayer, BasicDeconvLayer


class SVHN_MNIST_TripletALI(TripletALI):

    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'ali_for_svhn'
        super().__init__(*args, **kwargs)

    def build_Gz(self):
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x_input)
        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(128, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(256, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(512, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
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

        return Model(x_input, output, name="Gz")

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
        conditional_input = Input(shape=(self.conditional_dims,))
        orig_channels = self.input_shape[2]

        x = Concatenate()([z_input, conditional_input])
        x = Reshape((1, 1, -1))(x)

        x = BasicDeconvLayer(256, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(128, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(64, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(32, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)

        x = BasicConvLayer(32, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(orig_channels, (1, 1), activation='sigmoid', bnorm=False)(x)

        return Model([z_input, conditional_input], x, name="Gx")

    def build_D(self):
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_input)
        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(128, (4, 4), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(256, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(512, (4, 4), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = Flatten()(x)

        z_input = Input(shape=(self.z_dims,))
        conditional_input = Input(shape=(self.conditional_dims,))
        z = Concatenate()([z_input, conditional_input])
        z = Reshape((1, 1, -1))(z)
        z = BasicConvLayer(512, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
        z = BasicConvLayer(512, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
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

        return Model([x_input, z_input, conditional_input], xz, name="Discriminator")

    def build_optmizers(self):
        opt_d = RMSprop(lr=1e-4)
        opt_g = RMSprop(lr=1e-4)
        return opt_d, opt_g

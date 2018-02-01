from keras import Input, Model
from keras.applications import MobileNet
from keras.layers import Flatten, Dense, Activation, Reshape, BatchNormalization, Concatenate, Dropout, LeakyReLU, \
    LocallyConnected2D
from keras.optimizers import Adam

from models import ALI
from models.ali import DiscriminatorLossLayer, discriminator_accuracy, generator_accuracy, GeneratorLossLayer
from models.layers import BasicConvLayer, BasicDeconvLayer, SampleNormal
from models.utils import set_trainable, zero_loss


class WiderALI(ALI):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'wider_ali'
        super().__init__(*args, **kwargs)

    def build_D(self):
        x_inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=128 * 2, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x_inputs)
        x = BasicConvLayer(filters=256 * 2, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=256 * 2, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=512 * 2, kernel_size=(3, 3), bnorm=True)(x)
        x = Flatten()(x)

        z_inputs = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_inputs)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = Flatten()(z)

        xz = Concatenate(axis=-1)([x, z])
        xz = Dropout(0.2)(xz)
        xz = Dense(2048)(xz)
        xz = LeakyReLU(0.1)(xz)

        xz = Dropout(0.2)(xz)
        xz = Dense(1)(xz)
        xz = Activation('sigmoid')(xz)

        return Model([x_inputs, z_inputs], xz)


class DeeperALI(ALI):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'deeper_ali'
        super().__init__(*args, **kwargs)

    def build_D(self):
        x_inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x_inputs)
        x = BasicConvLayer(filters=128, kernel_size=(7, 7), strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=256, kernel_size=(7, 7), strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=512, kernel_size=(4, 4), strides=(1, 1), bnorm=True)(x)
        x = BasicConvLayer(filters=512, kernel_size=(4, 4), strides=(8, 8), bnorm=True)(x)
        x = BasicConvLayer(filters=512, kernel_size=(1, 1), strides=(1, 1), bnorm=True)(x)
        x = Flatten()(x)

        z_inputs = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_inputs)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = Flatten()(z)

        xz = Concatenate(axis=-1)([x, z])
        xz = Dropout(0.2)(xz)
        xz = Dense(2048)(xz)
        xz = LeakyReLU(0.1)(xz)

        xz = Dropout(0.2)(xz)
        xz = Dense(1)(xz)
        xz = Activation('sigmoid')(xz)

        return Model([x_inputs, z_inputs], xz)


class LocallyConnALI(ALI):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'local_conn_ali'
        super().__init__(*args, **kwargs)

    def build_D(self):
        x_inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=128, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x_inputs)
        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=256, kernel_size=(5, 5), strides=(2, 2), bnorm=True)(x)
        x = BasicConvLayer(filters=512, kernel_size=(3, 3), bnorm=True)(x)
        x = LocallyConnected2D(filters=64, kernel_size=(3, 3))(x)
        x = Flatten()(x)

        z_inputs = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_inputs)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = Flatten()(z)

        xz = Concatenate(axis=-1)([x, z])
        xz = Dropout(0.2)(xz)
        xz = Dense(2048)(xz)
        xz = LeakyReLU(0.1)(xz)

        xz = Dropout(0.2)(xz)
        xz = Dense(1)(xz)
        xz = Activation('sigmoid')(xz)

        return Model([x_inputs, z_inputs], xz)


class MobileNetALI(ALI):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'mobile_net_ali'
        super().__init__(*args, **kwargs)

    def build_D(self):
        mobile_net = MobileNet(input_shape=self.input_shape, weights=None, include_top=False)
        x = Flatten()(mobile_net.output)

        z_inputs = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_inputs)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = BasicConvLayer(filters=1024, kernel_size=(1, 1), dropout=0.2)(z)
        z = Flatten()(z)

        xz = Concatenate(axis=-1)([x, z])
        xz = Dropout(0.2)(xz)
        xz = Dense(2048)(xz)
        xz = LeakyReLU(0.1)(xz)

        xz = Dropout(0.2)(xz)
        xz = Dense(1)(xz)
        xz = Activation('sigmoid')(xz)

        return Model([mobile_net.input, z_inputs], xz)


class ALIforSVHN(ALI):
    """
    Based on the original ALI paper arch. Experiment on SVHN.
    See Table 4 on the paper for details
    """
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'ali_for_svhn'
        super().__init__(*args, **kwargs)

    def build_Gz(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=32, kernel_size=(5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(inputs)
        x = BasicConvLayer(filters=64, kernel_size=(4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(filters=128, kernel_size=(4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(filters=256, kernel_size=(4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(filters=512, kernel_size=(4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(filters=512, kernel_size=(1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)

        x = Flatten()(x)

        # the output is an average (mu) and std variation (sigma) 
        # describing the distribution that better describes the input
        z_avg = Dense(self.z_dims)(x)
        z_log_var = Dense(self.z_dims)(x)
        z_avg = Activation('linear')(z_avg)
        z_log_var = Activation('linear')(z_log_var)

        return Model(inputs, [z_avg, z_log_var])

    def build_Gx(self):
        inputs = Input(shape=(self.z_dims,))
        w = self.input_shape[0] // (2 ** 2)
        x = Dense(w * w * 512)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 512))(x)

        x = BasicDeconvLayer(filters=256, kernel_size=(4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(filters=128, kernel_size=(4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(filters=64, kernel_size=(4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(filters=32, kernel_size=(4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(filters=32, kernel_size=(5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(filters=32, kernel_size=(1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)

        d = self.input_shape[2]
        x = BasicConvLayer(filters=d, kernel_size=(1, 1), activation='sigmoid', bnorm=False)(x)

        return Model(inputs, x)

    def build_D(self):
        x_inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=32, kernel_size=(5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_inputs)
        x = BasicConvLayer(filters=64, kernel_size=(4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(filters=128, kernel_size=(4, 4), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(filters=256, kernel_size=(4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(filters=512, kernel_size=(4, 4), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = Flatten()(x)

        z_inputs = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_inputs)
        z = BasicConvLayer(filters=512, kernel_size=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
        z = BasicConvLayer(filters=512, kernel_size=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
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

        return Model([x_inputs, z_inputs], xz)
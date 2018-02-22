from keras import Input, Model
from keras.applications import MobileNet
from keras.layers import Flatten, Dense, Activation, Reshape, BatchNormalization, Concatenate, Dropout, LeakyReLU, \
    LocallyConnected2D
from keras.optimizers import Adam

from models import ALI
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
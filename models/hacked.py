from keras.models import Model
from keras.layers import Dense, BatchNormalization, Activation, Reshape, Dropout, Flatten, Input

from models import DCGAN, VAE, ImprovedGAN, ALI
from models.layers import BasicDeconvLayer, BasicConvLayer, MinibatchDiscrimination


class DropoutDcgan(DCGAN):
    """
    see https://github.com/soumith/ganhacks
     - added dropout (pt. 17 from ^)
    """

    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'drdcgan'
        super().__init__(*args, **kwargs)

    def build_decoder(self):
        inputs = Input(shape=(self.z_dims,))
        w = self.input_shape[0] // (2 ** 3)
        x = Dense(w * w * 256)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 256))(x)

        # x = BasicDeconvLayer(filters=256, strides=(2, 2), activation='relu')(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=64, strides=(2, 2), activation='relu')(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model(inputs, x)


class DropoutVae(VAE):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'drvae'
        super().__init__(*args, **kwargs)

    def build_decoder(self):
        inputs = Input(shape=(self.z_dims,))
        w = self.input_shape[0] // (2 ** 3)
        x = Dense(w * w * 256)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 256))(x)

        # x = BasicDeconvLayer(filters=512, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2))(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=64, strides=(2, 2))(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model(inputs, x)


class DropoutImprovedGAN(ImprovedGAN):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'drimprovedgan'
        super().__init__(*args, **kwargs)

    def build_generator(self):
        inputs = Input(shape=(self.z_dims,))
        w = self.input_shape[0] // (2 ** 3)
        x = Dense(w * w * 256)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 256))(x)

        # x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2))(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=64, strides=(2, 2))(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model(inputs, x)


class DropoutALI(ALI):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'drali'
        super().__init__(*args, **kwargs)

    def build_generator(self):
        inputs = Input(shape=(self.z_dims,))
        w = self.input_shape[0] // (2 ** 3)
        x = Dense(w * w * 256)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 256))(x)

        # x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2))(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=64, strides=(2, 2))(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model(inputs, x)


class VeryDcgan(DCGAN):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'vdcgan'
        super().__init__(*args, **kwargs)

    def build_decoder(self):
        inputs = Input(shape=(self.z_dims,))
        w = self.input_shape[0] // (2 ** 3)
        x = Dense(w * w * 256)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 256))(x)

        # x = BasicDeconvLayer(filters=512, strides=2, activation='relu')(x)
        # x = BasicDeconvLayer(filters=512, strides=2, activation='relu')(x)
        # x = BasicDeconvLayer(filters=256, strides=2, activation='relu')(x)
        x = BasicDeconvLayer(filters=256, strides=1, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=256, strides=1, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2), activation='relu')(x)
        x = BasicDeconvLayer(filters=64, strides=(2, 2), activation='relu')(x)
        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model(inputs, x)

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=64, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=128, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(1024)(x)
        x = Activation('relu')(x)

        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(inputs, x)


class VeryDeepImprovedGAN(ImprovedGAN):
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'vdimprovedgan'
        super().__init__(*args, **kwargs)

    def build_discriminator(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(filters=64, strides=(2, 2))(inputs)
        x = BasicConvLayer(filters=128, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=256, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)
        x = BasicConvLayer(filters=512, strides=(2, 2))(x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        f = Activation('relu')(x)

        x = MinibatchDiscrimination(kernels=50, dims=5)(f)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(inputs, [x, f])

    def build_generator(self):
        inputs = Input(shape=(self.z_dims,))
        w = self.input_shape[0] // (2 ** 3)
        x = Dense(w * w * 256)(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((w, w, 256))(x)

        x = BasicDeconvLayer(filters=256, strides=1, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=256, strides=1, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2), activation='relu')(x)
        x = Dropout(0.5)(x)
        x = BasicDeconvLayer(filters=128, strides=(2, 2), activation='relu')(x)
        x = BasicDeconvLayer(filters=64, strides=(2, 2), activation='relu')(x)

        d = self.input_shape[2]
        x = BasicDeconvLayer(filters=d, strides=(1, 1), bnorm=False, activation='tanh')(x)

        return Model(inputs, x)

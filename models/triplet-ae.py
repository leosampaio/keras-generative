from pprint import pprint

import numpy as np

from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Add,
                          Lambda)
from keras.optimizers import Adam
from keras import backend as K

from core.models import BaseModel

from .layers import conv2d, deconv2d, res


class ContrastiveAEwithBEGAN(BaseModel):
    name = 'contrastive-ae-began'
    loss_names = ['contrastive_ae_p', 'contrastive_ae_n']
    loss_plot_organization = [('contrastive_ae_p', 'contrastive_ae_n')]

    def __init__(self,
                 input_shape=(64, 64, 3),
                 embedding_dim=256,
                 n_filters_factor=32,
                 **kwargs):
        super().__init__(input_shape=input_shape, **kwargs)

        self.embedding_size = embedding_dim
        self.n_filters_factor = n_filters_factor

        pprint(vars(self))

        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        if self.input_noise > 1e-5:
            noise = np.random.normal(scale=self.input_noise, size=x_data.shape)
        else:
            noise = np.zeros(x_data.shape)

        input_data = [x_data[:len(x_data)//2], noise[:len(x_data)//2]]
        label_data = [x_data[:len(x_data)//2], x_data[len(x_data)//2:]]

        # train both networks
        ld = {}  # loss dictionary
        _, ld['contrastive_ae_p'], ld['contrastive_ae_n'] = self.trainer.train_on_batch(input_data, label_data)

        return ld

    def build_trainer(self):
        input_x = Input(shape=self.input_shape)
        input_noise = Input(shape=self.input_shape)

        clipping_layer = Lambda(lambda x: K.clip(x, 0., 1.))
        x_noisy = clipping_layer(Add()([input_x, input_noise]))

        z = self.encoder(x_noisy)
        x_hat = self.decoder(z)

        input = [input_x, input_noise]

        return Model(input, [x_hat, x_hat], name='contrastive-ae')

    def build_model(self):

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        self.optimizers = self.build_optmizers()

        # build discriminator
        self.trainer = self.build_trainer()
        self.trainer.compile(optimizer=self.optimizers["opt_ae"],
                             loss=['mse', 'mse'],
                             loss_weights=[self.losses['contrastive_ae_p'].backend,
                                           -self.losses['contrastive_ae_n'].backend])

        # store trainers
        self.store_to_save('trainer')

    def build_optmizers(self):
        return {"opt_ae": Adam(lr=self.lr, beta_1=0.5)}

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(inputs)
        x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)
        x = conv2d(self.n_filters_factor * 2, (3, 3), strides=(2, 2), activation='elu')(x)

        x = conv2d(self.n_filters_factor * 2, (3, 3), activation='elu')(x)
        x = conv2d(self.n_filters_factor * 3, (3, 3), strides=(2, 2), activation='elu')(x)

        x = conv2d(self.n_filters_factor * 3, (3, 3), activation='elu')(x)

        if self.input_shape[0] == 32:
            x = conv2d(self.n_filters_factor * 3, (3, 3), activation='elu')(x)
        elif self.input_shape[0] >= 64:
            x = conv2d(self.n_filters_factor * 4, (3, 3), strides=(2, 2), activation='elu')(x)
            x = conv2d(self.n_filters_factor * 4, (3, 3), activation='elu')(x)
            x = conv2d(self.n_filters_factor * 4, (3, 3), activation='elu')(x)

        x = Flatten()(x)

        x = Dense(self.embedding_size)(x)
        x = Activation('linear')(x)

        return Model(inputs, x)

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))
        orig_channels = self.input_shape[2]

        x = Dense(self.n_filters_factor * 8 * 8)(z_input)
        x = Reshape((8, 8, self.n_filters_factor))(x)

        x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)
        x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)
        x = deconv2d(self.n_filters_factor * 2, (3, 3), strides=(2, 2), activation='elu', padding='same')(x)
        x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)
        x = deconv2d(self.n_filters_factor * 2, (3, 3), strides=(2, 2), activation='elu', padding='same')(x)
        x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)

        if self.input_shape[0] >= 64:
            x = deconv2d(self.n_filters_factor * 2, (3, 3), strides=(2, 2), activation='elu', padding='same')(x)
            x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)

        x = conv2d(orig_channels, (3, 3), activation='sigmoid')(x)

        return Model(z_input, x)

    """
        Define computation of metrics inputs
    """

    def compute_labelled_embedding(self, n=10000):
        x_data, y_labels = self.dataset.get_random_fixed_batch(n)
        x_feats = self.encoder.predict(x_data, batch_size=self.batchsize)
        if self.dataset.has_test_set():
            x_test, y_test = self.dataset.get_random_perm_of_test_set(n=1000)
            x_test_feats = self.encoder.predict(x_test, batch_size=self.batchsize)
            self.save_precomputed_features('labelled_embedding', x_feats, Y=y_labels,
                                           test_set=(x_test_feats, y_test))
        else:
            self.save_precomputed_features('labelled_embedding', x_feats, Y=y_labels)
        return x_feats, y_labels

    def compute_generated_and_real_samples(self, n=10000):
        np.random.seed(14)
        samples = np.random.normal(size=(n, self.embedding_size))
        np.random.seed()

        generated_images = self.decoder.predict(samples, batch_size=self.batchsize)
        images_from_set, _ = self.dataset.get_random_fixed_batch(n)

        self.save_precomputed_features('generated_and_real_samples', generated_images, Y=images_from_set)
        return images_from_set, generated_images

    def compute_generated_image_samples(self, n=36):
        np.random.seed(14)
        samples = np.random.normal(size=(n, self.embedding_size))
        np.random.seed()

        generated_images = self.decoder.predict(samples, batch_size=n)
        return generated_images

    def compute_reconstruction_samples(self, n=18):
        imgs_from_dataset, _ = self.dataset.get_random_perm_of_test_set(n)
        np.random.seed(14)
        noise = np.random.normal(scale=self.input_noise, size=imgs_from_dataset.shape)
        np.random.seed()
        imgs_from_dataset += noise
        encoding = self.encoder.predict(imgs_from_dataset)
        x_hat = self.decoder.predict(encoding)
        return imgs_from_dataset, x_hat


class ContrastiveAESmall(ContrastiveAEwithBEGAN):
    name = 'contrastive-ae-small'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = conv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(inputs)
        x = Flatten()(x)
        x = Dense(self.embedding_size)(x)
        x = Activation('linear')(x)

        return Model(inputs, x)

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))
        orig_channels = self.input_shape[2]

        w = self.input_shape[0] // 2**1  # starting width
        x = Dense(64 * w * w)(z_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape((w, w, 64))(x)

        x = deconv2d(orig_channels, (4, 4), strides=(2, 2), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)


class ContrastiveAEfromDCGAN(ContrastiveAEwithBEGAN):
    name = 'contrastive-ae-dcgan'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = conv2d(self.n_filters_factor, (5, 5), strides=(2, 2), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(inputs)
        x = conv2d(self.n_filters_factor * 2, (5, 5), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(inputs)
        x = conv2d(self.n_filters_factor * 4, (5, 5), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(inputs)
        x = conv2d(self.n_filters_factor * 8, (5, 5), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(inputs)

        x = Flatten()(x)
        x = Dense(self.embedding_size)(x)

        return Model(inputs, x)

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))
        orig_channels = self.input_shape[2]

        w = self.input_shape[0] // 2**4

        x = Dense(self.n_filters_factor * 8 * w * w)(z_input)
        x = Reshape((w, w, self.n_filters_factor * 8))(x)
        x = BatchNormalization()(x)
        x = deconv2d(self.n_filters_factor * 4, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = deconv2d(self.n_filters_factor * 2, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = deconv2d(self.n_filters_factor, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)

        x = deconv2d(orig_channels, (5, 5), strides=(2, 2), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)

class ContrastiveAEwithRes(ContrastiveAEwithBEGAN):
    name = 'contrastive-ae-res'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = conv2d(self.n_filters_factor, (5, 5), strides=(2, 2), activation='relu', padding='same')(inputs)
        x = conv2d(self.n_filters_factor * 2, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = res(self.n_filters_factor * 4, (3, 3), activation='relu', padding='same')(x)
        x = res(self.n_filters_factor * 4, (3, 3), activation='relu', padding='same')(x)
        x = res(self.n_filters_factor * 4, (3, 3), activation='relu', padding='same')(x)
        x = res(self.n_filters_factor * 4, (3, 3), activation='relu', padding='same')(x)
        x = conv2d(self.n_filters_factor * 4, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)

        x = Flatten()(x)
        x = Dense(self.embedding_size)(x)

        return Model(inputs, x)

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))
        orig_channels = self.input_shape[2]

        w = self.input_shape[0] // 2**4

        x = Dense(self.n_filters_factor * 8 * w * w)(z_input)
        x = Reshape((w, w, self.n_filters_factor * 8))(x)
        x = BatchNormalization()(x)
        x = deconv2d(self.n_filters_factor * 4, (5, 5), strides=(2, 2), activation='relu', padding='same')(x)
        x = res(self.n_filters_factor * 4, (3, 3), activation='relu', padding='same')(x)
        x = res(self.n_filters_factor * 4, (3, 3), activation='relu', padding='same')(x)
        x = res(self.n_filters_factor * 4, (3, 3), activation='relu', padding='same')(x)
        x = res(self.n_filters_factor * 4, (3, 3), activation='relu', padding='same')(x)
        x = deconv2d(self.n_filters_factor * 2, (5, 5), strides=(2, 2), activation='relu', padding='same')(x)
        x = deconv2d(self.n_filters_factor, (5, 5), strides=(2, 2), activation='relu', padding='same')(x)

        x = deconv2d(orig_channels, (5, 5), strides=(2, 2), activation='sigmoid', padding='same')(x)

        return Model(z_input, x)

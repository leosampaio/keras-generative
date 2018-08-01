from pprint import pprint
from functools import partial

import numpy as np

from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU,
                          LocallyConnected2D, Add,
                          Lambda, AveragePooling1D, GlobalAveragePooling2D)
from keras.optimizers import Adam
from keras import backend as K

from core.models import BaseModel
from core.lossfuns import (began_gen_lossfun, began_dis_lossfun,
                           began_convergence_lossfun)

from .utils import set_trainable
from .layers import conv2d, deconv2d


class BEGAN(BaseModel):
    name = 'began-small'
    loss_names = ['g_loss', 'd_loss', 'gd_ratio', 'convergence_measure']
    loss_plot_organization = [('g_loss', 'd_loss', 'gd_ratio'), 'convergence_measure']

    def __init__(self,
                 input_shape=(64, 64, 3),
                 embedding_dim=256,
                 began_gamma=0.5,
                 n_filters_factor=32,
                 **kwargs):
        super().__init__(input_shape=input_shape, **kwargs)

        self.embedding_size = embedding_dim
        self.gamma = began_gamma
        self.n_filters_factor = n_filters_factor

        pprint(vars(self))

        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        batchsize = len(x_data)

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        input_data = [x_data, z_latent_dis]
        label_data = [x_data, x_data, x_data]

        # train both networks
        ld = {}  # loss dictionary
        _, ld['d_loss'], _, _ = self.dis_trainer.train_on_batch(input_data, label_data)
        _, ld['g_loss'], ld['ae_loss'], ld['convergence_measure'] = self.gen_trainer.train_on_batch(input_data, label_data)

        # update k
        ld['gd_ratio'] = K.get_value(self.k_gd_ratio) + self.lr * (self.gamma * ld['ae_loss'] - ld['g_loss'])
        K.set_value(self.k_gd_ratio, ld['gd_ratio'])
        ld['gd_ratio'] *= batchsize  # fix plotted value because losses are divided

        return ld

    def build_trainer(self):
        input_x = Input(shape=self.input_shape)
        input_z = Input(shape=(self.z_dims,))

        x_hat = self.f_Gx(input_z)

        x_hat_reconstructed = self.f_D(x_hat)
        x_reconstructed = self.f_D(input_x)

        input = [input_x, input_z]

        concatenated_dis = Concatenate(axis=-1, name="ae")([x_hat, x_hat_reconstructed, x_reconstructed])
        output = [concatenated_dis, x_hat, concatenated_dis]
        return Model(input, output, name='began')

    def build_model(self):

        self.f_Gx = self.build_Gx()  # Moriarty, the encoder
        self.f_D = self.build_D()   # Sherlock, the detective
        self.f_Gx.summary()
        self.f_D.summary()
        self.encoder.summary()
        self.decoder.summary()

        self.optimizers = self.build_optmizers()
        self.k_gd_ratio = K.variable(0)  # initialize k
        dis_lossfun = partial(began_dis_lossfun, k_gd_ratio=self.k_gd_ratio)
        dis_lossfun.__name__ = 'began_dis_loss'

        convergence_lossfun = partial(began_convergence_lossfun, gamma=self.gamma)
        convergence_lossfun.__name__ = 'convergence_lossfun'

        # build discriminator
        self.dis_trainer = self.build_trainer()
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, True)
        self.dis_trainer.compile(optimizer=self.optimizers["opt_d"],
                                 loss=[dis_lossfun, 'mae', convergence_lossfun],
                                 loss_weights=[1., 0., 0.])

        # build generator
        self.gen_trainer = self.build_trainer()
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=self.optimizers["opt_g"],
                                 loss=[began_gen_lossfun, 'mae', convergence_lossfun],
                                 loss_weights=[1., 0., 0.])

        # store trainers
        self.store_to_save('gen_trainer')
        self.store_to_save('dis_trainer')

    def build_optmizers(self):
        return {"opt_d": Adam(lr=self.lr, beta_1=0.5),
                "opt_g": Adam(lr=self.lr, beta_1=0.5)}

    def save_model(self, out_dir, epoch):
        self.trainers['f_Gx'] = self.f_Gx
        self.trainers['f_D'] = self.f_D
        super().save_model(out_dir, epoch)

    def build_D(self):
        x_input = Input(shape=self.input_shape)

        self.encoder = self.build_encoder()
        x_embedding = self.encoder(x_input)

        self.decoder = self.build_decoder()
        x_hat = self.decoder(x_embedding)

        return Model(x_input, x_hat)

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

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
        orig_channels = self.input_shape[2]

        w = self.input_shape[0] // 2**2  # starting width
        x = Dense(1024)(z_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(128 * w * w)(z_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Reshape((w, w, 128))(x)

        x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = deconv2d(orig_channels, (4, 4), strides=(2, 2), bnorm=False, activation='sigmoid', padding='same')(x)

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
        samples = np.random.normal(size=(n, self.z_dims))
        perm = np.random.permutation(len(self.dataset))
        np.random.seed()

        generated_images = self.f_Gx.predict(samples, batch_size=self.batchsize)
        images_from_set = self.dataset.images[perm[:n]]

        self.save_precomputed_features('generated_and_real_samples', generated_images, Y=images_from_set)
        return images_from_set, generated_images

    def compute_generated_image_samples(self, n=36):
        np.random.seed(14)
        samples = np.random.normal(size=(n, self.z_dims))
        np.random.seed()

        generated_images = self.f_Gx.predict(samples, batch_size=self.batchsize)
        return generated_images

    def compute_reconstruction_samples(self, n=18):
        imgs_from_dataset, _ = self.dataset.get_random_fixed_batch(n)
        encoding = self.encoder.predict(imgs_from_dataset)
        x_hat = self.decoder.predict(encoding)
        return imgs_from_dataset, x_hat


class BEGANwithDCGAN(BEGAN):
    name = 'began-dcgan'

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

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
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


class BEGANwithBEGAN(BEGAN):
    name = 'began-began'

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

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
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

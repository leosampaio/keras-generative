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

from .utils import *
from .layers import *


class BEGAN(BaseModel):
    name = 'began'
    loss_names = ['g_loss', 'd_loss', 'gd_ratio', 'convergence_measure']
    loss_plot_organization = [('g_loss', 'd_loss', 'gd_ratio'), 'convergence_measure']

    def __init__(self,
                 input_shape=(64, 64, 3),
                 embedding_dim=256,
                 began_gamma=0.5,
                 **kwargs):
        super().__init__(input_shape=input_shape, **kwargs)

        self.embedding_size = embedding_dim
        self.gamma = began_gamma

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
        return {"opt_d": Adam(lr=self.lr),
                "opt_g": Adam(lr=self.lr)}

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

        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(inputs)
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

        x = BasicDeconvLayer(orig_channels, (4, 4), strides=(2, 2), bnorm=False, activation='sigmoid', padding='same')(x)

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

        x = BasicDeconvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = BasicDeconvLayer(orig_channels, (4, 4), strides=(2, 2), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)

    """
        Define computation of metrics inputs
    """

    def compute_labelled_embedding(self, n=10000):
        np.random.seed(14)
        perm = np.random.permutation(len(self.dataset))
        x_data, y_labels = self.dataset.images[perm[:10000]], np.argmax(self.dataset.attrs[perm[:10000]], axis=1)
        np.random.seed()
        x_feats = self.encoder.predict(x_data, batch_size=2000)
        self.save_precomputed_features('labelled_embedding', x_feats, Y=y_labels)
        return x_feats, y_labels

    def compute_generated_and_real_samples(self, n=10000):
        np.random.seed(14)
        samples = np.random.normal(size=(n, self.z_dims))
        perm = np.random.permutation(len(self.dataset))
        np.random.seed()

        generated_images = self.f_Gx.predict(samples, batch_size=2000)
        images_from_set = self.dataset.images[perm[:n]]

        self.save_precomputed_features('generated_and_real_samples', generated_images, Y=images_from_set)
        return images_from_set, generated_images

    def compute_generated_image_samples(self, n=36):
        np.random.seed(14)
        samples = np.random.normal(size=(n, self.z_dims))
        np.random.seed()

        generated_images = self.f_Gx.predict(samples, batch_size=n)
        return generated_images

    def compute_reconstruction_samples(self, n=18):
        np.random.seed(14)
        perm = np.random.permutation(len(self.dataset))
        imgs_from_dataset = self.dataset.images[perm[:n]]
        noise = np.random.normal(scale=self.input_noise, size=imgs_from_dataset.shape)
        imgs_from_dataset += noise
        np.random.seed()
        encoding = self.encoder.predict(imgs_from_dataset)
        x_hat = self.decoder.predict(encoding)
        return imgs_from_dataset, x_hat

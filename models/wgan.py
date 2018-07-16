from pprint import pprint

import numpy as np

from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU,
                          LocallyConnected2D, Add,
                          Lambda, AveragePooling1D, GlobalAveragePooling2D)
from keras.optimizers import Adam, RMSprop
from keras.constraints import Constraint
from keras import backend as K

from core.models import BaseModel
from core.lossfuns import wasserstein_dis_lossfun, wasserstein_gen_lossfun

from .utils import *
from .layers import *


class WeightClip(Constraint):
    '''Clips the weights incident to each hidden unit to be inside a range
    '''

    def __init__(self, c=2):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': 'weight-clip',
                'c': self.c}


class WGAN(BaseModel):
    name = 'wgan'
    loss_names = ['g_loss', 'd_loss']
    loss_plot_organization = [('g_loss', 'd_loss')]

    def __init__(self,
                 input_shape=(64, 64, 3),
                 triplet_margin=1.,
                 wgan_n_critic=5,
                 **kwargs):
        super().__init__(input_shape=input_shape, **kwargs)

        self.n_critic = wgan_n_critic

        pprint(vars(self))
        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        batchsize = len(x_data)
        y_pos, y_neg = smooth_binary_labels(batchsize, self.label_smoothing, one_sided_smoothing=False)
        y = np.stack((y_neg, y_pos), axis=1)

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        input_data = [x_data, z_latent_dis]
        label_data = [y]

        ld = {}
        for _ in range(self.n_critic):
            ld['d_loss'] = self.dis_trainer.train_on_batch(input_data, label_data)
        ld['g_loss'] = self.gen_trainer.train_on_batch(input_data, label_data)

        return ld

    def build_trainer(self):
        input_z = Input(shape=(self.z_dims,))
        input_x = Input(shape=self.input_shape)
        p = self.f_D(self.f_Gx(input_z))
        q = self.f_D(input_x)
        concatenated_dis = Concatenate(axis=-1, name="dis_classification")([p, q])
        return Model([input_x, input_z], concatenated_dis, name='wgan')

    def build_model(self):

        self.f_Gx = self.build_Gx()
        self.f_D = self.build_D()

        self.optimizers = self.build_optmizers()

        # build discriminator
        self.dis_trainer = self.build_trainer()
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, True)
        self.dis_trainer.compile(optimizer=self.optimizers["opt_d"],
                                 loss=wasserstein_dis_lossfun)

        # build generators
        self.gen_trainer = self.build_trainer()
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=self.optimizers["opt_g"],
                                 loss=wasserstein_gen_lossfun)

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

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', k_constraint=WeightClip(0.01))(inputs)
        x = Flatten()(x)
        x = Dense(128, kernel_constraint=WeightClip(0.01))(x)
        x = Activation('linear')(x)

        return Model(inputs, x)

    def build_D(self):
        """
        Network Architecture based on the one presented in infoGAN
        """
        x_input = Input(shape=self.input_shape)

        self.encoder = self.build_encoder()
        x_embedding = self.encoder(x_input)

        self.d_classifier = self.build_d_classifier()
        discriminator_clas = self.d_classifier(x_embedding)

        return Model(x_input, discriminator_clas)

    def build_d_classifier(self):
        embedding_input = Input(shape=(128,))

        x = Activation('relu')(embedding_input)
        x = BatchNormalization()(x)
        x = Dense(64, kernel_constraint=WeightClip(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(64, kernel_constraint=WeightClip(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(1, kernel_constraint=WeightClip(0.01))(x)
        x = Activation('linear')(x)

        return Model(embedding_input, x)

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

from pprint import pprint

import numpy as np
import sklearn as sk
import sklearn.cluster as skcluster
import pandas as pd
from scipy.spatial.distance import cdist

import keras
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Add,
                          Lambda, Conv1D, UpSampling2D, Subtract)
from keras.optimizers import Adam, SGD
from keras import backend as K

from core.lossfuns import loss_is_output, MSELayer
from core.models import BaseModel

from .layers import (conv2d, deconv2d, LayerNorm, rdeconv, res, dense)
from .utils import set_trainable


class DiverseAutoencoder(BaseModel):
    name = 'diversicoder'

    def __init__(self,
                 input_shape=(64, 64, 3),
                 embedding_dim=128,
                 n_filters_factor=128,
                 n_domains=2,
                 **kwargs):

        self.loss_names = ["ae_loss_a", "ae_loss_b", "auxloss_a",
                           "auxloss_b", "metaloss"]
        self.loss_plot_organization = self.loss_names

        super().__init__(input_shape=input_shape, **kwargs)

        self.embedding_size = embedding_dim
        self.n_domains = n_domains
        self.n_filters_factor = n_filters_factor
        self.encoders, self.decoders = {}, {}
        self.l_enc = {}
        self.l_dec = {}

        pprint(vars(self))
        self.build_model()

    def build_trainer(self, id):
        a_input = Input(shape=self.input_shape)
        b_input = Input(shape=self.input_shape)

        self.decoders[id], _ = self.build_decoder()

        x_embedding = self.encoders['a'](a_input)
        x_hat = self.decoders[id](x_embedding)

        aux_loss = self.metaloss_net(x_embedding)

        return Model([a_input, b_input], [x_hat, aux_loss], name="ae_{}".format(id))

    def build_metatrainer(self):
        a_input = Input(shape=self.input_shape)
        b_input = Input(shape=self.input_shape)
        a_val_input = Input(shape=self.input_shape)
        b_val_input = Input(shape=self.input_shape)

        x_embedding_a = self.encoders['a'](a_input)
        x_embedding_b = self.encoders['a'](b_input)
        x_hat_a = self.decoders['a'](x_embedding_a)
        x_hat_b = self.decoders['b'](x_embedding_b)

        mse_loss = Add()([MSELayer()([a_input, x_hat_a]), MSELayer()([b_input, x_hat_b])])
        aux_loss = Add()([self.metaloss_net(x_embedding_a), self.metaloss_net(x_embedding_b)])

        # create models with each update
        omega_encs = self.build_encoder_omega(mse_loss, aux_loss)
        # self.decoder_omega_old[id] = self.build_decoder_omega(id, mse_loss, aux_loss)
        self.encoder_omega_old, self.encoder_omega_new = omega_encs

        # use new models on the validation set
        meta_val_old_embedding_a = self.encoder_omega_old(a_val_input)
        meta_val_new_embedding_a = self.encoder_omega_new(a_val_input)
        meta_val_old_hat_a = self.decoders['a'](meta_val_old_embedding_a)
        meta_val_new_hat_a = self.decoders['a'](meta_val_new_embedding_a)
        meta_val_old_embedding_b = self.encoder_omega_old(b_val_input)
        meta_val_new_embedding_b = self.encoder_omega_new(b_val_input)
        meta_val_old_hat_b = self.decoders['b'](meta_val_old_embedding_b)
        meta_val_new_hat_b = self.decoders['b'](meta_val_new_embedding_b)

        mse_val_old = Add()([MSELayer()([a_val_input, meta_val_old_hat_a]),
                             MSELayer()([b_val_input, meta_val_old_hat_b])])
        mse_val_new = Add()([MSELayer()([a_val_input, meta_val_new_hat_a]),
                             MSELayer()([b_val_input, meta_val_new_hat_b])])
        metaloss = Activation('tanh')(Subtract()([mse_val_new, mse_val_old]))
        fix_dim = Reshape((-1,))

        return Model([a_input, b_input, a_val_input, b_val_input], [fix_dim(metaloss)], name="meta_{}".format(id))

    def build_model(self):

        self.metaloss_net = self.build_metaloss_net()
        self.encoders['a'], self.l_enc['a'] = self.build_encoder()
        self.trainer_a = self.build_trainer('a')
        self.trainer_b = self.build_trainer('b')
        self.optimizers = self.build_optmizers()

        self.metatrainer = self.build_metatrainer()

        # compile autoencoders
        self.trainer_a.compile(
            optimizer=self.optimizers['a'],
            loss=['mse', loss_is_output],
            loss_weights=[1., 1.])
        self.trainer_b.compile(
            optimizer=self.optimizers['b'],
            loss=['mse', loss_is_output],
            loss_weights=[1., 1.])

        set_trainable(self.trainer_a, False)
        set_trainable(self.trainer_b, False)
        set_trainable(self.metaloss_net, True)
        self.metatrainer.compile(
            optimizer=self.optimizers['meta_a'],
            loss=[loss_is_output])

    def build_optmizers(self):
        ae_a = Adam(lr=self.lr, beta_1=0.5)
        ae_b = Adam(lr=self.lr, beta_1=0.5)
        meta_a = Adam(lr=self.lr, beta_1=0.5)
        meta_b = Adam(lr=self.lr, beta_1=0.5)
        return {"a": ae_a,
                "b": ae_b,
                "meta_a": meta_a,
                "meta_b": meta_b}

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        x_a, x_b, x_a_val, x_b_val = x_data
        y_a, y_b, y_a_val, y_b_val = y_batch
        dummy_y = np.zeros((len(x_a), 1))
        dummy_yy = np.zeros((len(x_a),))

        ld = {}
        _, ld['ae_loss_a'], ld['auxloss_a'] = self.trainer_a.train_on_batch([x_a, x_b], [x_a, dummy_y])
        _, ld['ae_loss_b'], ld['auxloss_b'] = self.trainer_b.train_on_batch([x_b, x_a], [x_b, dummy_y])
        ld['metaloss'] = self.metatrainer.train_on_batch([x_a, x_b, x_a_val, x_b_val], [dummy_yy])

        return ld

    """
        # Network Layers' Definition
    """

    def build_encoder_omega(self, loss, auxloss):

        inputs = Input(shape=self.input_shape)

        x = {}
        for layer in self.l_enc['a']:
            lr = self.optimizers['a'].lr
            try:
                k_grads_l, b_grads_l = layer.get_gradients(loss)
                k_grads_aux, b_grads_aux = layer.get_gradients(auxloss)
                print(k_grads_aux)
                layer.add_to_kernel(-lr * k_grads_l)
                layer.add_to_bias(-lr * b_grads_l)
            except AttributeError:
                pass
            x['old'] = layer(x.get('old', inputs))
            try:
                if k_grads_aux is not None:
                    layer.add_to_kernel(-lr * k_grads_aux)
                    layer.add_to_bias(-lr * b_grads_aux)
            except AttributeError:
                pass
            x['new'] = layer(x.get('new', inputs))

        return Model(inputs, x['old']), Model(inputs, x['new'])

    def build_decoder_omega(self, id, loss, auxloss):

        inputs = Input(shape=(self.embedding_size,))

        x = {}
        for layer in self.l_dec[id]:
            lr = self.optimizers[id].lr
            try:
                k_grads_l, b_grads_l = layer.get_gradients(loss)
                k_grads_aux, b_grads_aux = layer.get_gradients(auxloss)
                print(k_grads_aux)
                layer.add_to_kernel(-lr * k_grads_l)
                layer.add_to_bias(-lr * b_grads_l)
            except AttributeError:
                pass
            x['old'] = layer(x.get('old', inputs))

        return Model(inputs, x['old'])

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        ls = [conv2d(self.n_filters_factor, (3, 3), activation='elu'),
              conv2d(self.n_filters_factor, (3, 3), activation='elu'),
              conv2d(self.n_filters_factor * 2, (3, 3), strides=(2, 2), activation='elu'),
              conv2d(self.n_filters_factor * 2, (3, 3), activation='elu'),
              conv2d(self.n_filters_factor * 3, (3, 3), strides=(2, 2), activation='elu'),
              conv2d(self.n_filters_factor * 3, (3, 3), activation='elu')]

        if self.input_shape[0] == 32:
            ls.append(conv2d(self.n_filters_factor * 3, (3, 3), activation='elu'))
        elif self.input_shape[0] >= 64:
            ls += [conv2d(self.n_filters_factor * 4, (3, 3), strides=(2, 2), activation='elu'),
                   conv2d(self.n_filters_factor * 4, (3, 3), activation='elu'),
                   conv2d(self.n_filters_factor * 4, (3, 3), activation='elu'), ]

        ls += [Flatten(),
               dense(self.embedding_size, activation='linear')]

        x = inputs
        for l in ls:
            x = l(x)

        return Model(inputs, x), ls

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))
        orig_channels = self.input_shape[2]

        ls = [dense(self.n_filters_factor * 8 * 8, activation='linear'),
              Reshape((8, 8, self.n_filters_factor)),
              conv2d(self.n_filters_factor, (3, 3), activation='elu'),
              deconv2d(self.n_filters_factor, (3, 3), strides=(2, 2), activation='elu'),
              conv2d(self.n_filters_factor, (3, 3), activation='elu', padding='same'),
              deconv2d(self.n_filters_factor, (3, 3), strides=(2, 2), activation='elu'),
              conv2d(self.n_filters_factor, (3, 3), activation='elu', padding='same'),
              conv2d(self.n_filters_factor, (3, 3), activation='elu')]

        if self.input_shape[0] >= 64:
            ls += [deconv2d(self.n_filters_factor, (3, 3), strides=(2, 2), activation='elu'),
                   conv2d(self.n_filters_factor, (3, 3), activation='elu')]

        if self.input_shape[0] >= 128:
            ls += [deconv2d(self.n_filters_factor, (3, 3), strides=(2, 2), activation='elu'),
                   conv2d(self.n_filters_factor, (3, 3), activation='elu'), ]

        ls += [conv2d(orig_channels, (3, 3), activation='sigmoid')]

        x = z_input
        for l in ls:
            x = l(x)

        return Model(z_input, x), ls

    def build_metaloss_net(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = dense(128, activation='relu', bnorm=False)(embedding_input)
        x = dense(128, activation='relu', bnorm=False)(x)
        x = dense(1, activation='relu', bnorm=False)(x)

        return Model(embedding_input, x)

    """
        # Computation of metrics inputs
    """

    def compute_reconstruction_samples(self, n=18):

        (imgs_a, _), (imgs_b, _) = self.dataset.get_random_fixed_batch(n)
        enc_a = self.encoders['a'].predict(imgs_a)
        a_hat = self.decoders['a'].predict(enc_a)

        return imgs_a, a_hat

    def compute_reconstruction_samples2(self, n=18):

        (imgs_a, _), (imgs_b, _) = self.dataset.get_random_fixed_batch(n)
        enc_b = self.encoders['a'].predict(imgs_b)
        b_hat = self.decoders['b'].predict(enc_b)

        return imgs_b, b_hat

    def compute_reconstruction_samples3(self, n=18):

        (imgs_a, _), (imgs_b, _) = self.dataset.get_random_fixed_batch(n)
        enc_b = self.encoders['a'].predict(imgs_b)
        a_hat = self.decoders['a'].predict(enc_b)

        return imgs_b, a_hat

    def compute_reconstruction_samples4(self, n=18):

        (imgs_a, _), (imgs_b, _) = self.dataset.get_random_fixed_batch(n)
        enc_a = self.encoders['a'].predict(imgs_a)
        b_hat = self.decoders['b'].predict(enc_a)

        return imgs_a, b_hat

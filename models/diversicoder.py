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

from .layers import (conv2d, deconv2d, LayerNorm, rdeconv, res, dense,
                     reparametrization_layer)
from .utils import set_trainable


class DiverseAutoencoder(BaseModel):
    name = 'diversicoder'

    def __init__(self,
                 input_shape=(64, 64, 3),
                 embedding_dim=128,
                 n_filters_factor=128,
                 n_domains=2,
                 disentangled_embedding=False,
                 use_single_decoder=False,
                 cycle_consistency_type='data',
                 autoencoder_type='classic',
                 disentangled_embedding_type='equal',
                 feature_critic_coverage='full',
                 enforce_specificity_on_metatrain=False,
                 cycle_consistency_coverage='meta-only',
                 use_vae=False,
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
        self.use_single_decoder = use_single_decoder
        self.disentangled_embedding = disentangled_embedding
        self.cycle_consistency_type = cycle_consistency_type
        self.autoencoder_type = autoencoder_type
        self.disentangled_embedding_type = disentangled_embedding_type
        self.feature_critic_coverage = feature_critic_coverage
        self.enforce_specificity_on_metatrain = enforce_specificity_on_metatrain
        self.cycle_consistency_coverage = cycle_consistency_coverage
        self.use_vae = use_vae

        pprint(vars(self))
        self.build_model()

    def compute_full_embedding(self, x, general_embedding, vae_z, id):
        zero_embedding = Lambda(lambda x: K.zeros((K.shape(x)[0], self.embedding_size)))(general_embedding)
        if self.disentangled_embedding:
            sp_embedding = self.encoders[id](x)
            if self.use_vae:
                sp_embedding = reparametrization_layer(self.embedding_size)([sp_embedding, vae_z])
            if id == 'a':
                sp_embedding = Concatenate(axis=-1)([sp_embedding, zero_embedding])
            else:
                sp_embedding = Concatenate(axis=-1)([zero_embedding, sp_embedding])
            x_embedding = Concatenate(axis=-1)([general_embedding, sp_embedding])
        else:
            x_embedding = general_embedding
            sp_embedding = zero_embedding
        return x_embedding, sp_embedding

    def reparametrization_for_prediction(self, x, vae_z):
        mu = x[:, :self.embedding_size]
        sigma = x[:, self.embedding_size:]
        return sigma * vae_z + mu

    def predict_full_embedding(self, x, id):
        x_embedding = self.encoders['general'].predict(x)
        if self.disentangled_embedding:
            zero_embedding = np.zeros((x.shape[0], self.embedding_size))
            particular_embedding = self.encoders[id].predict(x)
            if self.use_vae:
                vae_z = np.random.normal(size=(len(x), self.embedding_size))
                x_embedding = self.reparametrization_for_prediction(x_embedding, vae_z)
                particular_embedding = self.reparametrization_for_prediction(particular_embedding, vae_z)
            if id == 'a':
                c = [x_embedding, particular_embedding, zero_embedding]
            else:
                c = [x_embedding, zero_embedding, particular_embedding]
            x_embedding = np.concatenate(c, axis=-1)
        else:
            if self.use_vae:
                vae_z = np.random.normal(size=(len(x), self.embedding_size))
                x_embedding = self.reparametrization_for_prediction(x_embedding, vae_z)
        return x_embedding

    def predict_crossdomain_embedding(self, x_a, x_b, target_id):
        x_embedding = self.encoders['general'].predict(x_a)
        if self.disentangled_embedding:
            zero_embedding = np.zeros((x_b.shape[0], self.embedding_size))
            particular_embedding = self.encoders[target_id].predict(x_b)
            if self.use_vae:
                vae_z = np.random.normal(size=(len(x_a), self.embedding_size))
                x_embedding = self.reparametrization_for_prediction(x_embedding, vae_z)
                particular_embedding = self.reparametrization_for_prediction(particular_embedding, vae_z)
            if id == 'b':
                c = [x_embedding, particular_embedding, zero_embedding]
            else:
                c = [x_embedding, zero_embedding, particular_embedding]
            x_embedding = np.concatenate(c, axis=-1)
        else:
            if self.use_vae:
                vae_z = np.random.normal(size=(len(x_a), self.embedding_size))
                x_embedding = self.reparametrization_for_prediction(x_embedding, vae_z)
        return x_embedding

    def invert_id(self, id):
        if id == 'a':
            return 'b'
        else:
            return 'a'

    def build_trainer(self, id):
        a_input = Input(shape=self.input_shape, name='a_input')
        b_input = Input(shape=self.input_shape, name='b_input')
        vae_z = Input(shape=(self.embedding_size,), name='vae_z')

        general_embedding = self.encoders['general'](a_input)
        if self.use_vae:
            general_embedding = reparametrization_layer(self.embedding_size)([general_embedding, vae_z])
        x_embedding, sp_embedding = self.compute_full_embedding(a_input, general_embedding, vae_z, id)
        x_hat = self.decoders[id](x_embedding)

        if self.feature_critic_coverage == 'full':
            aux_loss = self.metaloss_net(x_embedding)
        elif self.feature_critic_coverage == 'general':
            aux_loss = self.metaloss_net(general_embedding)
        elif self.feature_critic_coverage == 'specific':
            aux_loss = self.metaloss_net(sp_embedding)

        if self.cycle_consistency_coverage != 'meta-only':
            zero_emb = Lambda(lambda x: K.zeros((K.shape(x)[0], self.embedding_size)))(sp_embedding)
            inv_id = self.invert_id(id)
            inv_sp_emb = self.encoders[inv_id](b_input)
            if id == 'b':
                c = [general_embedding, inv_sp_emb, zero_emb]
            else:
                c = [general_embedding, zero_emb, inv_sp_emb]
            full_cross_emb = Concatenate(axis=-1)(c)
            hat_ab = self.decoders[inv_id](full_cross_emb)
            crossed_g_embedding = self.encoders['general'](hat_ab)
            if id == 'b':
                c = [crossed_g_embedding, sp_embedding]
            else:
                c = [crossed_g_embedding, sp_embedding]
            full_crossed_emb = Concatenate(axis=-1)(c)
            hat_aba = self.decoders[id](full_crossed_emb)
            aux_loss = Add()([aux_loss, MSELayer()([a_input, hat_aba])])

        return Model([a_input, b_input, vae_z], [x_hat, aux_loss], name="ae_{}".format(id))

    def compute_metaloss_with_cycled_data(self, omega_e, a_val, b_val, vae_z):

        # compute general embeddings
        m_emb_old_a = omega_e['g_old'](a_val)
        m_emb_old_b = omega_e['g_old'](b_val)
        m_emb_new_a = omega_e['g_new'](a_val)
        m_emb_new_b = omega_e['g_new'](b_val)

        if self.use_vae:
            m_emb_old_a = reparametrization_layer(self.embedding_size)([m_emb_old_a, vae_z])
            m_emb_old_b = reparametrization_layer(self.embedding_size)([m_emb_old_b, vae_z])
            m_emb_new_a = reparametrization_layer(self.embedding_size)([m_emb_new_a, vae_z])
            m_emb_new_b = reparametrization_layer(self.embedding_size)([m_emb_new_b, vae_z])

        # compute specific embeddings
        zero_emb = Lambda(lambda x: K.zeros((K.shape(x)[0], self.embedding_size)))(m_emb_old_a)
        sp_old_a = omega_e['a_old'](a_val)
        sp_old_b = omega_e['b_old'](b_val)
        sp_new_a = omega_e['a_new'](a_val)
        sp_new_b = omega_e['b_new'](b_val)

        if self.use_vae:
            sp_old_a = reparametrization_layer(self.embedding_size)([sp_old_a, vae_z])
            sp_old_b = reparametrization_layer(self.embedding_size)([sp_old_b, vae_z])
            sp_new_a = reparametrization_layer(self.embedding_size)([sp_new_a, vae_z])
            sp_new_b = reparametrization_layer(self.embedding_size)([sp_new_b, vae_z])

        # usual autoencoding
        emb_old_a = Concatenate(axis=-1)([m_emb_old_a, sp_old_a, zero_emb])
        emb_old_b = Concatenate(axis=-1)([m_emb_old_b, zero_emb, sp_old_b])
        emb_new_a = Concatenate(axis=-1)([m_emb_new_a, sp_new_a, zero_emb])
        emb_new_b = Concatenate(axis=-1)([m_emb_new_b, zero_emb, sp_new_b])
        old_hat_a = self.decoders['a'](emb_old_a)
        new_hat_a = self.decoders['a'](emb_new_a)
        old_hat_b = self.decoders['b'](emb_old_b)
        new_hat_b = self.decoders['b'](emb_new_b)

        # cycled autoencoding
        emb_cross_old_ab = Concatenate(axis=-1)([m_emb_old_a, zero_emb, sp_old_b])
        emb_cross_old_ba = Concatenate(axis=-1)([m_emb_old_b, sp_old_a, zero_emb])
        emb_cross_new_ab = Concatenate(axis=-1)([m_emb_new_a, zero_emb, sp_new_b])
        emb_cross_new_ba = Concatenate(axis=-1)([m_emb_new_b, sp_new_a, zero_emb])

        old_hat_ab = self.decoders['b'](emb_cross_old_ab)
        new_hat_ab = self.decoders['b'](emb_cross_new_ab)
        old_hat_ba = self.decoders['a'](emb_cross_old_ba)
        new_hat_ba = self.decoders['a'](emb_cross_new_ba)

        g_emb_crossed_old_ab = omega_e['g_old'](old_hat_ab)
        g_emb_crossed_old_ba = omega_e['g_old'](old_hat_ba)
        g_emb_crossed_new_ab = omega_e['g_new'](new_hat_ab)
        g_emb_crossed_new_ba = omega_e['g_new'](new_hat_ba)
        if self.use_vae:
            g_emb_crossed_old_ab = reparametrization_layer(self.embedding_size)([g_emb_crossed_old_ab, vae_z])
            g_emb_crossed_old_ba = reparametrization_layer(self.embedding_size)([g_emb_crossed_old_ba, vae_z])
            g_emb_crossed_new_ab = reparametrization_layer(self.embedding_size)([g_emb_crossed_new_ab, vae_z])
            g_emb_crossed_new_ba = reparametrization_layer(self.embedding_size)([g_emb_crossed_new_ba, vae_z])

        emb_crossed_old_aba = Concatenate(axis=-1)([g_emb_crossed_old_ab, sp_old_a, zero_emb])
        emb_crossed_old_bab = Concatenate(axis=-1)([g_emb_crossed_old_ba, zero_emb, sp_old_b])
        emb_crossed_new_aba = Concatenate(axis=-1)([g_emb_crossed_new_ab, sp_new_a, zero_emb])
        emb_crossed_new_bab = Concatenate(axis=-1)([g_emb_crossed_new_ba, zero_emb, sp_new_b])
        old_crossed_hat_a = self.decoders['a'](emb_crossed_old_aba)
        new_crossed_hat_a = self.decoders['a'](emb_crossed_new_aba)
        old_crossed_hat_b = self.decoders['b'](emb_crossed_old_bab)
        new_crossed_hat_b = self.decoders['b'](emb_crossed_new_bab)

        # finally, the losses
        mse_val_old = Add()([MSELayer()([a_val, old_hat_a]),
                             MSELayer()([b_val, old_hat_b]),
                             MSELayer()([a_val, old_crossed_hat_a]),
                             MSELayer()([b_val, old_crossed_hat_b])])
        mse_val_new = Add()([MSELayer()([a_val, new_hat_a]),
                             MSELayer()([b_val, new_hat_b]),
                             MSELayer()([a_val, new_crossed_hat_a]),
                             MSELayer()([b_val, new_crossed_hat_b])])
        if self.enforce_specificity_on_metatrain:
            mse_val_old = Subtract()([mse_val_old,
                                      Add()([MSELayer()([a_val, old_hat_ab]),
                                             MSELayer()([b_val, old_hat_ba])])])
            mse_val_new = Subtract()([mse_val_new,
                                      Add()([MSELayer()([a_val, new_hat_ab]),
                                             MSELayer()([b_val, new_hat_ba])])])
        return mse_val_old, mse_val_new

    def compute_metaloss_with_cycled_representation(self, omega_e, a_val, b_val, vae_z):

        # compute general embeddings
        m_emb_old_a = omega_e['g_old'](a_val)
        m_emb_old_b = omega_e['g_old'](b_val)
        m_emb_new_a = omega_e['g_new'](a_val)
        m_emb_new_b = omega_e['g_new'](b_val)

        if self.use_vae:
            m_emb_old_a = reparametrization_layer(self.embedding_size)([m_emb_old_a, vae_z])
            m_emb_old_b = reparametrization_layer(self.embedding_size)([m_emb_old_b, vae_z])
            m_emb_new_a = reparametrization_layer(self.embedding_size)([m_emb_new_a, vae_z])
            m_emb_new_b = reparametrization_layer(self.embedding_size)([m_emb_new_b, vae_z])

        # compute specific embeddings
        zero_emb = Lambda(lambda x: K.zeros((K.shape(x)[0], self.embedding_size)))(m_emb_old_a)
        sp_old_a = omega_e['a_old'](a_val)
        sp_old_b = omega_e['b_old'](b_val)
        sp_new_a = omega_e['a_new'](a_val)
        sp_new_b = omega_e['b_new'](b_val)

        if self.use_vae:
            sp_old_a = reparametrization_layer(self.embedding_size)([sp_old_a, vae_z])
            sp_old_b = reparametrization_layer(self.embedding_size)([sp_old_b, vae_z])
            sp_new_a = reparametrization_layer(self.embedding_size)([sp_new_a, vae_z])
            sp_new_b = reparametrization_layer(self.embedding_size)([sp_new_b, vae_z])

        # usual autoencoding
        emb_old_a = Concatenate(axis=-1)([m_emb_old_a, sp_old_a, zero_emb])
        emb_old_b = Concatenate(axis=-1)([m_emb_old_b, zero_emb, sp_old_b])
        emb_new_a = Concatenate(axis=-1)([m_emb_new_a, sp_new_a, zero_emb])
        emb_new_b = Concatenate(axis=-1)([m_emb_new_b, zero_emb, sp_new_b])
        old_hat_a = self.decoders['a'](emb_old_a)
        new_hat_a = self.decoders['a'](emb_new_a)
        old_hat_b = self.decoders['b'](emb_old_b)
        new_hat_b = self.decoders['b'](emb_new_b)

        # cycled autoencoding
        emb_cross_old_ab = Concatenate(axis=-1)([emb_old_a, zero_emb, sp_old_b])
        emb_cross_old_ba = Concatenate(axis=-1)([emb_old_b, sp_old_a, zero_emb])
        emb_cross_new_ab = Concatenate(axis=-1)([emb_new_a, zero_emb, sp_new_b])
        emb_cross_new_ba = Concatenate(axis=-1)([emb_new_b, sp_new_a, zero_emb])

        old_hat_ab = self.decoders['b'](emb_cross_old_ab)
        new_hat_ab = self.decoders['b'](emb_cross_new_ab)
        old_hat_ba = self.decoders['a'](emb_cross_old_ba)
        new_hat_ba = self.decoders['a'](emb_cross_new_ba)

        g_emb_crossed_old_ab = omega_e['g_old'](old_hat_ab)
        g_emb_crossed_old_ba = omega_e['g_old'](old_hat_ba)
        g_emb_crossed_new_ab = omega_e['g_new'](new_hat_ab)
        g_emb_crossed_new_ba = omega_e['g_new'](new_hat_ba)
        if self.use_vae:
            g_emb_crossed_old_ab = reparametrization_layer(self.embedding_size)([g_emb_crossed_old_ab, vae_z])
            g_emb_crossed_old_ba = reparametrization_layer(self.embedding_size)([g_emb_crossed_old_ba, vae_z])
            g_emb_crossed_new_ab = reparametrization_layer(self.embedding_size)([g_emb_crossed_new_ab, vae_z])
            g_emb_crossed_new_ba = reparametrization_layer(self.embedding_size)([g_emb_crossed_new_ba, vae_z])

        # finally, the losses
        mse_val_old = Add()([MSELayer()([a_val, old_hat_a]),
                             MSELayer()([b_val, old_hat_b]),
                             MSELayer()([m_emb_old_a, g_emb_crossed_old_ab]),
                             MSELayer()([m_emb_old_b, g_emb_crossed_old_ba])])
        mse_val_new = Add()([MSELayer()([a_val, new_hat_a]),
                             MSELayer()([b_val, new_hat_b]),
                             MSELayer()([m_emb_new_a, g_emb_crossed_new_ab]),
                             MSELayer()([m_emb_new_b, g_emb_crossed_new_ba])])
        return mse_val_old, mse_val_new

    def compute_metaloss_classic(self, omega_e, a_val, b_val, vae_z):

        # compute general embeddings
        emb_old_a = omega_e['g_old'](a_val)
        emb_old_b = omega_e['g_old'](b_val)
        emb_new_a = omega_e['g_new'](a_val)
        emb_new_b = omega_e['g_new'](b_val)
        if self.use_vae:
            emb_old_a = reparametrization_layer(self.embedding_size)([emb_old_a, vae_z])
            emb_old_b = reparametrization_layer(self.embedding_size)([emb_old_b, vae_z])
            emb_new_a = reparametrization_layer(self.embedding_size)([emb_new_a, vae_z])
            emb_new_b = reparametrization_layer(self.embedding_size)([emb_new_b, vae_z])

        # decode back into images
        meta_val_old_hat_a = self.decoders['a'](emb_old_a)
        meta_val_new_hat_a = self.decoders['a'](emb_new_a)
        meta_val_old_hat_b = self.decoders['b'](emb_old_b)
        meta_val_new_hat_b = self.decoders['b'](emb_new_b)
        mse_val_old = Add()([MSELayer()([a_val, meta_val_old_hat_a]),
                             MSELayer()([b_val, meta_val_old_hat_b])])
        mse_val_new = Add()([MSELayer()([a_val, meta_val_new_hat_a]),
                             MSELayer()([b_val, meta_val_new_hat_b])])

        return mse_val_old, mse_val_new

    def build_metatrainer(self):
        a_input = Input(shape=self.input_shape, name='a_input')
        b_input = Input(shape=self.input_shape, name='b_input')
        a_val_input = Input(shape=self.input_shape)
        b_val_input = Input(shape=self.input_shape)
        vae_z = Input(shape=(self.embedding_size,), name='meta_vae_z')

        # simulate training on the training set

        # compute autoencoding process
        general_embedding_a = self.encoders['general'](a_input)
        general_embedding_b = self.encoders['general'](b_input)
        if self.use_vae:
            general_embedding_a = reparametrization_layer(self.embedding_size)([general_embedding_a, vae_z])
            general_embedding_b = reparametrization_layer(self.embedding_size)([general_embedding_b, vae_z])

        x_embedding_a, sp_embedding_a = self.compute_full_embedding(a_input, general_embedding_a, vae_z, 'a')
        x_embedding_b, sp_embedding_b = self.compute_full_embedding(b_input, general_embedding_b, vae_z, 'b')
        x_hat_a = self.decoders['a'](x_embedding_a)
        x_hat_b = self.decoders['b'](x_embedding_b)

        # get both losses
        mse_loss = Add()([MSELayer()([a_input, x_hat_a]), MSELayer()([b_input, x_hat_b])])

        if self.feature_critic_coverage == 'full':
            aux_loss = Add()([self.metaloss_net(x_embedding_a), self.metaloss_net(x_embedding_b)])
        elif self.feature_critic_coverage == 'general':
            aux_loss = Add()([self.metaloss_net(general_embedding_a), self.metaloss_net(general_embedding_b)])
        elif self.feature_critic_coverage == 'specific':
            aux_loss = Add()([self.metaloss_net(sp_embedding_a), self.metaloss_net(sp_embedding_b)])

        # create models with each possible update (with and without aux_loss)
        omega_e = {}
        self.omega_e = omega_e
        omega_e['g_old'], omega_e['g_new'] = self.build_encoder_omega(mse_loss, aux_loss, id='general')

        if self.disentangled_embedding:
            omega_e['a_old'], omega_e['a_new'] = self.build_encoder_omega(mse_loss, aux_loss, id='a')
            omega_e['b_old'], omega_e['b_new'] = self.build_encoder_omega(mse_loss, aux_loss, id='b')

        # self.decoder_omega_old[id] = self.build_decoder_omega(id, mse_loss, aux_loss)

        # use new models on the validation set

        # if we are going to go disentangled, that should be cycle-based
        if self.disentangled_embedding and self.cycle_consistency_coverage != 'normal-only' and self.cycle_consistency_type == 'data':
            mse_val_old, mse_val_new = self.compute_metaloss_with_cycled_data(omega_e, a_val_input, b_val_input, vae_z)
        elif self.disentangled_embedding and self.cycle_consistency_coverage != 'normal-only' and self.cycle_consistency_type == 'representation':
            mse_val_old, mse_val_new = self.compute_metaloss_with_cycled_representation(omega_e, a_val_input, b_val_input, vae_z)
        else:
            mse_val_old, mse_val_new = self.compute_metaloss_classic(omega_e, a_val_input, b_val_input, vae_z)

        metaloss = Activation('tanh')(Subtract()([mse_val_new, mse_val_old]))
        fix_dim = Reshape((-1,))

        # print_layer = Lambda(lambda x: K.tf.Print(x, K.gradients(fix_dim(metaloss), self.metaloss_net.trainable_weights)))
        # metaloss = print_layer(metaloss)
        print(K.gradients(fix_dim(metaloss), self.metaloss_net.trainable_weights))
        dummy_loss = Subtract()([aux_loss, aux_loss])
        return Model([a_input, b_input, a_val_input, b_val_input, vae_z], [fix_dim(metaloss), dummy_loss], name="meta_{}".format(id))

    def build_model(self):

        self.metaloss_net = self.build_metaloss_net()
        self.encoders['general'], self.l_enc['general'] = self.build_encoder('general')

        if self.disentangled_embedding:
            self.encoders['a'], self.l_enc['a'] = self.build_encoder('a')
            self.encoders['b'], self.l_enc['b'] = self.build_encoder('b')

        if self.use_single_decoder:
            self.decoders['a'], _ = self.build_decoder('a')
            self.decoders['b'] = self.decoders['a']
        else:
            self.decoders['a'], _ = self.build_decoder('a')
            self.decoders['b'], _ = self.build_decoder('b')

        self.trainer_a = self.build_trainer('a')
        self.trainer_b = self.build_trainer('b')
        self.optimizers = self.build_optmizers()

        self.metatrainer = self.build_metatrainer()

        # compile autoencoders
        set_trainable(self.metaloss_net, False)
        self.trainer_a.compile(
            optimizer=self.optimizers['a'],
            loss=['mse', loss_is_output],
            loss_weights=[1., 1.])
        self.trainer_b.compile(
            optimizer=self.optimizers['b'],
            loss=['mse', loss_is_output],
            loss_weights=[1., 1.])

        for e in self.encoders.values():
            set_trainable(e, False)
        for d in self.decoders.values():
            set_trainable(d, False)
        set_trainable(self.metaloss_net, True)
        print(self.metatrainer.trainable_weights)
        self.metatrainer.compile(
            optimizer=self.optimizers['meta_a'],
            loss=[loss_is_output, loss_is_output],
            loss_weights=[1., 0.])

    def build_optmizers(self):
        ae_a = Adam(lr=self.lr, beta_1=0.5)
        ae_b = Adam(lr=self.lr, beta_1=0.5)
        meta_a = Adam(lr=self.lr * 1e-1, beta_1=0.5)
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

        vae_repar_z = np.random.normal(size=(self.batchsize, self.embedding_size))

        ld = {}
        _, ld['ae_loss_a'], ld['auxloss_a'] = self.trainer_a.train_on_batch([x_a, x_b, vae_repar_z], [x_a, dummy_y])
        _, ld['ae_loss_b'], ld['auxloss_b'] = self.trainer_b.train_on_batch([x_b, x_a, vae_repar_z], [x_b, dummy_y])
        _, ld['metaloss'], _ = self.metatrainer.train_on_batch([x_a, x_b, x_a_val, x_b_val, vae_repar_z], [dummy_yy, dummy_yy])
        # print(np.mean(K.eval(self.metaloss_net.trainable_weights[2])))
        # print(np.mean(K.eval(self.omega_e['g_new'].layers[6].kernel)))
        # print(np.mean(K.eval(self.omega_e['g_old'].layers[6].kernel)))
        # print(np.mean(K.eval(self.encoders['general'].layers[6].kernel)))
        return ld

    """
        # Network Layers' Definition
    """

    def build_encoder_omega(self, loss, auxloss, id='general'):

        inputs = Input(shape=self.input_shape)

        x = {}
        for layer in self.l_enc[id]:
            lr = 1e-1
            try:
                k_grads_l, b_grads_l = layer.get_gradients(loss)
                k_grads_aux, b_grads_aux = layer.get_gradients(auxloss)
                print(k_grads_aux)
                layer.create_new_call_with_added_values(-lr * k_grads_l, -lr * b_grads_l, 'old')
                layer.create_new_call_with_added_values(-lr * (k_grads_l + k_grads_aux), -lr * (b_grads_aux + b_grads_l), 'new')
                x['old'] = layer.fun(x.get('old', inputs), 'old')
                x['new'] = layer.fun(x.get('new', inputs), 'new')
            except AttributeError:
                x['old'] = layer(x.get('old', inputs))
                x['new'] = layer(x.get('new', inputs))

        old_model = Model(inputs, x['old'], name="encoder_omega_old_{}".format(id))
        new_model = Model(inputs, x['new'], name="encoder_omega_new_{}".format(id))

        return old_model, new_model

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

    def build_encoder(self, id):
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

        if self.use_vae:
            ls += [Flatten(),
                   dense(self.embedding_size * 2, activation='linear')]
        else:
            ls += [Flatten(),
                   dense(self.embedding_size, activation='linear')]

        x = inputs
        for l in ls:
            x = l(x)

        return Model(inputs, x, name='encoder_{}'.format(id)), ls

    def build_decoder(self, id):
        if self.disentangled_embedding:
            if self.disentangled_embedding_type == 'equal':
                e_size = self.embedding_size * 3
            elif self.disentangled_embedding_type == 'full-half-half':
                e_size = self.embedding_size * 2
        else:
            e_size = self.embedding_size
        z_input = Input(shape=(e_size,))
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

        return Model(z_input, x, name='decoder_{}'.format(id)), ls

    def build_metaloss_net(self):
        if self.disentangled_embedding:
            if self.feature_critic_coverage == 'full':
                if self.disentangled_embedding_type == 'equal':
                    e_size = self.embedding_size * 3
                elif self.disentangled_embedding_type == 'full-half-half':
                    e_size = self.embedding_size * 2
            elif self.feature_critic_coverage == 'general':
                e_size = self.embedding_size
            elif self.feature_critic_coverage == 'specific':
                if self.disentangled_embedding_type == 'equal':
                    e_size = self.embedding_size * 2
                elif self.disentangled_embedding_type == 'full-half-half':
                    e_size = self.embedding_size

        else:
            e_size = self.embedding_size
        embedding_input = Input(shape=(e_size,))

        x = dense(128, activation='relu', bnorm=False)(embedding_input)
        x = dense(128, activation='relu', bnorm=False)(x)
        x = dense(1, activation='softplus', bnorm=False)(x)

        return Model(embedding_input, x, name='aux_net')

    """
        # Computation of metrics inputs
    """

    def compute_reconstruction_samples(self, n=18):

        (imgs_a, _), (imgs_b, _) = self.dataset.get_batch_from_validation_set(n)
        enc_a = self.predict_full_embedding(imgs_a, 'a')
        a_hat = self.decoders['a'].predict(enc_a)

        return imgs_a, a_hat

    def compute_reconstruction_samples2(self, n=18):

        (imgs_a, _), (imgs_b, _) = self.dataset.get_batch_from_validation_set(n)
        enc_b = self.predict_full_embedding(imgs_b, 'b')
        b_hat = self.decoders['b'].predict(enc_b)

        return imgs_b, b_hat

    def compute_reconstruction_samples3(self, n=18):

        (imgs_a, _), (imgs_b, _) = self.dataset.get_batch_from_validation_set(n)
        enc_b = self.predict_crossdomain_embedding(imgs_b, imgs_a, 'a')
        a_hat = self.decoders['a'].predict(enc_b)

        return imgs_b, a_hat

    def compute_reconstruction_samples4(self, n=18):

        (imgs_a, _), (imgs_b, _) = self.dataset.get_batch_from_validation_set(n)
        enc_a = self.predict_crossdomain_embedding(imgs_a, imgs_b, 'b')
        b_hat = self.decoders['b'].predict(enc_a)

        return imgs_a, b_hat

    def compute_labelled_embedding(self, n=10000):
        ds = self.dataset.datasets[0]
        x_data, y_labels = ds.get_random_fixed_batch(n)
        x_feats = self.predict_full_embedding(x_data, 'a')
        if ds.has_test_set():
            x_test, y_test = ds.get_random_perm_of_test_set(n=1000)
            x_test_feats = self.predict_full_embedding(x_test, 'a')
            self.save_precomputed_features('labelled_embedding', x_feats, Y=y_labels,
                                           test_set=(x_test_feats, y_test))
        else:
            self.save_precomputed_features('labelled_embedding', x_feats, Y=y_labels)
        return x_feats, y_labels

    def compute_labelled_embedding2(self, n=10000):
        ds = self.dataset.datasets[0]
        x_data, y_labels = ds.get_random_fixed_batch(n)
        x_feats = self.predict_full_embedding(x_data, 'b')
        if ds.has_test_set():
            x_test, y_test = ds.get_random_perm_of_test_set(n=1000)
            x_test_feats = self.predict_full_embedding(x_test, 'b')
            self.save_precomputed_features('labelled_embedding', x_feats, Y=y_labels,
                                           test_set=(x_test_feats, y_test))
        else:
            self.save_precomputed_features('labelled_embedding2', x_feats, Y=y_labels)
        return x_feats, y_labels


class DiverseAutoencoderSmall(DiverseAutoencoder):
    name = 'diversicoder-small'

    def build_encoder(self, id):
        inputs = Input(shape=self.input_shape)

        ls = []

        if self.input_shape[0] == 32:
            ls = [conv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu')]
        elif self.input_shape[0] == 64:
            ls = [conv2d(64, (4, 4), strides=(4, 4), bnorm=False, activation='relu')]
        elif self.input_shape[0] == 128:
            ls = [conv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu'),
                  conv2d(64, (4, 4), strides=(4, 4), bnorm=False, activation='relu')]

        ls += [conv2d(8, (4, 4), strides=(2, 2), bnorm=False, activation='relu')]

        if self.use_vae:
            ls += [Flatten(),
                   dense(self.embedding_size * 2, activation='linear')]
        else:
            ls += [Flatten(),
                   dense(self.embedding_size, activation='linear')]

        x = inputs
        for l in ls:
            x = l(x)

        return Model(inputs, x, name='encoder_{}'.format(id)), ls

    def build_decoder(self, id):
        if self.disentangled_embedding:
            e_size = self.embedding_size * 3
        else:
            e_size = self.embedding_size
        z_input = Input(shape=(e_size,))
        orig_channels = self.input_shape[2]

        w = 8  # starting width
        ls = [dense(8 * w * w, activation='relu'),
              Reshape((w, w, 8))]

        if self.input_shape[0] >= 64:
            ls += [deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')]
        if self.input_shape[0] >= 128:
            ls += [deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')]

        ls += [deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same'),
               deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same'),
               conv2d(orig_channels, (3, 3), strides=(1, 1), bnorm=False, activation='sigmoid', padding='same')]

        x = z_input
        for l in ls:
            x = l(x)

        return Model(z_input, x, name='decoder_{}'.format(id)), ls
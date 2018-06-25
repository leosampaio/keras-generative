import os
import random
from abc import ABCMeta, abstractmethod
from pprint import pprint

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

import keras
from keras.engine.topology import Layer
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU,
                          LocallyConnected2D, Add,
                          Lambda, AveragePooling1D, GlobalAveragePooling2D)
from keras.optimizers import Adam, Adadelta, RMSprop
from keras import regularizers
from keras import initializers
from keras import backend as K
from keras.applications.mobilenet import MobileNet

from .base import BaseModel

from .utils import *
from .layers import *
from .metrics import *
from . import mmd
from . import inception_score
from . import server

try:
    from .notifyier import *
except ImportError as e:
    print(e)
    print("You did not set a notifyier. Notifications will not be sent anywhere")


def triplet_lossfun_creator(margin=1., zdims=256, inverted=False):
    def triplet_lossfun(_, y_pred):

        m = K.constant(margin)
        zero = K.constant(0.)
        a, p, n = [y_pred[..., i:i + zdims] for i in range(0, y_pred.shape[-1], zdims)]
        if inverted:
            return K.maximum(zero, m + K.sqrt(K.sum(K.square(a - n))))
        else:
            return K.maximum(zero, m + K.sqrt(K.sum(K.square(a - p))) - K.sqrt(K.sum(K.square(a - n))))

    return triplet_lossfun


def discriminator_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p, prediction for pairs (Gx(z), z)
    y_pred[:,1]: q, prediction for pairs (x, Gz(z))
    y_pred[:,2]: p_cycle, prediction for pairs (x, x)
    y_pred[:,3]: q_cycle, prediction for pairs (x, Gx(x))
    """
    p = K.clip(y_pred[:, 0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:, 1], K.epsilon(), 1.0 - K.epsilon())
    p_true = y_true[:, 0]
    q_true = y_true[:, 1]

    q_error = -K.mean(K.log(K.abs(q_true - q)))
    p_error = -K.mean(K.log(K.abs(p - p_true)))

    return q_error + p_error


def generator_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p, prediction for pairs (Gx(z), z)
    y_pred[:,1]: q, prediction for pairs (x, Gz(z))

    """
    p = K.clip(y_pred[:, 0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:, 1], K.epsilon(), 1.0 - K.epsilon())
    p_true = y_true[:, 0]
    q_true = y_true[:, 1]

    q_error = -K.mean(K.log(K.abs(p_true - q)))
    p_error = -K.mean(K.log(K.abs(p - q_true)))

    return q_error + p_error


def mmd_lossfun(y_true, y_pred):
    x = K.batch_flatten(y_true)
    x_hat = K.batch_flatten(y_pred)
    return tf.log(mmd.rbf_mmd2(x, x_hat))


class TOPGAN(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=128,
                 name='topgan',
                 **kwargs):
        super().__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.f_Gx = None
        self.f_D = None

        self.gen_trainer = None
        self.dis_trainer = None

        self.is_conditional = kwargs.get('is_conditional', False)
        self.auxiliary_classifier = kwargs.get('auxiliary_classifier', False)
        self.conditional_dims = kwargs.get('conditional_dims', 0)
        self.conditionals_for_samples = kwargs.get('conditionals_for_samples', False)

        self.triplet_margin = kwargs.get('triplet_margin', 1.0)
        self.triplet_weight = kwargs.get('triplet_weight', 1.0)

        self.last_losses = {
            'g_loss': 10.,
            'd_loss': 10.
        }

        self.svm = LinearSVC()
        self.topgan_type = 'epoch_svm'  # or 'batch_svm'
        self.did_train_svm_for_the_first_time = False

        self.embedding_size = 1024

        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        batchsize = len(x_data)

        # perform label smoothing if applicable
        y_pos, y_neg = smooth_binary_labels(batchsize, self.label_smoothing, one_sided_smoothing=False)
        y = np.stack((y_neg, y_pos), axis=1)

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        x_permutation = np.array(np.random.permutation(batchsize), dtype='int64')
        input_data = [x_data, x_permutation, z_latent_dis]
        label_data = [y, y]

        # train both networks
        _, d_loss, d_triplet = self.dis_trainer.train_on_batch(input_data, label_data)
        _, g_loss, g_triplet = self.gen_trainer.train_on_batch(input_data, label_data)
        if self.last_losses['d_loss'] < self.dis_loss_control:
            _, g_loss, g_triplet = self.gen_trainer.train_on_batch(input_data, label_data)
        if self.last_losses['d_loss'] < self.dis_loss_control * 1e-5:
            for i in range(0, 5):
                _, g_loss, g_triplet = self.gen_trainer.train_on_batch(input_data, label_data)

        losses = {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'd_triplet': d_triplet,
            'g_triplet': g_triplet
        }

        self.last_losses = losses
        return losses

    def predict(self, z_samples):
        return self.f_Gx.predict(z_samples)

    def build_trainer(self):
        input_x = Input(shape=self.input_shape)
        input_x_perm = Input(shape=(1,), dtype='int64')
        input_z = Input(shape=(self.z_dims,))

        assert self.f_D is not None

        x_hat = self.f_Gx(input_z)
        negative_embedding, p = self.f_D(x_hat)
        anchor_embedding, q = self.f_D(input_x)
        positive_embedding = Lambda(lambda x: K.squeeze(K.gather(anchor_embedding, input_x_perm), 1))(anchor_embedding)

        input = [input_x, input_x_perm, input_z]

        concatenated_dis = Concatenate(axis=-1, name="dis_classification")([p, q])
        concatenated_triplet = Concatenate(axis=-1, name="triplet")([anchor_embedding, positive_embedding, negative_embedding])
        output = [concatenated_dis, concatenated_triplet]
        return Model(input, output, name='topgan')

    def build_model(self):

        self.f_Gx = self.build_Gx()  # Moriarty, the encoder
        self.f_D = self.build_D()   # Sherlock, the detective
        self.f_Gx.summary()
        self.f_D.summary()

        opt_d, opt_g = self.build_optmizers()
        loss_d, loss_g, triplet_d_loss, triplet_g_loss = self.define_loss_functions()

        # build discriminator
        self.dis_trainer = self.build_trainer()
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, True)
        self.dis_trainer.compile(optimizer=opt_d,
                                 loss=[loss_d, triplet_d_loss],
                                 loss_weights=[1., 1.])

        # build generators
        self.gen_trainer = self.build_trainer()
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=opt_g,
                                 loss=[loss_g, triplet_g_loss],
                                 loss_weights=[1., self.triplet_weight])

        self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('gen_trainer')

    def save_model(self, out_dir, epoch):
        self.trainers['f_Gx'] = self.f_Gx
        self.trainers['f_D'] = self.f_D
        super().save_model(out_dir, epoch)

    def define_loss_functions(self):
        return (discriminator_lossfun, generator_lossfun,
                triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size),
                triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size, inverted=True))

    def save_images(self, samples, filename, conditionals_for_samples=None):
        '''
        Save images generated from random sample numbers
        '''
        imgs = self.predict(samples)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        self.save_image_as_plot(imgs, filename)

    def did_collapse(self, losses):
        return False

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))

        x = Dense(256, input_dim=self.z_dims, kernel_initializer=initializers.RandomNormal(stddev=0.02))(z_input)
        x = LeakyReLU(0.2)(x)
        x = Dense(512)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(1024)(x)
        x = LeakyReLU(0.2)(x)
        x = Dense(self.input_shape[0] * self.input_shape[1] * self.input_shape[2], activation='sigmoid')(x)
        x = Reshape((self.input_shape[0], self.input_shape[1], self.input_shape[2]))(x)

        return Model(z_input, x)

    def build_D(self):
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.01)(x_input)
        x = BasicConvLayer(64, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        # x = ResLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        # x = ResLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        # x = ResLayer(128, (3, 3), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(256, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x_embedding = Flatten()(x)

        x_embedding = Dense(self.embedding_size)(x_embedding)
        x_embedding = LeakyReLU(0.01)(x_embedding)
        x_embedding = Dropout(0.2)(x_embedding)

        fc_x = Dense(1024)(x_embedding)
        fc_x = LeakyReLU(0.01)(fc_x)
        fc_x = Dropout(0.2)(fc_x)

        fc_x = Dense(1)(fc_x)
        fc_x = Activation('sigmoid')(fc_x)

        return Model(x_input, [x_embedding, fc_x])

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr)
        opt_g = Adam(lr=self.lr)
        return opt_d, opt_g


class TOPGANbasedonInfoGAN(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=128,
                 name='topgan_binfogan',
                 **kwargs):
        super().__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.f_Gx = None
        self.f_D = None

        self.gen_trainer = None
        self.dis_trainer = None

        self.is_conditional = kwargs.get('is_conditional', False)
        self.auxiliary_classifier = kwargs.get('auxiliary_classifier', False)
        self.conditional_dims = kwargs.get('conditional_dims', 0)
        self.conditionals_for_samples = kwargs.get('conditionals_for_samples', False)

        self.triplet_margin = kwargs.get('triplet_margin', 1.0)
        self.triplet_weight = kwargs.get('triplet_weight', 1.0)

        self.last_losses = {
            'g_loss': 10.,
            'd_loss': 10.
        }

        self.svm = LinearSVC()
        self.topgan_type = 'epoch_svm'  # or 'batch_svm'
        self.did_train_svm_for_the_first_time = False

        self.embedding_size = 1024

        self.build_model()

    train_on_batch = TOPGAN.__dict__['train_on_batch']
    predict = TOPGAN.__dict__['predict']
    build_trainer = TOPGAN.__dict__['build_trainer']
    build_model = TOPGAN.__dict__['build_model']
    define_loss_functions = TOPGAN.__dict__['define_loss_functions']
    save_images = TOPGAN.__dict__['save_images']
    did_collapse = TOPGAN.__dict__['did_collapse']
    build_optmizers = TOPGAN.__dict__['build_optmizers']

    def save_model(self, out_dir, epoch):
        self.trainers['f_Gx'] = self.f_Gx
        self.trainers['f_D'] = self.f_D
        super().save_model(out_dir, epoch)

    def build_Gx(self):
        """
        Network Architecture based on the one presented in infoGAN
        """
        z_input = Input(shape=(self.z_dims,))

        x = Dense(1024, input_dim=self.z_dims)(z_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Dense(128 * (self.input_shape[0] // 4) * (self.input_shape[1] // 4))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Reshape((self.input_shape[0] // 4, self.input_shape[1] // 4, 128))(x)

        x = BasicDeconvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = BasicDeconvLayer(self.input_shape[2], (4, 4), strides=(2, 2), bnorm=False, padding='same', activation='sigmoid')(x)

        return Model(z_input, x)

    def build_D(self):
        """
        Network Architecture based on the one presented in infoGAN
        """
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.2)(x_input)
        x = BasicConvLayer(128, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2)(x)
        x_embedding = Flatten()(x)

        x_embedding = Dense(self.embedding_size)(x_embedding)
        x_embedding = BatchNormalization()(x_embedding)
        x_embedding = LeakyReLU(0.2)(x_embedding)

        fc_x = Dense(1)(x_embedding)
        fc_x = Activation('sigmoid')(fc_x)

        return Model(x_input, [x_embedding, fc_x])


class TOPGANwithAEfromEBGAN(BaseModel, metaclass=ABCMeta):

    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=128,
                 name='topgan_ae_ebgan',
                 embedding_dim=256,
                 isolate_d_classifier=False,
                 loss_weights=[1., 1., 1., 1.],
                 loss_control=None,
                 loss_control_epoch=None,
                 **kwargs):
        super().__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.f_Gx = None
        self.f_D = None

        self.gen_trainer = None
        self.dis_trainer = None

        self.loss_names = ['g_loss', 'd_loss', 'd_triplet',
                           'g_triplet', 'ae_loss']
        self.n_losses = len(self.loss_names)

        self.last_losses = dict(list(zip(self.loss_names, [100.] * self.n_losses)))
        self.all_losses = dict(list(zip(self.loss_names, [[] for i in range(self.n_losses)])))
        self.loss_weights = dict(list(zip(self.loss_names, loss_weights)))

        self.triplet_margin = kwargs.get('triplet_margin', 1.0)

        self.embedding_size = embedding_dim
        self.isolate_d_classifier = isolate_d_classifier

        if loss_control:
            if loss_control_epoch is None:
                raise ValueError("You need to specify the epoches to "
                                 "implement with loss control")
            self.loss_control = dict(list(zip(self.loss_names, loss_control)))
            self.loss_control_epoch = dict(list(zip(self.loss_names, loss_control_epoch)))
        else:
            self.loss_control = dict(list(zip(self.loss_names, ['none'] * self.n_losses)))
            self.loss_control_epoch = dict(list(zip(self.loss_names, [1] * self.n_losses)))

        self.last_loss_weights = {l: 0. for l in self.loss_names}
        self.loss_weights_across_time = {l: [] for l in self.loss_names}
        self.backend_losses = {l: K.variable(0) for l in self.loss_names}
        self.opt_states = None
        self.optimizers = None
        self.update_loss_weights()

        pprint(vars(self))

        self.build_model()

    def get_experiment_id(self):
        id = "{}_zdim{}_edim{}".format(self.name, self.z_dims, self.embedding_size)
        for l, k_loss in self.backend_losses.items():
            id = "{}_L{}{}{}{}".format(id, l.replace('_', ''), self.loss_weights[l],
                                       self.loss_control[l].replace('-', ''), self.loss_control_epoch[l])
        return id.replace('.', '')

    train_on_batch = TOPGAN.__dict__['train_on_batch']
    predict = TOPGAN.__dict__['predict']
    build_trainer = TOPGAN.__dict__['build_trainer']
    define_loss_functions = TOPGAN.__dict__['define_loss_functions']
    did_collapse = TOPGAN.__dict__['did_collapse']
    build_optmizers = TOPGAN.__dict__['build_optmizers']

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        batchsize = len(x_data)

        # perform label smoothing if applicable
        y_pos, y_neg = smooth_binary_labels(batchsize, self.label_smoothing, one_sided_smoothing=False)
        y = np.stack((y_neg, y_pos), axis=1)

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        if self.input_noise > 1e-5:
            noise = np.random.normal(scale=self.input_noise, size=x_data.shape)
        else:
            noise = np.zeros(x_data.shape)

        x_permutation = np.array(np.random.permutation(batchsize), dtype='int64')
        input_data = [x_data, noise, x_permutation, z_latent_dis]
        label_data = [y, y, x_data]

        # train both networks
        # _, _, d_triplet, ae_loss = self.ae_triplet_trainer.train_on_batch(input_data, label_data)
        _, d_loss, d_triplet, ae_loss = self.dis_trainer.train_on_batch(input_data, label_data)
        _, g_loss, g_triplet, _ = self.gen_trainer.train_on_batch(input_data, label_data)
        if self.last_losses['d_loss'] < self.dis_loss_control:
            _, g_loss, g_triplet, _ = self.gen_trainer.train_on_batch(input_data, label_data)
        if self.last_losses['d_loss'] < self.dis_loss_control * 1e-2:
            for i in range(0, 5):
                _, g_loss, g_triplet, _ = self.gen_trainer.train_on_batch(input_data, label_data)

        losses = dict(zip(self.loss_names, [g_loss, d_loss, d_triplet, g_triplet, ae_loss]))
        self.last_losses = losses
        for k, v in losses.items():
            self.all_losses[k].append(v)
            self.loss_weights_across_time[k].append(self.last_loss_weights[k])
        return losses

    def did_train_over_an_epoch(self):
        self.update_loss_weights()

    def update_loss_weights(self):
        log_message = "New loss weights: "
        weight_delta = 0.
        for l, k_loss in self.backend_losses.items():
            new_loss = change_losses_weight_over_time(self.current_epoch,
                                                      self.loss_control_epoch[l],
                                                      self.loss_control[l],
                                                      self.loss_weights[l])
            K.set_value(k_loss, new_loss)
            log_message = "{}{}:{}, ".format(log_message, l, K.get_value(k_loss))
            weight_delta = np.max((weight_delta, np.abs(new_loss - self.last_loss_weights[l])))

        if weight_delta > 0.1:
            self.last_loss_weights = {l: K.get_value(k_loss) for l, k_loss in self.backend_losses.items()}
            self.reset_optimizers()
            log_message = "{}. Did reset optmizers".format(log_message)
        print(log_message)

    def build_trainer(self):
        input_x = Input(shape=self.input_shape)
        input_noise = Input(shape=self.input_shape)
        input_x_perm = Input(shape=(1,), dtype='int64')
        input_z = Input(shape=(self.z_dims,))

        assert self.f_D is not None

        clipping_layer = Lambda(lambda x: K.clip(x, 0., 1.))
        
        x_noisy = clipping_layer(Add()([input_x, input_noise]))
        x_hat = self.f_Gx(input_z)

        negative_embedding, p, _ = self.f_D(x_hat)
        anchor_embedding, q, _ = self.f_D(input_x)
        positive_embedding = Lambda(lambda x: K.squeeze(K.gather(anchor_embedding, input_x_perm), 1))(anchor_embedding)
        _, _, x_reconstructed = self.f_D(x_noisy)

        input = [input_x, input_noise, input_x_perm, input_z]

        concatenated_dis = Concatenate(axis=-1, name="dis_classification")([p, q])
        concatenated_triplet = Concatenate(axis=-1, name="triplet")([anchor_embedding, positive_embedding, negative_embedding])
        output = [concatenated_dis, concatenated_triplet, x_reconstructed]
        return Model(input, output, name='topgan')


    def build_model(self):

        self.f_Gx = self.build_Gx()  # Moriarty, the encoder
        self.f_D = self.build_D()   # Sherlock, the detective
        self.f_Gx.summary()
        self.f_D.summary()
        self.encoder.summary()
        self.decoder.summary()
        self.aux_classifier.summary()

        self.optimizers = self.build_optmizers()
        loss_d, loss_g, triplet_d_loss, triplet_g_loss = self.define_loss_functions()

        # build discriminator
        self.dis_trainer = self.build_trainer()
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, not self.isolate_d_classifier)
        set_trainable(self.aux_classifier, True)
        self.dis_trainer.compile(optimizer=self.optimizers["opt_d"],
                                 loss=[loss_d, triplet_d_loss, 'mse'],
                                 loss_weights=[self.backend_losses['d_loss'], self.backend_losses['d_triplet'], self.backend_losses['ae_loss']])

        # build autoencoder+triplet
        # self.ae_triplet_trainer = self.build_trainer()
        # set_trainable(self.f_Gx, False)
        # set_trainable(self.f_D, True)
        # set_trainable(self.aux_classifier, not self.isolate_d_classifier)
        # self.ae_triplet_trainer.compile(optimizer=self.optimizers["opt_ae"],
        #                                 loss=[loss_d, triplet_d_loss, 'mse'],
        #                                 loss_weights=[0., self.backend_losses['d_triplet'], self.backend_losses['ae_loss']])

        # build generators
        self.gen_trainer = self.build_trainer()
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=self.optimizers["opt_g"],
                                 loss=[loss_g, triplet_g_loss, 'mse'],
                                 loss_weights=[self.backend_losses['g_loss'], self.backend_losses['g_triplet'], 0.])

        self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('gen_trainer')
        self.store_to_save('f_Gx')
        self.store_to_save('f_D')

        self.build_mmd_calc_model()
        self.build_inception_eval_model()

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr)
        opt_g = Adam(lr=self.lr)
        opt_ae = RMSprop(lr=self.lr)
        return {"opt_d": opt_d,
                "opt_g": opt_g,
                "opt_ae": opt_ae}

    def reset_optimizers(self):
        if self.opt_states is None:  # never did that, initialize
            if self.optimizers is not None:
                self.opt_states = {k: opt.get_weights() for k, opt in self.optimizers.items()}
            return
        for k, opt in self.optimizers.items():
            opt.set_weights(self.opt_states[k])

    def save_model(self, out_dir, epoch):
        self.trainers['f_Gx'] = self.f_Gx
        self.trainers['f_D'] = self.f_D
        super().save_model(out_dir, epoch)

    def plot_losses_hist(self, outfile):
        pass

    def get_extra_metrics_for_plot(self):
        metrics = [(self.all_losses['g_loss'], self.all_losses['d_loss']),
                   self.all_losses['g_triplet'],
                   self.all_losses['d_triplet'],
                   self.all_losses['ae_loss'], tuple(self.loss_weights_across_time.values())]
        iters = [list(range(len(self.all_losses['ae_loss'])))] * len(metrics)
        m_names = [('g_loss', 'd_loss'), 'g_triplet', 'd_triplet', 'ae_loss', tuple(self.loss_weights_across_time.keys())]
        types = ['lines'] * len(metrics)
        return metrics, iters, m_names, types

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(inputs)
        x = BasicConvLayer(64, (4, 4), strides=(1, 1), bnorm=False, activation='relu')(x)
        x = BasicConvLayer(128, (4, 4), strides=(1, 1), bnorm=True, activation='relu')(x)
        x = BasicConvLayer(128, (4, 4), strides=(2, 2), bnorm=True, activation='relu')(x)
        x = BasicConvLayer(256, (4, 4), strides=(1, 1), bnorm=True, activation='relu')(x)
        x = Flatten()(x)

        x = Dense(self.embedding_size)(x)
        x = Activation('linear')(x)

        return Model(inputs, x)

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))
        orig_channels = self.input_shape[2]

        w = self.input_shape[0] // 2**3
        c = self.embedding_size // (w * w)
        try:
            x = Reshape((w, w, c))(z_input)
        except ValueError:
            raise ValueError("The embedding size must be divisible by {}*{}"
                             " for input shape {}".format(w, w, self.input_shape))

        x = BasicDeconvLayer(256, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(128, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(32, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(32, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)

        x = BasicConvLayer(32, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(orig_channels, (1, 1), activation='sigmoid', bnorm=False)(x)

        return Model(z_input, x)

    def build_aux_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Dense(256)(embedding_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(256)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(embedding_input, x)

    def build_D(self):
        """
        Network Architecture based on the one presented in infoGAN
        """
        x_input = Input(shape=self.input_shape)

        self.encoder = self.build_encoder()
        x_embedding = self.encoder(x_input)

        self.aux_classifier = self.build_aux_classifier()
        discriminator = self.aux_classifier(x_embedding)

        self.decoder = self.build_decoder()
        x_hat = self.decoder(x_embedding)

        return Model(x_input, [x_embedding, discriminator, x_hat])

    def make_predict_functions(self):
        self.encoder._make_predict_function()
        self.decoder._make_predict_function()
        self.aux_classifier._make_predict_function()
        self.f_D._make_predict_function()
        self.f_Gx._make_predict_function()

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
        orig_channels = self.input_shape[2]

        w = self.input_shape[0] // 2**3
        c = self.z_dims // (w * w)
        try:
            x = Reshape((w, w, c))(z_input)
        except ValueError:
            raise ValueError("Latent space dims must be divisible by {}*{}"
                             " for input shape {}".format(w, w, self.input_shape))

        x = BasicDeconvLayer(256, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(128, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(32, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)
        x = BasicDeconvLayer(32, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1, padding='same')(x)

        x = BasicConvLayer(32, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.1)(x)
        x = BasicConvLayer(orig_channels, (1, 1), activation='sigmoid', bnorm=False)(x)

        return Model(z_input, x)

    def build_mmd_calc_model(self):

        x_ph = tf.placeholder(tf.float32, shape=[None] + list(self.input_shape), name='mmd_x')
        x_hat_ph = tf.placeholder(tf.float32, shape=[None, self.input_shape[0], self.input_shape[1], 3], name='mmd_x_hat')
        if self.input_shape[2] == 1:
            x_ph = tf.image.grayscale_to_rgb(x_ph)
        x_flat = K.batch_flatten(x_ph)
        x_hat_flat = K.batch_flatten(x_hat_ph)
        self.mmd_computer = tf.log(mmd.rbf_mmd2(x_flat, x_hat_flat))

    def build_inception_eval_model(self):
        input_z = Input(shape=(self.z_dims,))

        x_hat = self.f_Gx(input_z)

        if self.input_shape[2] == 1:
            x_hat = Lambda(lambda x: tf.image.grayscale_to_rgb(x))(x_hat)

        self.inception_eval_Gx = Model(input_z, x_hat)
        self.inception_eval_Gx._make_predict_function()

    """
        Define all metrics that can be calculated here
    """

    def save_embedding_features(self, n=10000):
        np.random.seed(14)
        perm = np.random.permutation(len(self.dataset))
        x_data, y_labels = self.dataset.images[perm[:10000]], np.argmax(self.dataset.attrs[perm[:10000]], axis=1)
        np.random.seed()
        x_feats = self.encoder.predict(x_data, batch_size=2000)
        self.save_precalculated_features('embedding', x_feats, Y=y_labels)
        return x_feats, y_labels

    def save_generated_images(self, n=10000):
        np.random.seed(14)
        samples = np.random.normal(size=(n, self.z_dims))
        np.random.seed()

        generated_images = self.inception_eval_Gx.predict(samples, batch_size=2000)
        generated_images = (generated_images*255.)

        self.save_precalculated_features('samples'.format(n), generated_images)
        return generated_images

    def calculate_svm_eval(self, finished_cgraph_use_event):

        x_feats, y_labels = self.load_precalculated_features_if_they_exist('embedding')
        if not isinstance(x_feats, np.ndarray):
            x_feats, y_labels = self.save_embedding_features()
        finished_cgraph_use_event.set()

        return svm_eval(x_feats[1000:], y_labels[1000:], x_feats[:1000], y_labels[:1000])
    svm_eval_metric_type = 'lines'

    def calculate_svm_rbf_eval(self, finished_cgraph_use_event):

        x_feats, y_labels = self.load_precalculated_features_if_they_exist('embedding')
        if not isinstance(x_feats, np.ndarray):
            x_feats, y_labels = self.save_embedding_features()
        finished_cgraph_use_event.set()

        return svm_rbf_eval(x_feats[1000:], y_labels[1000:], x_feats[:1000], y_labels[:1000])
    svm_rbf_eval_metric_type = 'lines'

    def calculate_tsne(self, finished_cgraph_use_event):

        x_feats, y_labels = self.load_precalculated_features_if_they_exist('embedding')
        if not isinstance(x_feats, np.ndarray):
            x_feats, y_labels = self.save_embedding_features()
        finished_cgraph_use_event.set()

        return tsne(x_feats[:1000], y_labels[:1000])
    tsne_metric_type = 'scatter'

    def calculate_lda(self, finished_cgraph_use_event):

        x_feats, y_labels = self.load_precalculated_features_if_they_exist('embedding')
        if not isinstance(x_feats, np.ndarray):
            x_feats, y_labels = self.save_embedding_features()
        finished_cgraph_use_event.set()

        return lda(x_feats[:1000], y_labels[:1000])
    lda_metric_type = 'scatter'

    def calculate_pca(self, finished_cgraph_use_event):

        x_feats, y_labels = self.load_precalculated_features_if_they_exist('embedding')
        if not isinstance(x_feats, np.ndarray):
            x_feats, y_labels = self.save_embedding_features()
        finished_cgraph_use_event.set()

        return pca(x_feats[:1000], y_labels[:1000])
    pca_metric_type = 'scatter'

    def calculate_samples(self, finished_cgraph_use_event):
        np.random.seed(14)
        samples = np.random.normal(size=(36, self.z_dims))
        np.random.seed()
        imgs = self.f_Gx.predict(samples)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        finished_cgraph_use_event.set()
        return imgs
    samples_metric_type = 'image-grid'

    def calculate_ae_rec(self, finished_cgraph_use_event):
        np.random.seed(14)
        perm = np.random.permutation(len(self.dataset))
        imgs_from_dataset = self.dataset.images[perm[:18]]
        noise = np.random.normal(scale=self.input_noise, size=imgs_from_dataset.shape)
        imgs_from_dataset += noise
        np.random.seed()

        imgs = np.zeros((36,) + self.input_shape)
        encoding = self.encoder.predict(imgs_from_dataset)
        x_hat = self.decoder.predict(encoding)
        imgs[0::2] = imgs_from_dataset
        imgs[1::2] = x_hat
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        finished_cgraph_use_event.set()
        return imgs
    ae_rec_metric_type = 'image-grid'

    def calculate_mmd(self, finished_cgraph_use_event):
        x_hat = self.load_precalculated_features_if_they_exist('samples10000', has_labels=False)
        if not isinstance(x_hat, np.ndarray):
            x_hat = self.save_generated_images()
        finished_cgraph_use_event.set()

        np.random.seed(14)
        perm = np.random.permutation(len(self.dataset))
        np.random.seed()

        imgs_from_dataset = self.dataset.images[perm[:10000]]*255.
        mmd = K.get_session().run(self.mmd_computer, feed_dict={'mmd_x:0': imgs_from_dataset, 'mmd_x_hat:0': x_hat})
        
        return mmd
    mmd_metric_type = 'lines'

    def calculate_inception_score(self, finished_cgraph_use_event):
        x_hat = self.load_precalculated_features_if_they_exist('samples10000', has_labels=False)
        if not isinstance(x_hat, np.ndarray):
            x_hat = self.save_generated_images()
        finished_cgraph_use_event.set()

        mean, std = inception_score.get_inception_score(x_hat)
        return mean
    inception_score_metric_type = 'lines'

    def calculate_s_inception_score(self, finished_cgraph_use_event):
        filename = os.path.join(
            self.tmp_out_dir,
            "precalculated_features_{}_e{}.h5".format('samples', self.current_epoch))
        if not os.path.exists(filename):
            _ = self.save_generated_images()
        finished_cgraph_use_event.set()

        mean, std = server.ask_server_for_inception_score(filename)
        return mean
    s_inception_score_metric_type = 'lines'


class TOPGANwithAEforMNIST(TOPGANwithAEfromEBGAN):

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

    def build_aux_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Activation('relu')(embedding_input)
        x = BatchNormalization()(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(64)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

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

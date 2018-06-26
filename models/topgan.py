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


class TOPGANwithAEfromEBGAN(BaseModel, metaclass=ABCMeta):
    name = 'topgan_ae_ebgan'
    loss_names = ['g_loss', 'd_loss', 'd_triplet',
                  'g_triplet', 'ae_loss']
    loss_plot_organization = [('g_loss', 'd_loss'), 'd_triplet',
                              'g_triplet', 'ae_loss']

    def __init__(self,
                 input_shape=(64, 64, 3),
                 embedding_dim=256,
                 isolate_d_classifier=False,
                 triplet_margin=1.,
                 **kwargs):
        super().__init__(input_shape=input_shape, **kwargs)

        self.embedding_size = embedding_dim
        self.isolate_d_classifier = isolate_d_classifier
        self.triplet_margin = triplet_margin

        pprint(vars(self))

        self.build_model()

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
        ld = {}  # loss dictionary
        # _, _, d_triplet, ae_loss = self.ae_triplet_trainer.train_on_batch(input_data, label_data)
        _, ld['d_loss'], ld['d_triplet'], ld['ae_loss'] = self.dis_trainer.train_on_batch(input_data, label_data)
        _, ld['g_loss'], ld['g_triplet'], _ = self.gen_trainer.train_on_batch(input_data, label_data)
        if self.losses['d_loss'].last_value < 0.1:
            _, ld['g_loss'], ld['g_triplet'], _ = self.gen_trainer.train_on_batch(input_data, label_data)
        if self.losses['d_loss'].last_value < 0.001:
            for i in range(0, 5):
                _, ld['g_loss'], ld['g_triplet'], _ = self.gen_trainer.train_on_batch(input_data, label_data)

        return ld

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
                                 loss_weights=[self.losses['d_loss'].backend, self.losses['d_triplet'].backend, self.losses['ae_loss'].backend])

        # build autoencoder+triplet
        # self.ae_triplet_trainer = self.build_trainer()
        # set_trainable(self.f_Gx, False)
        # set_trainable(self.f_D, True)
        # set_trainable(self.aux_classifier, not self.isolate_d_classifier)
        # self.ae_triplet_trainer.compile(optimizer=self.optimizers["opt_ae"],
        #                                 loss=[loss_d, triplet_d_loss, 'mse'],
        #                                 loss_weights=[0., self.losses['d_triplet'].backend, self.losses['ae_loss'].backend])

        # build generators
        self.gen_trainer = self.build_trainer()
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=self.optimizers["opt_g"],
                                 loss=[loss_g, triplet_g_loss, 'mse'],
                                 loss_weights=[self.losses['g_loss'].backend, self.losses['g_triplet'].backend, 0.])

        # store trainers
        self.store_to_save('gen_trainer')
        self.store_to_save('dis_trainer')

        self.build_mmd_calc_model()
        self.build_inception_eval_model()

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr)
        opt_g = Adam(lr=self.lr)
        opt_ae = RMSprop(lr=self.lr)
        return {"opt_d": opt_d,
                "opt_g": opt_g,
                "opt_ae": opt_ae}

    def save_model(self, out_dir, epoch):
        self.trainers['f_Gx'] = self.f_Gx
        self.trainers['f_D'] = self.f_D
        super().save_model(out_dir, epoch)

    def define_loss_functions(self):
        return (discriminator_lossfun, generator_lossfun,
                triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size),
                triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size, inverted=True))

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

    def precompute_and_save_embedding(self, n=10000):
        np.random.seed(14)
        perm = np.random.permutation(len(self.dataset))
        x_data, y_labels = self.dataset.images[perm[:10000]], np.argmax(self.dataset.attrs[perm[:10000]], axis=1)
        np.random.seed()
        x_feats = self.encoder.predict(x_data, batch_size=2000)
        self.save_precalculated_features('embedding', x_feats, Y=y_labels)
        return x_feats, y_labels

    def precompute_and_save_image_samples(self, n=10000):
        np.random.seed(14)
        samples = np.random.normal(size=(n, self.z_dims))
        np.random.seed()

        generated_images = self.inception_eval_Gx.predict(samples, batch_size=2000)
        generated_images = (generated_images * 255.)

        self.save_precalculated_features('samples', generated_images)
        return generated_images

    def calculate_svm_eval(self, finished_cgraph_use_event):

        x_feats, y_labels = self.load_precalculated_features_if_they_exist('embedding')
        if not isinstance(x_feats, np.ndarray):
            x_feats, y_labels = self.precompute_and_save_embedding()
        finished_cgraph_use_event.set()

        return svm_eval(x_feats[1000:], y_labels[1000:], x_feats[:1000], y_labels[:1000])
    svm_eval_metric_type = 'lines'

    def calculate_svm_rbf_eval(self, finished_cgraph_use_event):

        x_feats, y_labels = self.load_precalculated_features_if_they_exist('embedding')
        if not isinstance(x_feats, np.ndarray):
            x_feats, y_labels = self.precompute_and_save_embedding()
        finished_cgraph_use_event.set()

        return svm_rbf_eval(x_feats[1000:], y_labels[1000:], x_feats[:1000], y_labels[:1000])
    svm_rbf_eval_metric_type = 'lines'

    def calculate_tsne(self, finished_cgraph_use_event):

        x_feats, y_labels = self.load_precalculated_features_if_they_exist('embedding')
        if not isinstance(x_feats, np.ndarray):
            x_feats, y_labels = self.precompute_and_save_embedding()
        finished_cgraph_use_event.set()

        return tsne(x_feats[:1000], y_labels[:1000])
    tsne_metric_type = 'scatter'

    def calculate_lda(self, finished_cgraph_use_event):

        x_feats, y_labels = self.load_precalculated_features_if_they_exist('embedding')
        if not isinstance(x_feats, np.ndarray):
            x_feats, y_labels = self.precompute_and_save_embedding()
        finished_cgraph_use_event.set()

        return lda(x_feats[:1000], y_labels[:1000])
    lda_metric_type = 'scatter'

    def calculate_pca(self, finished_cgraph_use_event):

        x_feats, y_labels = self.load_precalculated_features_if_they_exist('embedding')
        if not isinstance(x_feats, np.ndarray):
            x_feats, y_labels = self.precompute_and_save_embedding()
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
        x_hat = self.load_precalculated_features_if_they_exist('samples', has_labels=False)
        if not isinstance(x_hat, np.ndarray):
            x_hat = self.precompute_and_save_image_samples()
        finished_cgraph_use_event.set()

        np.random.seed(14)
        perm = np.random.permutation(len(self.dataset))
        np.random.seed()

        imgs_from_dataset = self.dataset.images[perm[:10000]] * 255.
        mmd = K.get_session().run(self.mmd_computer, feed_dict={'mmd_x:0': imgs_from_dataset, 'mmd_x_hat:0': x_hat})

        return mmd
    mmd_metric_type = 'lines'

    def calculate_inception_score(self, finished_cgraph_use_event):
        x_hat = self.load_precalculated_features_if_they_exist('samples', has_labels=False)
        if not isinstance(x_hat, np.ndarray):
            x_hat = self.precompute_and_save_image_samples()
        finished_cgraph_use_event.set()

        mean, std = inception_score.get_inception_score(x_hat)
        return mean
    inception_score_metric_type = 'lines'

    def calculate_s_inception_score(self, finished_cgraph_use_event):
        filename = os.path.join(
            self.tmp_out_dir,
            "precalculated_features_{}_e{}.h5".format('samples', self.current_epoch))
        if not os.path.exists(filename):
            _ = self.precompute_and_save_image_samples()
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

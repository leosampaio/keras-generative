from pprint import pprint

import numpy as np

from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU,
                          LocallyConnected2D, Add,
                          Lambda, AveragePooling1D, GlobalAveragePooling2D)
from keras.optimizers import Adam, RMSprop
from keras import backend as K

from core.models import BaseModel

from .utils import *
from .layers import *


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


class TOPGANwithAEfromEBGAN(BaseModel):
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
        _, ld['d_loss'], ld['d_triplet'], ld['ae_loss'] = self.dis_trainer.train_on_batch(input_data, label_data)
        _, ld['g_loss'], ld['g_triplet'], _ = self.gen_trainer.train_on_batch(input_data, label_data)
        # if self.losses['d_triplet'].get_mean_of_latest() < 1e-5:
        #     _, ld['g_loss'], ld['g_triplet'], _ = self.gen_trainer.train_on_batch(input_data, label_data)
        # if self.losses['d_triplet'].get_mean_of_latest() == 0.:
        #     for i in range(0, 5):
        #         _, ld['g_loss'], ld['g_triplet'], _ = self.gen_trainer.train_on_batch(input_data, label_data)

        return ld

    def build_trainer(self):
        input_x = Input(shape=self.input_shape)
        input_noise = Input(shape=self.input_shape)
        input_x_perm = Input(shape=(1,), dtype='int64')
        input_z = Input(shape=(self.z_dims,))

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
        set_trainable(self.f_D, True)
        self.dis_trainer.compile(optimizer=self.optimizers["opt_d"],
                                 loss=[loss_d, triplet_d_loss, 'mse'],
                                 loss_weights=[self.losses['d_loss'].backend, self.losses['d_triplet'].backend, self.losses['ae_loss'].backend])

        # build generators
        self.gen_trainer = self.build_trainer()
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=self.optimizers["opt_g"],
                                 loss=[loss_g, triplet_g_loss, 'mse'],
                                 loss_weights=[self.losses['g_loss'].backend, self.losses['g_triplet'].backend, self.losses['ae_loss'].backend])

        # store trainers
        self.store_to_save('gen_trainer')
        self.store_to_save('dis_trainer')

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


class TOPGANwithAEforMNIST(TOPGANwithAEfromEBGAN):
    name = 'topgan_ae_mnist'

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

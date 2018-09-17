from pprint import pprint

import numpy as np
import sklearn as sk

from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Add,
                          Lambda, Conv1D, UpSampling2D)
from keras.optimizers import Adam
from keras import backend as K

from core.models import BaseModel
from core.lossfuns import (triplet_lossfun_creator, discriminator_lossfun,
                           generator_lossfun, triplet_balance_creator,
                           eq_triplet_lossfun_creator,
                           triplet_std_creator)

from .layers import conv2d, deconv2d, LayerNorm
from .utils import (set_trainable, smooth_binary_labels)


class TOPGANwithAEfromBEGAN(BaseModel):
    name = 'topgan-ae-began'
    loss_names = ['g_loss', 'd_loss', 'd_triplet',
                  'g_triplet', 'ae_loss', 'k', 'margin', 'g_std', 'g_mean']
    loss_plot_organization = [('g_loss', 'd_loss'), 'd_triplet',
                              'g_triplet', 'ae_loss', 'k',
                              'margin', ('g_std', 'g_mean')]

    def __init__(self,
                 input_shape=(64, 64, 3),
                 embedding_dim=256,
                 triplet_margin=1.,
                 n_filters_factor=32,
                 use_began_equilibrium=False,
                 began_k_lr=1e-2,
                 use_alignment_layer=False,
                 began_gamma=0.5,
                 use_simplified_triplet=False,
                 use_magan_equilibrium=True,
                 **kwargs):
        super().__init__(input_shape=input_shape, **kwargs)

        self.embedding_size = embedding_dim
        self.triplet_margin = K.variable(triplet_margin)
        self.n_filters_factor = n_filters_factor
        self.use_began_equilibrium = use_began_equilibrium
        self.k_lr = began_k_lr
        self.use_alignment_layer = use_alignment_layer
        self.gamma = began_gamma
        self.use_simplified_triplet = use_simplified_triplet
        self.use_magan_equilibrium = use_magan_equilibrium
        self.did_set_g_triplet_count = 0

        if self.use_magan_equilibrium:
            self.gamma = 1.

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
        label_data = [y, y, x_data, y, y]

        # train both networks
        ld = {}  # loss dictionary
        _, ld['d_loss'], ld['d_triplet'], ld['ae_loss'], _, _ = self.dis_trainer.train_on_batch(input_data, label_data)
        _, ld['g_loss'], ld['g_triplet'], _, ld['g_mean'], ld['g_std'] = self.gen_trainer.train_on_batch(input_data, label_data)
        if self.use_alignment_layer:
            _, _, _, _, balance, _ = self.alignment_layer_trainer.train_on_batch(input_data, label_data)
        if self.use_began_equilibrium:
            ld['k'] = np.clip(K.get_value(self.k_gd_ratio) + self.k_lr * (balance), 0, 1)
            K.set_value(self.k_gd_ratio, ld['k'])
        else:
            ld['k'] = 1
        self.did_set_g_triplet_count += int(K.get_value(self.losses['g_triplet'].backend) > 0.)
        if (self.use_magan_equilibrium and
                (K.get_value(self.triplet_margin) > self.losses['g_triplet'].get_mean_of_latest(100)) and
                self.did_set_g_triplet_count >= 100):
            cur_margin = K.get_value(self.triplet_margin)
            cur_balance = self.losses['g_triplet'].get_mean_of_latest(100)
            direction = cur_balance - cur_margin
            ld['margin'] = cur_margin + self.k_lr * direction
            K.set_value(self.triplet_margin, ld['margin'])
        else:
            ld['margin'] = K.get_value(self.triplet_margin)

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
        concatenated_triplet = Concatenate(axis=-1, name="triplet")(
            [anchor_embedding, positive_embedding, negative_embedding])

        if self.use_alignment_layer:
            aligned_negative_embedding = self.alignment_layer(negative_embedding)
            concatenated_aligned_triplet = Concatenate(axis=-1, name="aligned_triplet")(
                [anchor_embedding, positive_embedding, aligned_negative_embedding])
            output = [concatenated_dis, concatenated_triplet, x_reconstructed,
                      concatenated_aligned_triplet, concatenated_aligned_triplet]
        else:
            output = [concatenated_dis, concatenated_triplet, x_reconstructed,
                      concatenated_triplet, concatenated_triplet]
        return Model(input, output, name='topgan')

    def build_model(self):

        self.f_Gx = self.build_Gx()  # Moriarty, the encoder
        self.f_D = self.build_D()   # Sherlock, the detective
        self.f_Gx.summary()
        self.encoder.summary()
        self.f_D.summary()

        self.optimizers = self.build_optmizers()
        self.k_gd_ratio = K.variable(0)
        loss_d, loss_g, triplet_d_loss, triplet_g_loss, t_mean, t_std = self.define_loss_functions()

        if self.use_alignment_layer:
            self.alignment_layer.summary()
            self.alignment_layer_trainer = self.build_trainer()
            set_trainable(self.f_Gx, False)
            set_trainable(self.f_D, False)
            set_trainable(self.alignment_layer, True)
            self.alignment_layer_trainer.compile(
                optimizer=self.optimizers["opt_ae"],
                loss=[loss_d, triplet_d_loss, 'mse', triplet_g_loss],
                loss_weights=[0, 0, 0, 1.])
            loss_weights = [self.losses['d_loss'].backend, 0, self.losses['ae_loss'].backend, self.losses['d_triplet'].backend, 0.]
            set_trainable(self.alignment_layer, False)
        else:
            loss_weights = [self.losses['d_loss'].backend, self.losses['d_triplet'].backend, self.losses['ae_loss'].backend, 0., 0.]
        self.dis_trainer = self.build_trainer()
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, True)
        self.dis_trainer.compile(optimizer=self.optimizers["opt_d"],
                                 loss=[loss_d, triplet_d_loss, 'mse', triplet_d_loss, t_std],
                                 loss_weights=loss_weights)

        # build generators
        self.gen_trainer = self.build_trainer()
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=self.optimizers["opt_g"],
                                 loss=[loss_g, triplet_g_loss, 'mse', t_mean, t_std],
                                 loss_weights=[self.losses['g_loss'].backend, self.losses['g_triplet'].backend, self.losses['ae_loss'].backend, 0., 0.])

        # store trainers
        self.store_to_save('gen_trainer')
        self.store_to_save('dis_trainer')

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr, beta_1=0.5)
        opt_g = Adam(lr=self.lr, beta_1=0.5)
        opt_ae = Adam(lr=self.lr)
        return {"opt_d": opt_d,
                "opt_g": opt_g,
                "opt_ae": opt_ae}

    def save_model(self, out_dir, epoch):
        self.trainers['f_Gx'] = self.f_Gx
        self.trainers['f_D'] = self.f_D
        super().save_model(out_dir, epoch)

    def define_loss_functions(self):
        if self.use_began_equilibrium:
            triplet_d = eq_triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size, k=self.k_gd_ratio, simplified=self.use_simplified_triplet)
        else:
            triplet_d = triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size, simplified=self.use_simplified_triplet)
        return (discriminator_lossfun, generator_lossfun,
                triplet_d,
                triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size, inverted=True),
                triplet_balance_creator(margin=self.triplet_margin, zdims=self.embedding_size, gamma=K.variable(self.gamma)),
                triplet_std_creator(margin=self.triplet_margin, zdims=self.embedding_size))

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
        x = UpSampling2D()(x)
        x = conv2d(self.n_filters_factor, (3, 3), activation='elu', padding='same')(x)
        x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)
        x = UpSampling2D()(x)
        x = conv2d(self.n_filters_factor, (3, 3), activation='elu', padding='same')(x)
        x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)

        if self.input_shape[0] >= 64:
            x = UpSampling2D()(x)
            x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)
            x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)

        if self.input_shape[0] >= 128:
            x = UpSampling2D()(x)
            x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)
            x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)

        x = conv2d(orig_channels, (3, 3), activation=None)(x)

        return Model(z_input, x)

    def build_d_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Dense(256)(embedding_input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(embedding_input, x)

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
        orig_channels = self.input_shape[2]

        x = Dense(self.n_filters_factor * 8 * 8)(z_input)
        x = Reshape((8, 8, self.n_filters_factor))(x)

        x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)
        x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)
        x = UpSampling2D()(x)
        x = conv2d(self.n_filters_factor, (3, 3), activation='elu', padding='same')(x)
        x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)
        x = UpSampling2D()(x)
        x = conv2d(self.n_filters_factor, (3, 3), activation='elu', padding='same')(x)
        x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)

        if self.input_shape[0] >= 64:
            x = UpSampling2D()(x)
            x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)
            x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)

        if self.input_shape[0] >= 128:
            x = UpSampling2D()(x)
            x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)
            x = conv2d(self.n_filters_factor, (3, 3), activation='elu')(x)

        x = conv2d(orig_channels, (3, 3), activation=None)(x)

        return Model(z_input, x)

    def build_D(self):
        """
        Network Architecture based on the one presented in infoGAN
        """
        x_input = Input(shape=self.input_shape)

        self.encoder = self.build_encoder()
        x_embedding = self.encoder(x_input)

        if self.use_alignment_layer:
            z_input = Input(shape=(self.embedding_size,))
            z = Dense(self.embedding_size, use_bias=False)(z_input)
            self.alignment_layer = Model(z_input, z)

        self.d_classifier = self.build_d_classifier()
        discriminator = self.d_classifier(x_embedding)

        self.decoder = self.build_decoder()
        x_hat = self.decoder(x_embedding)

        return Model(x_input, [x_embedding, discriminator, x_hat])

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
        np.random.seed()

        generated_images = self.f_Gx.predict(samples, batch_size=self.batchsize)
        images_from_set, _ = self.dataset.get_random_fixed_batch(n)

        self.save_precomputed_features('generated_and_real_samples', generated_images, Y=images_from_set)
        return images_from_set, generated_images

    def compute_triplet_distance_vectors(self, n=5000):
        np.random.seed(14)
        samples = np.random.normal(size=(n, self.z_dims))
        np.random.seed()

        generated_images = self.f_Gx.predict(samples, batch_size=self.batchsize)
        images_from_set, _ = self.dataset.get_random_fixed_batch(n)

        encoded_x_hat = self.encoder.predict(generated_images)
        encoded_x = self.encoder.predict(images_from_set)
        encoded_x_r = sk.utils.shuffle(encoded_x)

        d_p = np.sqrt(np.maximum(np.finfo(float).eps, np.sum(np.square(encoded_x - encoded_x_r), axis=1)))
        d_n = np.sqrt(np.maximum(np.finfo(float).eps, np.sum(np.square(encoded_x - encoded_x_hat), axis=1)))        
        triplet = d_n - d_p

        self.save_precomputed_features('triplet_distance_vectors', triplet)
        return triplet

    def compute_generated_image_samples(self, n=36):
        np.random.seed(14)
        samples = np.random.normal(size=(n, self.z_dims))
        np.random.seed()

        generated_images = self.f_Gx.predict(samples, batch_size=n)
        return generated_images

    def compute_reconstruction_samples(self, n=18):
        if self.dataset.has_test_set():
            imgs_from_dataset, _ = self.dataset.get_random_perm_of_test_set(n)
        else:
            imgs_from_dataset, _ = self.dataset.get_random_fixed_batch(n)
        np.random.seed(14)
        noise = np.random.normal(scale=self.input_noise, size=imgs_from_dataset.shape)
        np.random.seed()
        imgs_from_dataset += noise
        encoding = self.encoder.predict(imgs_from_dataset)
        x_hat = self.decoder.predict(encoding)
        return imgs_from_dataset, x_hat


class TOPGANwithAESmall(TOPGANwithAEfromBEGAN):
    name = 'topgan-ae-small'

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

    def build_d_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Activation('relu')(embedding_input)
        x = BatchNormalization()(x)
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

        x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = deconv2d(orig_channels, (4, 4), strides=(2, 2), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)


class TOPGANwithAEfromDCGAN(TOPGANwithAEfromBEGAN):
    name = 'topgan-ae-dcgan'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = conv2d(self.n_filters_factor, (5, 5), strides=(2, 2), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(inputs)
        x = conv2d(self.n_filters_factor * 2, (5, 5), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(x)
        x = conv2d(self.n_filters_factor * 4, (5, 5), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(x)
        x = conv2d(self.n_filters_factor * 8, (5, 5), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(x)

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

    def build_d_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Dense(1)(embedding_input)
        x = Activation('sigmoid')(x)

        return Model(embedding_input, x)

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


class TOPGANwithAEfromConv(TOPGANwithAEfromBEGAN):
    name = 'topgan-ae-conv'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = conv2d(self.n_filters_factor, (5, 5), strides=(2, 2), bnorm=False, activation='relu', padding='same')(inputs)
        x = conv2d(self.n_filters_factor * 2, (3, 3), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = conv2d(self.n_filters_factor * 2, (3, 3), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)

        x = Flatten()(x)
        x = Dense(self.embedding_size)(x)

        return Model(inputs, x)

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))
        orig_channels = self.input_shape[2]

        w = self.input_shape[0] // 2**3

        x = Dense(w * w)(z_input)
        x = Reshape((w, w, 1))(x)
        x = UpSampling2D((4, 4))(x)
        x = conv2d(self.n_filters_factor * 2, (3, 3), bnorm=False, activation='relu', padding='same')(x)
        x = conv2d(self.n_filters_factor * 2, (3, 3), bnorm=False, activation='relu', padding='same')(x)
        x = UpSampling2D()(x)
        x = conv2d(self.n_filters_factor, (3, 3), bnorm=False, activation='relu', padding='same')(x)

        x = conv2d(orig_channels, (3, 3), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)

    def build_d_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Dense(1)(embedding_input)
        x = Activation('sigmoid')(x)

        return Model(embedding_input, x)

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
        orig_channels = self.input_shape[2]

        w = self.input_shape[0] // 2**3

        x = Dense(w * w)(z_input)
        x = Reshape((w, w, 1))(x)
        x = UpSampling2D((4, 4))(x)
        x = conv2d(self.n_filters_factor * 2, (3, 3), bnorm=False, activation='relu', padding='same')(x)
        x = conv2d(self.n_filters_factor * 2, (3, 3), bnorm=False, activation='relu', padding='same')(x)
        x = UpSampling2D()(x)
        x = conv2d(self.n_filters_factor, (3, 3), bnorm=False, activation='relu', padding='same')(x)

        x = conv2d(orig_channels, (3, 3), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)


class TOPGANwithAESmall2(TOPGANwithAEfromBEGAN):
    name = 'topgan-ae-small2'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = conv2d(32, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(inputs)
        x = Flatten()(x)
        x = Dense(self.embedding_size)(x)
        x = Activation('linear')(x)

        return Model(inputs, x)

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))
        orig_channels = self.input_shape[2]

        w = 8  # starting width
        x = Dense(8 * w * w)(z_input)
        x = Activation('relu')(x)
        x = Reshape((w, w, 8))(x)

        if self.input_shape[0] >= 64:
            x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        if self.input_shape[0] >= 128:
            x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = conv2d(orig_channels, (3, 3), strides=(1, 1), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)

    def build_d_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Activation('relu')(embedding_input)
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(embedding_input, x)

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
        orig_channels = self.input_shape[2]

        w = 8  # starting width
        x = Dense(8 * w * w)(z_input)
        x = Activation('relu')(x)
        x = Reshape((w, w, 8))(x)

        if self.input_shape[0] >= 64:
            x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        if self.input_shape[0] >= 128:
            x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = conv2d(orig_channels, (3, 3), strides=(1, 1), bnorm=False, activation=None, padding='same')(x)

        return Model(z_input, x)


class TOPGANwithAESmall3(TOPGANwithAEfromBEGAN):
    name = 'topgan-ae-small2-bn'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        if self.input_shape[0] == 32:
            x = conv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(inputs)
        elif self.input_shape[0] == 64:
            x = conv2d(64, (4, 4), strides=(4, 4), bnorm=False, activation='relu')(inputs)
        elif self.input_shape[0] == 128:
            x = conv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(inputs)
            x = conv2d(64, (4, 4), strides=(4, 4), bnorm=False, activation='relu')(inputs)
        x = Flatten()(x)
        x = Dense(self.embedding_size)(x)
        x = Activation('linear')(x)

        return Model(inputs, x)

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))
        orig_channels = self.input_shape[2]

        w = 8  # starting width
        x = Dense(8 * w * w)(z_input)
        x = Activation('relu')(x)
        x = Reshape((w, w, 8))(x)

        if self.input_shape[0] >= 64:
            x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        if self.input_shape[0] >= 128:
            x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = conv2d(orig_channels, (3, 3), strides=(1, 1), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)

    def build_d_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Activation('relu')(embedding_input)
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(embedding_input, x)

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
        orig_channels = self.input_shape[2]

        w = 8  # starting width
        x = Dense(8 * w * w)(z_input)
        x = Activation('relu')(x)
        x = BatchNormalization()(x)
        x = Reshape((w, w, 8))(x)

        if self.input_shape[0] >= 64:
            x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
        if self.input_shape[0] >= 128:
            x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
        x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = deconv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = conv2d(orig_channels, (3, 3), strides=(1, 1), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)


class TOPGANwithAEmlp(TOPGANwithAEfromBEGAN):
    name = 'topgan-ae-mlp'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = Flatten()(inputs)
        x = Dense(512)(x)
        x = Activation('relu')(x)
        x = Dense(self.embedding_size)(x)
        x = Activation('linear')(x)

        return Model(inputs, x)

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))

        x = Dense(512)(z_input)
        x = Activation('relu')(x)
        x = Dense(np.prod(self.input_shape))(x)
        x = Activation('sigmoid')(x)
        x = Reshape(self.input_shape)(x)

        return Model(z_input, x)

    def build_d_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Activation('relu')(embedding_input)
        x = Dense(64)(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(embedding_input, x)

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))

        x = Dense(512)(z_input)
        x = Activation('relu')(x)
        x = Dense(np.prod(self.input_shape))(x)
        x = Activation('sigmoid')(x)
        x = Reshape(self.input_shape)(x)

        return Model(z_input, x)


class TOPGANwithAESmallLN(TOPGANwithAEfromBEGAN):
    name = 'topgan-ae-small-ln'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = conv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(inputs)
        x = conv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(inputs)
        x = Flatten()(x)
        x = Dense(self.embedding_size)(x)
        x = Activation('linear')(x)

        return Model(inputs, x)

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))
        orig_channels = self.input_shape[2]

        w = self.input_shape[0] // 2**2  # starting width
        x = Dense(128 * w * w)(z_input)
        x = LayerNorm()(x)
        x = Activation('relu')(x)
        x = Reshape((w, w, 128))(x)

        x = deconv2d(64, (4, 4), strides=(2, 2), lnorm=True, activation='relu', padding='same')(x)
        x = deconv2d(orig_channels, (4, 4), strides=(2, 2), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)

    def build_d_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Activation('relu')(embedding_input)
        x = LayerNorm()(x)
        x = Dense(64)(x)
        x = LayerNorm()(x)
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(embedding_input, x)

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
        orig_channels = self.input_shape[2]

        w = self.input_shape[0] // 2**2  # starting width
        x = Dense(1024)(z_input)
        x = LayerNorm()(x)
        x = Activation('relu')(x)
        x = Dense(128 * w * w)(z_input)
        x = LayerNorm()(x)
        x = Activation('relu')(x)
        x = Reshape((w, w, 128))(x)

        x = deconv2d(64, (4, 4), strides=(2, 2), lnorm=True, activation='relu', padding='same')(x)
        x = deconv2d(orig_channels, (4, 4), strides=(2, 2), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)

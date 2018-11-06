from pprint import pprint

import numpy as np
import sklearn as sk

from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Add,
                          Lambda, Conv1D, UpSampling2D)
from keras.optimizers import Adam, SGD
from keras import backend as K, regularizers

from core.models import BaseModel
from core.lossfuns import (triplet_lossfun_creator, discriminator_lossfun,
                           generator_lossfun, triplet_balance_creator,
                           eq_triplet_lossfun_creator,
                           triplet_std_creator, generic_triplet_lossfun_creator,
                           topgan_began_dis_lossfun, topgan_began_gen_lossfun,
                           ae_lossfun, quadruplet_lossfun_creator)

from .layers import (conv2d, deconv2d, LayerNorm, squared_pairwise_distance,
                     k_largest_indexes, print_tensor_shape)
from .utils import (set_trainable, smooth_binary_labels)


class TOPGANwithAEfromBEGAN(BaseModel):
    name = 'topgan-ae-began'

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
                 use_magan_equilibrium=False,
                 topgan_enforce_std_dev=False,
                 topgan_use_data_trilet_regularization=False,
                 use_began_loss=False,
                 use_gradnorm=False,
                 gradnorm_alpha=0.5,
                 distance_metric='l2',
                 use_sigmoid_triplet=False,
                 online_mining=None,
                 online_mining_ratio=1,
                 use_quadruplet=False,
                 generator_mining=False,
                 **kwargs):

        self.loss_names = ['g_loss', 'd_loss', 'd_triplet',
                           'g_triplet', 'g_std', 'g_mean']
        self.loss_plot_organization = [('g_loss', 'd_loss'), 'd_triplet',
                                       'g_triplet', ('g_std', 'g_mean')]
        if use_magan_equilibrium:
            self.loss_names += ['margin']
            self.loss_plot_organization += ['margin']
        if use_alignment_layer or use_began_equilibrium:
            self.loss_names += ['k']
            self.loss_plot_organization += ['k']
        if topgan_use_data_trilet_regularization:
            self.loss_names += ['data_triplet']
            self.loss_plot_organization += ['data_triplet']
        if use_began_loss:
            self.loss_names += ['began_d', 'began_g']
            self.loss_plot_organization += [('began_d', 'began_g')]
        else:
            self.loss_names += ['ae_loss']
            self.loss_plot_organization += ['ae_loss']
        if use_gradnorm:
            self.loss_names += ['gradnorm']
            self.loss_plot_organization += ['gradnorm']

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
        self.enforce_std_dev = topgan_enforce_std_dev
        self.use_data_trilet_regularization = topgan_use_data_trilet_regularization
        self.use_began_loss = use_began_loss
        self.use_gradnorm = use_gradnorm
        self.gradnorm_trainer = None
        self.gradnorm_alpha = gradnorm_alpha
        self.distance_metric = distance_metric
        self.use_sigmoid_triplet = use_sigmoid_triplet
        self.use_quadruplet = use_quadruplet
        self.generator_mining = generator_mining

        self.online_mining = online_mining
        self.online_mining_ratio = online_mining_ratio
        self.mining_model = None

        if self.use_magan_equilibrium:
            self.gamma = 1.

        pprint(vars(self))

        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(self.batchsize, self.z_dims))
        x_permutation = np.array(np.random.permutation(self.batchsize), dtype='int64')

        if self.mining_model is not None:
            triplets = self.mining_model.predict([x_data, z_latent_dis], batch_size=self.batchsize // self.online_mining_ratio)
            x_data = x_data[triplets[:, 0]]
            z_latent_dis = z_latent_dis[triplets[:, 1]]

            if self.online_mining == 'pairs-negative' or self.online_mining == 'negative':
                x_permutation = np.array(np.random.permutation(len(x_data)), dtype='int64')
            elif self.online_mining == 'pairs-both' or self.online_mining == 'both':
                x_permutation = triplets[:, 2]

        batchsize = len(x_data)

        # perform label smoothing if applicable
        y_pos, y_neg = smooth_binary_labels(batchsize, self.label_smoothing, one_sided_smoothing=False)
        y = np.stack((y_neg, y_pos), axis=1)

        if self.input_noise > 1e-5:
            noise = np.random.normal(scale=self.input_noise, size=x_data.shape)
        else:
            noise = np.zeros(x_data.shape)

        dummy_y = np.expand_dims(y, axis=-1)
        input_data = [x_data, noise, x_permutation, z_latent_dis]
        label_data = [y, y, dummy_y, y, y, dummy_y, dummy_y]

        # train both networks
        ld = {}  # loss dictionary
        _, ld['g_loss'], ld['g_triplet'], _, ld['g_mean'], ld['g_std'], ld['data_triplet'], ld['began_g'] = self.gen_trainer.train_on_batch(input_data, label_data)
        _, ld['d_loss'], ld['d_triplet'], ld['ae_loss'], _, _, _, ld['began_d'] = self.dis_trainer.train_on_batch(input_data, label_data)
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
        if self.use_gradnorm:
            if self.gradnorm_trainer is None:
                self.gradnorm_trainer = self.build_grad_norm_trainer(
                    initial_losses=[ld['d_loss'], ld['d_triplet'],
                                    ld['ae_loss'], ld['d_triplet'],
                                    ld['g_std'], ld['data_triplet'], ld['began_d']])
                # ld['began_d'], ld['g_loss'],
                # ld['g_triplet'], ld['ae_loss'],
                # ld['g_mean'], ld['g_std'],
                # ld['data_triplet'], ld['began_g']])
            input_data_gn = [x_data, noise, np.expand_dims(x_permutation, axis=-1), z_latent_dis] + label_data + [np.ones(len(y)) for _ in range(len(label_data))]
            ld['gradnorm'], = self.gradnorm_trainer(input_data_gn + input_data_gn)

        return ld

    def build_online_mining_model(self):
        input_x = Input(shape=self.input_shape)
        input_z = Input(shape=(self.z_dims,))

        x_hat = self.f_Gx(input_z)
        negative_embedding, _, _ = self.f_D(x_hat)
        anchor_embedding, _, _ = self.f_D(input_x)

        n = self.batchsize // self.online_mining_ratio
        d_n = squared_pairwise_distance()([anchor_embedding, negative_embedding])
        d_p = squared_pairwise_distance()([anchor_embedding, anchor_embedding])

        if self.generator_mining:
            signal = 1
        else:
            signal = -1

        if self.online_mining == 'pairs-both' or 'pairs-negative':

            # we first mine negative values
            k_d_n = Lambda(lambda x: k_largest_indexes(k=self.batchsize // self.online_mining_ratio, signal=signal)(x))(d_n)

            # then, for each positive pair, we mine its positive counterpart
            k_d_p = k_largest_indexes(k=1, idx_dims=1, signal=-signal)(d_p)
            positive_k_idx = Lambda(lambda x: K.gather(x, k_d_n[:, 0]))(k_d_p)

            triplets = Concatenate(axis=-1, name="aligned_triplet")([k_d_n, positive_k_idx])

        elif self.online_mining == 'both' or self.online_mining == 'negative':
            k_d_a = K.tf.range(0, n)
            k_d_n = k_largest_indexes(k=1, idx_dims=1, signal=signal)(d_n)[:n]
            k_d_p = k_largest_indexes(k=1, idx_dims=1, signal=-signal)(d_p)[:n]

            triplets = Concatenate(axis=-1, name="aligned_triplet")([k_d_a, k_d_n, k_d_p])

        return Model([input_x, input_z], triplets, name='mined_triplets')

    def build_trainer(self, use_quadruplet=False):
        input_x = Input(shape=self.input_shape)
        input_noise = Input(shape=self.input_shape)
        input_x_perm = Input(shape=(1,), dtype='int64')
        input_z = Input(shape=(self.z_dims,))

        clipping_layer = Lambda(lambda x: K.clip(x, 0., 1.))

        x_noisy = clipping_layer(Add()([input_x, input_noise]))
        x_hat = self.f_Gx(input_z)

        negative_embedding, p, x_hat_rec = self.f_D(x_hat)
        anchor_embedding, q, x_rec = self.f_D(input_x)
        shuffled_x = Lambda(lambda x: K.gather(x, input_x_perm))(input_x)
        positive_embedding = Lambda(lambda x: K.squeeze(K.gather(x, input_x_perm), 1))(anchor_embedding)
        _, _, x_rec_noise = self.f_D(x_noisy)

        if self.use_sigmoid_triplet:
            act = Activation('sigmoid')
            anchor_embedding, positive_embedding, negative_embedding = act(anchor_embedding), act(positive_embedding), act(negative_embedding)

        fix_dim = Reshape((-1, 1))

        input = [input_x, input_noise, input_x_perm, input_z]

        concatenated_dis = Concatenate(axis=-1, name="dis_classification")([p, q])
        concatenated_data_triplet = Concatenate(axis=-1, name="data_triplet")(
            [fix_dim(input_x), fix_dim(shuffled_x), fix_dim(x_hat)])
        concatenated_began_recons = Concatenate(axis=-1, name="began_ae")(
            [fix_dim(x_hat), fix_dim(x_hat_rec), fix_dim(x_rec), fix_dim(input_x)])
        concatenated_ae = Concatenate(axis=-1, name="ae")(
            [fix_dim(input_x), fix_dim(x_rec_noise)])

        if not use_quadruplet:
            triplet = Concatenate(axis=-1, name="triplet")(
                [anchor_embedding, positive_embedding, negative_embedding])
            metric_triplet = triplet
        else:
            nn_embedding = Lambda(lambda x: K.squeeze(K.gather(x, input_x_perm), 1))(negative_embedding)
            triplet = Concatenate(axis=-1, name="triplet")(
                [anchor_embedding, positive_embedding, negative_embedding, nn_embedding])
            metric_triplet = Concatenate(axis=-1, name="triplet_metrics")(
                [anchor_embedding, positive_embedding, negative_embedding])

        # if self.use_alignment_layer:
        #     aligned_negative_embedding = self.alignment_layer(negative_embedding)
        #     concatenated_aligned_triplet = Concatenate(axis=-1, name="aligned_triplet")(
        #         [anchor_embedding, positive_embedding, aligned_negative_embedding])
        #     output = [concatenated_dis, triplet, concatenated_ae,
        #               metric_triplet,
        #               metric_triplet, concatenated_data_triplet,
        #               concatenated_began_recons]
        # else:
        output = [concatenated_dis, triplet, concatenated_ae,
                  metric_triplet, metric_triplet,
                  concatenated_data_triplet, concatenated_began_recons]
        return Model(input, output, name='topgan')

    def build_model(self):

        self.f_Gx = self.build_Gx()  # Moriarty, the encoder
        self.f_D = self.build_D()   # Sherlock, the detective
        self.f_Gx.summary()
        self.encoder.summary()
        self.f_D.summary()

        self.optimizers = self.build_optmizers()
        self.k_gd_ratio = K.variable(0)
        loss_d, loss_g, triplet_d_loss, triplet_g_loss, t_mean, t_std, x_triplet = self.define_loss_functions()

        if self.enforce_std_dev:
            std_dev_weight = -0.01
        else:
            std_dev_weight = 0.

        if self.use_data_trilet_regularization:
            x_triplet_weight = self.losses['data_triplet'].backend
        else:
            x_triplet_weight = 0.

        if self.use_began_loss:
            began_loss_w = self.losses['began_d'].backend
            ae_loss = 0.
        else:
            ae_loss = self.losses['ae_loss'].backend
            began_loss_w = 0.

        if self.online_mining:
            self.mining_model = self.build_online_mining_model()

        if self.use_alignment_layer:
            self.alignment_layer.summary()
            self.alignment_layer_trainer = self.build_trainer()
            set_trainable(self.f_Gx, False)
            set_trainable(self.f_D, False)
            set_trainable(self.alignment_layer, True)
            self.alignment_layer_trainer.compile(
                optimizer=self.optimizers["opt_ae"],
                loss=[loss_d, triplet_d_loss, ae_lossfun, triplet_g_loss, t_std, x_triplet, topgan_began_dis_lossfun],
                loss_weights=[0, 0, 0, 1.])
            loss_weights = [self.losses['d_loss'].backend, 0, ae_loss, self.losses['d_triplet'].backend, std_dev_weight, 0., began_loss_w]
            set_trainable(self.alignment_layer, False)
        else:
            loss_weights = [self.losses['d_loss'].backend, self.losses['d_triplet'].backend, ae_loss, 0., std_dev_weight, 0., began_loss_w]
        self.dis_trainer = self.build_trainer()
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, True)
        self.dis_trainer.compile(optimizer=self.optimizers["opt_d"],
                                 loss=[loss_d, triplet_d_loss, ae_lossfun, triplet_d_loss, t_std, x_triplet, topgan_began_dis_lossfun],
                                 loss_weights=loss_weights)

        # build generators
        self.gen_trainer = self.build_trainer(use_quadruplet=self.use_quadruplet)
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=self.optimizers["opt_g"],
                                 loss=[loss_g, triplet_g_loss, ae_lossfun, t_mean, t_std, x_triplet, topgan_began_gen_lossfun],
                                 loss_weights=[self.losses['g_loss'].backend, self.losses['g_triplet'].backend, 0., 0., 0., x_triplet_weight, began_loss_w])

        # store trainers
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, True)

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr, beta_1=0.5)
        opt_g = Adam(lr=self.lr, beta_1=0.5)
        opt_ae = Adam(lr=self.lr)
        opt_gradnorm = SGD(lr=self.lr)
        return {"opt_d": opt_d,
                "opt_g": opt_g,
                "opt_ae": opt_ae,
                "opt_gradnorm": opt_gradnorm}

    def define_loss_functions(self):
        if self.use_began_equilibrium:
            triplet_d = eq_triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size, k=self.k_gd_ratio,
                                                   simplified=self.use_simplified_triplet, distance_metric=self.distance_metric)
        else:
            triplet_d = triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size, distance_metric=self.distance_metric)
        if self.use_quadruplet:
            triplet_g = quadruplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size, ttype='inverted', distance_metric=self.distance_metric)
        else:
            triplet_g = triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size, ttype='inverted', distance_metric=self.distance_metric)
        return (discriminator_lossfun, generator_lossfun,
                triplet_d, triplet_g,
                triplet_balance_creator(margin=self.triplet_margin, zdims=self.embedding_size, gamma=K.variable(self.gamma), distance_metric=self.distance_metric),
                triplet_std_creator(margin=self.triplet_margin, zdims=self.embedding_size, distance_metric=self.distance_metric),
                generic_triplet_lossfun_creator(margin=self.triplet_margin, ttype='inverted', distance_metric=self.distance_metric))

    def build_grad_norm_trainer(self, initial_losses=1.):
        individual_grad_norms = []
        loss_ratios = []
        shared_layer_weights = self.encoder.trainable_weights
        loss_tensors = self.dis_trainer.metrics_tensors  # + self.gen_trainer.metrics_tensors
        loss_w_tensors = self.dis_trainer.loss_weights  # + self.gen_trainer.loss_weights
        for l, w, ini in zip(loss_tensors, loss_w_tensors, initial_losses):
            if w != 0:
                last_w = w
                grads = K.tf.gradients(l, shared_layer_weights)
                flattened_g = K.concatenate([K.flatten(g) for g in grads])
                flattened_g = K.tf.where(K.tf.is_nan(flattened_g), K.tf.zeros_like(flattened_g), flattened_g)
                grad_norm = K.tf.norm(flattened_g)
                individual_grad_norms.append(grad_norm * w)
                loss_ratios.append(l / ini)
        k_grad_norms = K.stack(individual_grad_norms)
        k_loss_ratios = K.stack(loss_ratios)
        # k_grad_norms = K.tf.Print(k_grad_norms, [grad_norm, l, last_w, l / ini], message="[grad, l, w, ratio]:")

        alpha = self.gradnorm_alpha
        mean_norm = K.mean(k_grad_norms)
        mean_loss_ratios = K.mean(k_loss_ratios)
        inverse_train_rate = k_loss_ratios / mean_loss_ratios

        constant_mean = K.tf.stop_gradient(mean_norm)
        constant_mean = K.tf.stop_gradient(constant_mean)
        # constant_mean = K.tf.Print(constant_mean, [k_grad_norms, k_loss_ratios], message="v: ")
        nonnegativity = K.tf.nn.l2_loss(K.tf.nn.relu(K.tf.negative(self.losses["ae_loss"].backend - 0.1)))  # regularizer
        gradnorm_loss = K.mean(K.abs(k_grad_norms - constant_mean)) + 0.1 * nonnegativity
        opt = self.optimizers["opt_gradnorm"]
        trainable_weights = [self.losses["ae_loss"].backend]  # [l for l in self.dis_trainer.loss_weights if l != 0]
        updates = opt.get_updates(trainable_weights, [], gradnorm_loss)
        return K.function(self.dis_trainer._feed_inputs +
                          self.dis_trainer._feed_targets +
                          self.dis_trainer._feed_sample_weights +
                          self.gen_trainer._feed_inputs +
                          self.gen_trainer._feed_targets +
                          self.gen_trainer._feed_sample_weights,
                          [gradnorm_loss], updates=updates)

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

        return Model(x_input, [x_embedding, discriminator, x_hat], name="D")

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
        return generated_images, images_from_set

    def compute_generated_samples_and_possible_labels(self, n=10000):
        np.random.seed(14)
        samples = np.random.normal(size=(n, self.z_dims))
        np.random.seed()

        generated_images = self.f_Gx.predict(samples, batch_size=self.batchsize)

        self.save_precomputed_features('generated_samples_and_possible_labels', generated_images, Y=self.dataset.attr_names)
        return generated_images, self.dataset.attr_names

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

        if self.input_shape[0] == 32:
            x = conv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(inputs)
        elif self.input_shape[0] == 64:
            x = conv2d(64, (4, 4), strides=(4, 4), bnorm=False, activation='relu')(inputs)
        elif self.input_shape[0] == 128:
            x = conv2d(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(inputs)
            x = conv2d(64, (4, 4), strides=(4, 4), bnorm=False, activation='relu')(x)
        x = conv2d(8, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(self.embedding_size)(x)

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
        x = conv2d(orig_channels, (3, 3), strides=(1, 1), bnorm=False, activation=None, padding='same')(x)

        return Model(z_input, x)

    def build_d_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Activation('relu')(embedding_input)
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
            x = conv2d(64, (4, 4), strides=(4, 4), bnorm=False, activation='relu')(x)
        x = conv2d(8, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(x)
        x = Dense(self.embedding_size)(x)

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
        x = conv2d(orig_channels, (3, 3), strides=(1, 1), bnorm=False, activation=None, padding='same')(x)

        return Model(z_input, x)

    def build_d_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Activation('relu')(embedding_input)
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


class TOPGANwithAEmlpSynth(TOPGANwithAEfromBEGAN):
    name = 'topgan-ae-mlp-synth'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = Dense(512, activity_regularizer=lambda x: K.maximum(0., 1e-6 - K.std(x)))(inputs)
        x = Activation('relu')(x)
        x = Dense(512, activity_regularizer=lambda x: K.maximum(0., 1e-6 - K.std(x)))(x)

        x = Activation('relu')(x)
        x = Dense(self.embedding_size)(x)
        # x = Dense(self.embedding_size, kernel_regularizer=regularizers.l2(0.01))(x)
        # x = Dense(self.embedding_size)(x)
        # x = Activation('relu')(x)

        return Model(inputs, x, name='encoder')

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))

        x = Dense(512)(z_input)
        x = Activation('relu')(x)
        x = Dense(512)(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(self.input_shape))(x)

        return Model(z_input, x, name='decoder')

    def build_d_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Activation('relu')(embedding_input)
        x = Dense(512)(x)
        x = Dense(1)(x)

        return Model(embedding_input, x, name='clas')

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))

        x = Dense(512)(z_input)
        x = Activation('relu')(x)
        x = Dense(512)(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(self.input_shape))(x)

        return Model(z_input, x, name='G')


class TOPGANwithAEmlpSynthNoReg(TOPGANwithAEfromBEGAN):
    name = 'topgan-ae-mlp-synth-noreg'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = Dense(512)(inputs)
        x = Activation('relu')(x)
        x = Dense(512)(x)

        x = Activation('relu')(x)
        x = Dense(self.embedding_size)(x)

        return Model(inputs, x, name='encoder')

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))

        x = Dense(512)(z_input)
        x = Activation('relu')(x)
        x = Dense(512)(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(self.input_shape))(x)

        return Model(z_input, x, name='decoder')

    def build_d_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Activation('relu')(embedding_input)
        x = Dense(512)(x)
        x = Dense(1)(x)

        return Model(embedding_input, x, name='clas')

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))

        x = Dense(512)(z_input)
        x = Activation('relu')(x)
        x = Dense(512)(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(self.input_shape))(x)

        return Model(z_input, x, name='G')


class TOPGANwithAEmlpSynthNoRegVeegan(TOPGANwithAEfromBEGAN):
    name = 'topgan-ae-mlp-synth-veegan'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = Dense(128)(inputs)
        x = Activation('relu')(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dense(self.embedding_size)(x)

        return Model(inputs, x, name='encoder')

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))

        x = Dense(128)(z_input)
        x = Activation('relu')(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(self.input_shape))(x)

        return Model(z_input, x, name='decoder')

    def build_d_classifier(self):
        embedding_input = Input(shape=(self.embedding_size,))

        x = Activation('relu')(embedding_input)
        x = Dense(128)(x)
        x = Dense(1)(x)

        return Model(embedding_input, x, name='clas')

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))

        x = Dense(128)(z_input)
        x = Activation('relu')(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(self.input_shape))(x)

        return Model(z_input, x, name='G')


def regfun(x):
    tmp = K.maximum(0., 1e-6 - K.std(K.batch_flatten(x), axis=0))
    # tmp = K.tf.Print(tmp, [tmp, K.shape(tmp)], "")
    return K.mean(tmp)


class TOPGANwithAEfromDCGANRegularized(TOPGANwithAEfromBEGAN):
    name = 'topgan-ae-dcgan-reg'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = conv2d(self.n_filters_factor, (5, 5), strides=(2, 2), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(inputs)
        x = conv2d(self.n_filters_factor * 2, (5, 5), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(x)
        x = conv2d(self.n_filters_factor * 4, (5, 5), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(x)
        x = conv2d(self.n_filters_factor * 8, (5, 5), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2, padding='same', reg=regfun)(x)

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

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
                          Lambda, Conv1D, UpSampling2D)
from keras.optimizers import Adam, SGD
from keras import backend as K, regularizers

from core.models import BaseModel
from core.lossfuns import (discriminator_lossfun, generator_lossfun,
                           loss_is_output, magnetgan_ae_a, magnetgan_ae_b,
                           magnetgan_ae_a_dis, magnetgan_ae_b_dis,
                           magnetgan_ae_a_gen, magnetgan_ae_b_gen)

from .layers import (conv2d, deconv2d, LayerNorm, squared_pairwise_distance,
                     k_largest_indexes, print_tensor_shape, rdeconv, res)
from .utils import (set_trainable, smooth_binary_labels)


class MagnetGANwithAEfromBEGAN(BaseModel):
    name = 'magnetgan-began'

    def __init__(self,
                 input_shape=(64, 64, 3),
                 embedding_dim=128,
                 triplet_margin=1.,
                 n_filters_factor=128,
                 use_began_loss=False,
                 use_gradnorm=False,
                 gradnorm_alpha=0.5,
                 distance_metric='l2',
                 use_uniform_z=False,
                 multilayer_triplet=False,
                 minibatch_size=32,
                 n_clusters=10,
                 n_clusters_to_select=5,
                 refresh_clusters=5,
                 magnet_type='normal',
                 magnet_polarity='real',
                 **kwargs):

        self.loss_names = ['g_loss', 'd_loss', 'd_magnet',
                           'g_magnet']
        self.loss_plot_organization = [('g_loss', 'd_loss'), 'd_magnet',
                                       'g_magnet']

        if use_began_loss:
            self.loss_names += ['ae_loss', 'ae_fake']
            self.loss_plot_organization += ['ae_loss', 'ae_fake']
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
        self.use_began_loss = use_began_loss

        self.did_set_g_triplet_count = 0

        self.use_gradnorm = use_gradnorm
        self.gradnorm_trainer = None
        self.gradnorm_alpha = gradnorm_alpha
        self.distance_metric = distance_metric

        self.mining_model = None
        self.use_uniform_z = use_uniform_z
        self.k_gd_ratio = K.variable(0)
        self.multilayer_triplet = multilayer_triplet
        self.triplet_bound_layers = []
        self.minibatch_size = minibatch_size
        self.n_clusters = n_clusters
        self.n_clusters_to_select = n_clusters_to_select
        self.refresh_clusters_rate = refresh_clusters
        self.magnet_type = magnet_type
        self.magnet_polarity = magnet_polarity

        pprint(vars(self))

        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        # break into smaller batches and loop to get representations for entire batch
        n_data = len(x_data)
        real_e = np.zeros((n_data, self.embedding_size))
        fake_e = np.zeros((n_data, self.embedding_size))
        z_latent = np.zeros((n_data, self.z_dims))
        for b in range(0, n_data, self.minibatch_size):
            if self.minibatch_size > n_data - b:
                continue
            b_end = b + self.minibatch_size

            # get embedding for real samples
            x = x_data[b:b_end, ...]
            real_e[b:b_end, :] = self.encoder.predict(x)

            # get fake samples end their embedding
            if self.use_uniform_z:
                z_latent[b:b_end, :] = np.random.uniform(low=-1.0, high=1.0, size=(self.minibatch_size, self.z_dims))
            else:
                z_latent[b:b_end, :] = np.random.normal(size=(self.minibatch_size, self.z_dims))
            z = z_latent[b:b_end, :]
            fake_e[b:b_end, :] = self.encoder.predict(self.f_Gx.predict(z))

        # run k-means on both real and fake representations
        kmeans_r = skcluster.KMeans(n_clusters=self.n_clusters, init='k-means++')
        kmeans_f = skcluster.KMeans(n_clusters=self.n_clusters, init='k-means++')
        clusters_real = kmeans_r.fit_predict(real_e)
        clusters_fake = kmeans_f.fit_predict(fake_e)

        # prepare labels for discriminator
        y_pos, y_neg = smooth_binary_labels(self.minibatch_size, self.label_smoothing, one_sided_smoothing=False)
        y_d, y_g = {}, {}
        y_d['r'] = y_pos
        y_d['f'] = y_neg
        y_g['r'] = y_neg
        y_g['f'] = y_pos
        y_dummy = np.zeros((self.minibatch_size,))
        y_dummy_ae = np.expand_dims(np.expand_dims(y_dummy, axis=-1), axis=-1)

        for _ in range(self.refresh_clusters_rate):

            ld = {}
            # select a class and anchor cluster
            for anchor_is_real in [True, False]:
                a_cluster = np.random.choice(self.n_clusters)
                if anchor_is_real:
                    a_data = x_data
                    a_labels = clusters_real
                    n_data = z_latent
                    n_labels = clusters_fake
                    a_centers = kmeans_r.cluster_centers_
                    n_centers = kmeans_f.cluster_centers_
                    key = 'r'
                else:
                    a_data = z_latent
                    a_labels = clusters_fake
                    n_data = x_data
                    n_labels = clusters_real
                    a_centers = kmeans_f.cluster_centers_
                    n_centers = kmeans_r.cluster_centers_
                    key = 'f'
                a_cls_idx = a_labels == a_cluster

                # get closest negative clusters (by looking at centers)
                if self.magnet_polarity == 'real':
                    sel_n_cls = np.argsort(np.linalg.norm(n_centers - a_centers[a_cluster], axis=1))[-self.n_clusters_to_select:]
                elif self.magnet_polarity == 'fake':
                    sel_n_cls = np.argsort(np.linalg.norm(n_centers - a_centers[a_cluster], axis=1))[:self.n_clusters_to_select]
                elif self.magnet_polarity == 'both':
                    tmp_sort = np.argsort(np.linalg.norm(n_centers - a_centers[a_cluster], axis=1))
                    sel_n_cls = np.concatenate((tmp_sort[:self.n_clusters_to_select // 2], tmp_sort[-self.n_clusters_to_select // 2:]))

                # select a minibatch from the anchor and each of the negative clusters
                a_samples = a_data[a_cls_idx]
                a_samples = a_samples[np.random.choice(len(a_samples), self.minibatch_size)]
                n_samples = np.zeros([self.n_clusters_to_select, self.minibatch_size] + list(n_data.shape[1:]))
                for i, c in enumerate(sel_n_cls):
                    n_cls_idx = n_labels == c
                    tmp_samples = n_data[n_cls_idx]
                    n_samples[i, ...] = tmp_samples[np.random.choice(len(tmp_samples), self.minibatch_size)]

                n_samples = np.moveaxis(n_samples, 1, 0)

                # train both networks
                ldtemp = {}
                _, ldtemp['d_magnet'], _, ldtemp['d_loss'], ldtemp['ae_loss'] = self.dis_trainers[key].train_on_batch([a_samples, n_samples], [y_dummy, y_dummy, y_d[key], y_dummy_ae])
                _, l_, ldtemp['g_magnet'], ldtemp['g_loss'], ldtemp['ae_fake'] = self.gen_trainers[key].train_on_batch([a_samples, n_samples], [y_dummy, y_dummy, y_g[key], y_dummy_ae])

                for k, x in ldtemp.items():
                    ld[k] = ld.get(k, 0) + x

        return ld

    def build_trainer_for_real_anchor(self):
        input_x = Input(shape=self.input_shape)
        input_z = Input(shape=[self.n_clusters_to_select, self.z_dims])

        # fix the hacked shape
        n = Lambda(lambda x: K.permute_dimensions(x, (1, 0, 2)))(input_z)

        def generate_for_each_cluster(x):
            negs = [None] * self.n_clusters_to_select
            for i, _ in enumerate(negs):
                negs[i] = self.f_Gx(x[i, :])
            return Concatenate(axis=0)(negs)
        n = Lambda(generate_for_each_cluster)(n)
        a = Lambda(lambda x: x)(input_x)

        return Model([input_x, input_z], [a, n], name='model_real_anchor')

    def build_trainer_for_fake_anchor(self):
        input_z = Input(shape=(self.z_dims,))
        input_x = Input(shape=[self.n_clusters_to_select] + list(self.input_shape))

        # fix the hacked shape
        n = Lambda(lambda x: K.permute_dimensions(x, [1, 0] + list(range(2, len(self.input_shape) + 2))))(input_x)

        # generate images for each z
        a = self.f_Gx(input_z)

        return Model([input_z, input_x], [a, n], name='model_fake_anchor')

    def build_magnet_computer(self, input_a, input_n):

        emb_a, d_a, a_hat = self.f_D(input_a)

        simplified_n = Lambda(lambda x: K.reshape(x, [-1] + list(self.input_shape)))(input_n)
        emb_n_flat, d_n, n_hat = self.f_D(simplified_n)

        mean_a = Lambda(lambda x: K.mean(x, axis=0))(emb_a)
        emb_n = Lambda(lambda x: K.reshape(x, [self.n_clusters_to_select, self.minibatch_size, self.embedding_size]))(emb_n_flat)
        means_n = Lambda(lambda x: K.mean(x, axis=1))(emb_n)
        # emb_n = printt(emb_n, 'en')
        # emb_a = printt(emb_a, 'ea')

        # compute variance for each cluster and average them
        expand_dim = Lambda(lambda x: K.reshape(x, (1,)))
        var_a = Lambda(lambda x: K.var(x, axis=[0, 1]))(emb_a)
        var_n = Lambda(lambda x: K.var(x, axis=[1, 2]))(emb_n)
        variances = Concatenate(axis=0)([expand_dim(var_a), var_n])
        # variances = printt(variances, 'variances')
        variance = Lambda(lambda x: K.mean(x, axis=0))(variances)

        alpha = 1
        # variance = printt(variance, 'variance')
        # mean_a = printt(mean_a, 'mean_a')
        # means_n = printt(means_n, 'means_n')
        numerator = Lambda(lambda x: K.exp(-(1 / (2 * (variance**2))) * K.sum(K.square(x - mean_a), axis=1) - alpha))(emb_a)
        denominator = Lambda(lambda x: K.sum(K.exp(-(1 / (2 * (variance**2))) * K.sum(K.square(K.expand_dims(x, 1) - K.expand_dims(means_n, 0)), axis=-1)), axis=-1))(emb_a)
        # numerator = printt(numerator, 'nominator')
        # denominator = printt(denominator, 'denominator')
        epsilon = 1e-8
        magnet = Lambda(lambda x: K.maximum(0., -K.log(x / (denominator + epsilon) + epsilon)))(numerator)
        if self.magnet_type == 'normal':
            inverse_magnet = Lambda(lambda x: -K.log(denominator / (x + epsilon) + epsilon))(numerator)
        elif self.magnet_type == 'denominator':
            inverse_magnet = Lambda(lambda x: -K.log(x + epsilon))(denominator)
        elif self.magnet_type == 'negative':
            inverse_magnet = Lambda(lambda x: K.log(denominator / (x + epsilon) + epsilon))(numerator)
        elif self.magnet_type == 'contrastive':
            inverse_magnet = Lambda(lambda x: K.sum(K.square(K.expand_dims(emb_a, 0) - emb_n), axis=[-1, 0]))(emb_a)

        fix_dim = Reshape((-1, 1))
        remove_dim = Reshape((1,))
        slice_mb = Lambda(lambda x: x[:self.minibatch_size, ...])
        recons_concat = Concatenate(axis=-1, name="ae")(
            [fix_dim(input_a), fix_dim(a_hat), fix_dim(slice_mb(simplified_n)), fix_dim(slice_mb(n_hat))])
        # dis_concat = Concatenate(axis=0, name="dis_classification")([d_a, d_n])
        # magnet = printt(magnet, 'm')
        # inverse_magnet = printt(inverse_magnet, 'i')
        # d_a = printt(d_a, 'd')
        # recons_concat = printt(recons_concat, 'r')
        return remove_dim(magnet), remove_dim(inverse_magnet), remove_dim(d_a), recons_concat

    def build_trainer(self):
        a_input_x = Input(shape=self.input_shape, name='a_input_x')
        a_input_z = Input(shape=[self.n_clusters_to_select, self.z_dims], name='a_input_z')
        b_input_z = Input(shape=(self.z_dims,), name='b_input_z')
        b_input_x = Input(shape=[self.n_clusters_to_select] + list(self.input_shape), name='b_input_x')

        self.model_for_real_anchor = self.build_trainer_for_real_anchor()
        self.model_for_fake_anchor = self.build_trainer_for_fake_anchor()

        a_a, a_n = self.model_for_real_anchor([a_input_x, a_input_z])
        b_a, b_n = self.model_for_fake_anchor([b_input_z, b_input_x])

        a_magnet, a_inv_magnet, a_dis, a_recon = self.build_magnet_computer(a_a, a_n)
        b_magnet, b_inv_magnet, b_dis, b_recon = self.build_magnet_computer(b_a, b_n)

        model_a = Model([a_input_x, a_input_z], [a_magnet, a_inv_magnet, a_dis, a_recon], name='model_a')

        # we invert the second model magnets so that [0] always trains discriminator
        model_b = Model([b_input_z, b_input_x], [b_magnet, b_inv_magnet, b_dis, b_recon], name='model_b')

        return model_a, model_b

    def build_model(self):

        self.f_Gx = self.build_Gx()
        self.f_D = self.build_D()
        self.f_Gx.summary()
        self.encoder.summary()
        self.f_D.summary()

        self.optimizers = self.build_optmizers()
        self.k_gd_ratio = K.variable(0)
        (loss_d, loss_g, out_is_loss, ae_loss_a,
            ae_loss_b, ae_loss_a_gen, ae_loss_b_gen) = self.define_loss_functions()

        ae_loss_w = self.losses['ae_loss'].backend
        if self.use_began_loss:
            ae_loss_gen = self.losses['ae_fake'].backend
        else:
            ae_loss_gen = 0.

        self.dis_trainers = {}
        self.dis_trainers['r'], self.dis_trainers['f'] = self.build_trainer()
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, True)
        self.dis_trainers['r'].compile(
            optimizer=self.optimizers["opt_d"],
            loss=[out_is_loss, out_is_loss, 'binary_crossentropy', ae_loss_a],
            loss_weights=[self.losses['d_magnet'].backend, 0., self.losses['d_loss'].backend, ae_loss_w])
        self.dis_trainers['f'].compile(
            optimizer=self.optimizers["opt_d"],
            loss=[out_is_loss, out_is_loss, 'binary_crossentropy', ae_loss_b],
            loss_weights=[self.losses['d_magnet'].backend, 0., self.losses['d_loss'].backend, ae_loss_w])

        # build generators
        self.gen_trainers = {}
        self.gen_trainers['r'], self.gen_trainers['f'] = self.build_trainer()
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainers['r'].compile(
            optimizer=self.optimizers["opt_g"],
            loss=[out_is_loss, out_is_loss, 'binary_crossentropy', ae_loss_a_gen],
            loss_weights=[0., self.losses['g_magnet'].backend, self.losses['g_loss'].backend, ae_loss_gen])
        self.gen_trainers['f'].compile(
            optimizer=self.optimizers["opt_g"],
            loss=[out_is_loss, out_is_loss, 'binary_crossentropy', ae_loss_b_gen],
            loss_weights=[0., self.losses['g_magnet'].backend, self.losses['g_loss'].backend, ae_loss_gen])

        # store trainers
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, True)

        self.gen_trainers['r'].summary()

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr, beta_1=0.5)
        opt_g = Adam(lr=self.lr, beta_1=0.5)
        opt_ae = Adam(lr=self.lr)
        opt_gradnorm = Adam(lr=self.lr)
        return {"opt_d": opt_d,
                "opt_g": opt_g,
                "opt_ae": opt_ae,
                "opt_gradnorm": opt_gradnorm}

    def define_loss_functions(self):
        if self.use_began_loss:
            return (discriminator_lossfun, generator_lossfun,
                    loss_is_output, magnetgan_ae_a_dis, magnetgan_ae_b_dis,
                    magnetgan_ae_a_gen, magnetgan_ae_b_gen)
        else:
            return (discriminator_lossfun, generator_lossfun,
                    loss_is_output, magnetgan_ae_a, magnetgan_ae_b,
                    magnetgan_ae_a, magnetgan_ae_b)

    """
        # GradNorm Formulation and Training
    """

    def build_grad_norm_trainer(self, initial_losses=1.):
        individual_grad_norms = []
        shared_layer_weights = self.encoder.trainable_weights
        loss_tensors = self.dis_trainer.metrics_tensors + self.gen_trainer.metrics_tensors
        loss_w_tensors = self.dis_trainer.loss_weights + self.gen_trainer.loss_weights
        valued_w_tensors = []
        for l, w in zip(loss_tensors, loss_w_tensors):
            if w != 0:
                grads = K.tf.gradients(l, shared_layer_weights)
                flattened_g = K.concatenate([K.flatten(g) for g in grads])
                flattened_g = K.tf.where(K.tf.is_nan(flattened_g), K.tf.zeros_like(flattened_g), flattened_g)
                grad_norm = K.tf.norm(flattened_g)
                individual_grad_norms.append(grad_norm * w)
                valued_w_tensors.append(w)
        k_grad_norms = K.stack(individual_grad_norms)

        mean_norm = K.mean(k_grad_norms)
        constant_mean = K.tf.stop_gradient(mean_norm)
        constant_mean = K.tf.stop_gradient(constant_mean)

        nonnegativity = K.tf.nn.l2_loss(K.tf.nn.relu(K.tf.negative(valued_w_tensors)))  # regularizer
        gradnorm_loss = K.mean(K.abs(k_grad_norms - constant_mean)) + 0.1 * nonnegativity
        opt = self.optimizers["opt_gradnorm"]
        updates = opt.get_updates(valued_w_tensors, [], gradnorm_loss)
        return K.function(self.dis_trainer._feed_inputs +
                          self.dis_trainer._feed_targets +
                          self.dis_trainer._feed_sample_weights +
                          self.gen_trainer._feed_inputs +
                          self.gen_trainer._feed_targets +
                          self.gen_trainer._feed_sample_weights,
                          [gradnorm_loss], updates=updates)

    def build_loss_weights_adjustment_calculator(self):
        l_weights_tensors = self.dis_trainer.loss_weights + self.gen_trainer.loss_weights
        valued_w_tensors = [l for l in l_weights_tensors if l != 0]
        n = K.tf.cast(K.tf.count_nonzero(l_weights_tensors), dtype=K.tf.float32)
        w_sum = K.sum(valued_w_tensors)
        assignments = []
        for w in valued_w_tensors:
            assignment = K.tf.assign(w, w * (n / w_sum))
            assignments.append(assignment)
        assignments_k = K.stack(assignments)
        return K.function([], [assignments_k])

    def train_gradnorm(self, net_input):
        if self.gradnorm_trainer is None:
            self.gradnorm_trainer = self.build_grad_norm_trainer()
            self.gradnorm_adjustment = self.build_loss_weights_adjustment_calculator()
        gradnorm_loss, = self.gradnorm_trainer(net_input + net_input)
        self.gradnorm_adjustment([])
        if self.use_began_loss:
            self.losses['began_d'].adjust_base_weight(K.get_value(self.losses['began_d'].backend))
            if not self.tie_loss_weights:
                self.losses['began_g'].adjust_base_weight(K.get_value(self.losses['began_g'].backend))
        d_triplet = K.get_value(self.losses['d_triplet'].backend)
        self.losses['d_triplet'].adjust_base_weight(d_triplet)
        if not self.tie_loss_weights:
            self.losses['g_triplet'].adjust_base_weight(K.get_value(self.losses['g_triplet'].backend))
        return gradnorm_loss

    """
        # Network Layers' Definition
    """

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

        x = conv2d(orig_channels, (3, 3), activation='sigmoid')(x)

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

        x = conv2d(orig_channels, (3, 3), activation='sigmoid')(x)

        return Model(z_input, x)

    def build_D(self):
        """
        Network Architecture based on the one presented in infoGAN
        """
        x_input = Input(shape=self.input_shape)

        self.encoder = self.build_encoder()
        x_embedding = self.encoder(x_input)

        if self.multilayer_triplet:
            ls = []
            for l in self.triplet_bound_layers:
                ls.append(l(x_input))
            t_embedding = Concatenate(axis=-1)([x_embedding] + ls)
        else:
            t_embedding = x_embedding

        self.d_classifier = self.build_d_classifier()
        discriminator = self.d_classifier(x_embedding)

        self.decoder = self.build_decoder()
        x_hat = self.decoder(x_embedding)

        return Model(x_input, [t_embedding, discriminator, x_hat], name="D")

    """
        # Computation of metrics inputs
    """

    def compute_classification_samples(self, n=26000):

        np.random.seed(14)
        samples = np.random.normal(size=(n, self.z_dims))
        np.random.seed()

        generated_images = self.f_Gx.predict(samples, batch_size=self.batchsize)
        images_from_set, _ = self.dataset.get_random_fixed_batch(32)

        model = keras.models.load_model('mnist_mode_count_classifier.h5')
        x_hat_pred = model.predict(generated_images, batch_size=64)
        x_pred = model.predict(images_from_set, batch_size=64)

        self.save_precomputed_features('classification_sample', x_hat_pred, Y=x_pred)
        return x_hat_pred, x_pred

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


class MagnetGANwithAESmall(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-small'

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


class MagnetGANwithAEfromDCGAN(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-dcgan'

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


class MagnetGANwithAEfromConv(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-conv'

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


class MagnetGANwithAESmall2(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-small2'

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
        x = conv2d(orig_channels, (3, 3), strides=(1, 1), bnorm=False, activation='sigmoid', padding='same')(x)

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
        x = conv2d(orig_channels, (3, 3), strides=(1, 1), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)


class MagnetGANwithAESmall3(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-small2-bn'

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


class MagnetGANwithAEmlp(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-mlp'

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


class MagnetGANwithAESmallLN(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-small-ln'

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


class MagnetGANwithAEmlpSynth(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-mlp-synth'

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


class MagnetGANwithAEmlpSynthNoReg(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-mlp-synth-noreg'

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


class MagnetGANwithAEmlpSynthNoRegVeegan(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-mlp-synth-veegan'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = Dense(128)(inputs)
        x = Activation('relu')(x)
        self.triplet_bound_layers.append(Model(inputs, x))
        x = Dense(128)(x)
        x = Activation('relu')(x)
        self.triplet_bound_layers.append(Model(inputs, x))
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
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(embedding_input, x, name='clas')

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))

        x = Dense(128)(z_input)
        x = Activation('relu')(x)
        x = Dense(128)(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(self.input_shape))(x)

        return Model(z_input, x, name='G')


class MagnetGANwithAEfromDCGANnobn(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-dcgan-nobn'

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
        x = deconv2d(self.n_filters_factor * 4, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = deconv2d(self.n_filters_factor * 2, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = deconv2d(self.n_filters_factor, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)

        x = deconv2d(orig_channels, (5, 5), strides=(2, 2), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)


class MagnetGANwithAEmlpSynthTest(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-mlp-synth-test'

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
        x = Activation('relu')(x)
        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        return Model(embedding_input, x, name='clas')

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))

        x = Dense(32)(z_input)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(32)(x)
        x = Activation('relu')(x)
        x = Dense(np.prod(self.input_shape))(x)

        return Model(z_input, x, name='G')


class MagnetGANwithAETesting(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-testing'

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
        x = rdeconv(self.n_filters_factor * 4, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = rdeconv(self.n_filters_factor * 2, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = rdeconv(self.n_filters_factor, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)

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
        x = rdeconv(self.n_filters_factor * 4, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = rdeconv(self.n_filters_factor * 2, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = rdeconv(self.n_filters_factor, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)

        x = deconv2d(orig_channels, (5, 5), strides=(2, 2), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)


class MagnetGANwithAETesting2(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-testing2'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        x = conv2d(self.n_filters_factor, (5, 5), strides=(2, 2), bnorm=False, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(inputs)
        self.triplet_bound_layers.append(Model(inputs, Flatten()(x)))
        x = res(self.n_filters_factor * 2, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(x)
        self.triplet_bound_layers.append(Model(inputs, Flatten()(x)))
        x = res(self.n_filters_factor * 4, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(x)
        self.triplet_bound_layers.append(Model(inputs, Flatten()(x)))
        x = res(self.n_filters_factor * 8, (3, 3), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.2, padding='same')(x)
        self.triplet_bound_layers.append(Model(inputs, Flatten()(x)))

        x = Flatten()(x)
        x = Dense(self.embedding_size)(x)

        return Model(inputs, x)

    def build_decoder(self):
        z_input = Input(shape=(self.embedding_size,))
        orig_channels = self.input_shape[2]

        w = self.input_shape[0] // 2**4

        x = Dense(self.n_filters_factor * 8 * w * w)(z_input)
        x = Reshape((w, w, self.n_filters_factor * 8))(x)
        x = rdeconv(self.n_filters_factor * 4, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = rdeconv(self.n_filters_factor * 2, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = rdeconv(self.n_filters_factor, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)

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
        x = rdeconv(self.n_filters_factor * 4, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = rdeconv(self.n_filters_factor * 2, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)
        x = rdeconv(self.n_filters_factor, (5, 5), strides=(2, 2), bnorm=True, activation='relu', padding='same')(x)

        x = deconv2d(orig_channels, (5, 5), strides=(2, 2), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)


class MagnetGANwithAETesting3(MagnetGANwithAEfromBEGAN):
    name = 'magnetgan-testing3'

    def build_encoder(self):
        inputs = Input(shape=self.input_shape)

        if self.input_shape[0] == 32:
            x = res(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(inputs)
            self.triplet_bound_layers.append(Model(inputs, Flatten()(x)))
        elif self.input_shape[0] == 64:
            x = res(64, (4, 4), strides=(4, 4), bnorm=False, activation='relu')(inputs)
            self.triplet_bound_layers.append(Model(inputs, Flatten()(x)))
        elif self.input_shape[0] == 128:
            x = res(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(inputs)
            self.triplet_bound_layers.append(Model(inputs, Flatten()(x)))
            x = res(64, (4, 4), strides=(4, 4), bnorm=False, activation='relu')(x)
            self.triplet_bound_layers.append(Model(inputs, Flatten()(x)))
        x = res(8, (4, 4), strides=(2, 2), bnorm=False, activation='relu')(x)
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
            x = rdeconv(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        if self.input_shape[0] >= 128:
            x = rdeconv(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = rdeconv(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = rdeconv(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = conv2d(orig_channels, (3, 3), strides=(1, 1), bnorm=False, activation='sigmoid', padding='same')(x)

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
            x = rdeconv(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        if self.input_shape[0] >= 128:
            x = rdeconv(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = rdeconv(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = rdeconv(64, (4, 4), strides=(2, 2), bnorm=False, activation='relu', padding='same')(x)
        x = conv2d(orig_channels, (3, 3), strides=(1, 1), bnorm=False, activation='sigmoid', padding='same')(x)

        return Model(z_input, x)

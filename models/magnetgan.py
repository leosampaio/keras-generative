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
from core.lossfuns import (triplet_lossfun_creator, discriminator_lossfun,
                           generator_lossfun, triplet_balance_creator,
                           eq_triplet_lossfun_creator,
                           triplet_std_creator, generic_triplet_lossfun_creator,
                           topgan_began_dis_lossfun_creator,
                           topgan_began_gen_lossfun_creator,
                           ae_lossfun, quadruplet_lossfun_creator)

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
                 use_began_equilibrium=False,
                 use_gradnorm=False,
                 gradnorm_alpha=0.5,
                 distance_metric='l2',
                 use_uniform_z=False,
                 multilayer_triplet=False,
                 **kwargs):

        self.loss_names = ['g_loss', 'd_loss', 'd_triplet',
                           'g_triplet', 'g_std', 'g_mean']
        self.loss_plot_organization = [('g_loss', 'd_loss'), 'd_triplet',
                                       'g_triplet', ('g_std', 'g_mean')]

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

        pprint(vars(self))

        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        # get real latent variables distribution
        if self.use_uniform_z:
            z_latent_dis = np.random.uniform(low=-1.0, high=1.0, size=(self.batchsize, self.z_dims))
        else:
            z_latent_dis = np.random.normal(size=(self.batchsize, self.z_dims))
        x_permutation = np.array(np.random.permutation(self.batchsize), dtype='int64')

        if self.mining_model is not None:
            x_data, z_latent_dis, x_permutation = self.run_triplet_mining(x_data, z_latent_dis)

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
        label_data = [y, dummy_y, dummy_y, dummy_y, dummy_y]

        # train both networks
        ld = {}  # loss dictionary
        _, ld['d_loss'], ld['d_triplet'], ld['ae_loss'], _, _ = self.dis_trainer.train_on_batch(input_data, label_data)
        _, ld['g_loss'], ld['g_triplet'], _, ld['g_mean'], ld['g_std'] = self.gen_trainer.train_on_batch(input_data, label_data)

        if self.use_gradnorm:
            input_data_gn = [x_data, noise, np.expand_dims(x_permutation, axis=-1), z_latent_dis] + label_data + [np.ones(len(y)) for _ in range(len(label_data))]
            ld['gradnorm'] = self.train_gradnorm(net_input=input_data_gn)

        return ld

    def build_trainer(self):
        input_x = Input(shape=self.input_shape)
        input_noise = Input(shape=self.input_shape)
        input_x_perm = Input(shape=(1,), dtype='int64')
        input_z = Input(shape=(self.z_dims,))

        clipping_layer = Lambda(lambda x: K.clip(x, 0., 1.))

        x_noisy = clipping_layer(Add()([input_x, input_noise]))

        anchor_embedding, q, x_rec = self.f_D(input_x)
        positive_embedding = Lambda(lambda x: K.squeeze(K.gather(x, input_x_perm), 1))(anchor_embedding)
        _, _, x_rec_noise = self.f_D(x_noisy)

        x_hat = self.f_Gx(input_z)

        negative_embedding, p, x_hat_rec = self.f_D(x_hat)

        fix_dim = Reshape((-1, 1))

        input = [input_x, input_noise, input_x_perm, input_z]

        concatenated_dis = Concatenate(axis=-1, name="dis_classification")([p, q])
        concatenated_ae = Concatenate(axis=-1, name="ae")(
            [fix_dim(input_x), fix_dim(x_rec_noise)])
        
        nn_embedding = Lambda(lambda x: K.squeeze(K.gather(x, input_x_perm), 1))(negative_embedding)
        triplet = Concatenate(axis=-1, name="triplet")(
            [fix_dim(anchor_embedding), fix_dim(positive_embedding), fix_dim(negative_embedding), fix_dim(nn_embedding)])
        metric_triplet = Concatenate(axis=-1, name="triplet_metrics")(
            [fix_dim(anchor_embedding), fix_dim(positive_embedding), fix_dim(negative_embedding)])

        output = [concatenated_dis, triplet, concatenated_ae,
                  metric_triplet, metric_triplet]
        return Model(input, output, name='magnetgan')

    def build_model(self):

        self.f_Gx = self.build_Gx()
        self.f_D = self.build_D()
        self.f_Gx.summary()
        self.encoder.summary()
        self.f_D.summary()

        self.optimizers = self.build_optmizers()
        self.k_gd_ratio = K.variable(0)
        (loss_d, loss_g, triplet_d_loss, triplet_g_loss, t_mean, t_std) = self.define_loss_functions()

        ae_loss_w = self.losses['ae_loss'].backend

        self.dis_trainer = self.build_trainer()
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, True)
        self.dis_trainer.compile(optimizer=self.optimizers["opt_d"],
                                 loss=[loss_d, triplet_d_loss, ae_lossfun, triplet_d_loss, t_std],
                                 loss_weights=[self.losses['d_loss'].backend, self.losses['d_triplet'].backend, ae_loss_w, 0., 0.])

        # build generators
        self.gen_trainer = self.build_trainer()
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=self.optimizers["opt_g"],
                                 loss=[loss_g, triplet_g_loss, ae_lossfun, t_mean, t_std],
                                 loss_weights=[self.losses['g_loss'].backend, self.losses['d_triplet'].backend, 0., 0., 0.])

        # store trainers
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, True)

        self.gen_trainer.summary()

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
        triplet_d = triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size, distance_metric=self.distance_metric)
        triplet_g = triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.embedding_size, ttype='inverted', distance_metric=self.distance_metric)
        return (discriminator_lossfun, generator_lossfun,
                triplet_d, triplet_g,
                triplet_balance_creator(margin=self.triplet_margin, zdims=self.embedding_size, gamma=K.variable(0.5), distance_metric=self.distance_metric),
                triplet_std_creator(margin=self.triplet_margin, zdims=self.embedding_size, distance_metric=self.distance_metric))

    """
        # Triplet Mining Formulation and "Training" (actually just running)
    """

    def build_online_mining_model(self):
        input_x = Input(shape=self.input_shape)
        input_z = Input(shape=(self.z_dims,))

        anchor_embedding, _, _ = self.f_D(input_x)

        n = self.batchsize // self.online_mining_ratio

        if self.online_mining in ('pairs-both', 'pairs-negative', 'both', 'negative'):
            x_hat = self.f_Gx(input_z)
            negative_embedding, _, _ = self.f_D(x_hat)
            triplets = self.build_online_mining_model_literature_ver_(anchor_embedding, negative_embedding, n)
            return Model([input_x, input_z], triplets, name='mined_triplets')

        elif self.online_mining == 'anchor-cluster':
            return Model([input_x, input_z], anchor_embedding, name='mined_triplets')

    def build_online_mining_model_literature_ver_(self, a, n, batchsize):
        d_n = squared_pairwise_distance()([a, n])
        d_p = squared_pairwise_distance()([a, a])

        if self.generator_mining:
            signal = 1
        else:
            signal = -1

        if self.online_mining in ('pairs-both', 'pairs-negative'):

            # we first mine negative values
            k_d_n = Lambda(lambda x: k_largest_indexes(k=self.batchsize // self.online_mining_ratio, signal=signal)(x))(d_n)

            # then, for each positive pair, we mine its positive counterpart
            k_d_p = k_largest_indexes(k=1, idx_dims=1, signal=-signal)(d_p)
            positive_k_idx = Lambda(lambda x: K.gather(x, k_d_n[:, 0]))(k_d_p)

            triplets = Concatenate(axis=-1, name="aligned_triplet")([k_d_n, positive_k_idx])

        elif self.online_mining in ('both', 'negative'):
            k_d_a = K.tf.range(0, batchsize)
            k_d_n = k_largest_indexes(k=1, idx_dims=1, signal=signal)(d_n)[:batchsize]
            k_d_p = k_largest_indexes(k=1, idx_dims=1, signal=-signal)(d_p)[:batchsize]

            triplets = Concatenate(axis=-1, name="aligned_triplet")([k_d_a, k_d_n, k_d_p])
        return triplets

    def run_triplet_mining(self, x_data, z_latent_dis):
        if self.online_mining in ('pairs-both', 'pairs-negative', 'both', 'negative'):

            triplets = self.mining_model.predict([x_data, z_latent_dis], batch_size=self.batchsize // self.online_mining_ratio)
            x_data = x_data[triplets[:, 0]]
            z_latent_dis = z_latent_dis[triplets[:, 1]]

            if self.online_mining == 'pairs-negative' or self.online_mining == 'negative':
                x_permutation = np.array(np.random.permutation(len(x_data)), dtype='int64')
            elif self.online_mining == 'pairs-both' or self.online_mining == 'both':
                x_permutation = triplets[:, 2]

        elif self.online_mining == 'anchor-cluster':
            a = self.mining_model.predict([x_data, z_latent_dis])
            batchsize = self.batchsize // self.online_mining_ratio
            kmeans = skcluster.KMeans(n_clusters=batchsize // 2, init='random')
            clusassign = kmeans.fit_predict(a)
            X = pd.DataFrame(a)
            min_dist = np.min(cdist(a, kmeans.cluster_centers_, 'euclidean'), axis=1)
            # max_dist = np.max(cdist(a, kmeans.cluster_centers_, 'euclidean'), axis=1)
            Y = pd.DataFrame(min_dist, index=X.index, columns=['centers'])
            Z = pd.DataFrame(clusassign, index=X.index, columns=['c_id'])
            PAP = pd.concat([Y, Z], axis=1)
            grouped = PAP.groupby(['c_id'])
            idx_min = grouped.idxmin()["centers"].as_matrix()
            Y = pd.DataFrame(min_dist, index=X.index, columns=['centers'])
            PAP = pd.concat([Y, Z], axis=1)
            grouped = PAP.groupby(['c_id'])
            idx_max = grouped.idxmax()["centers"].as_matrix()
            idx = np.concatenate((idx_min, idx_max), axis=-1)
            x_data = x_data[idx]
            x_permutation = np.array(np.random.permutation(len(x_data)), dtype='int64')
            z_latent_dis = z_latent_dis[idx]

        return x_data, z_latent_dis, x_permutation

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

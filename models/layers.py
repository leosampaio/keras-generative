import keras
from keras.engine.topology import Layer
from keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, Dense
from keras.layers import Activation, ELU, LeakyReLU, Dropout, Lambda
from keras.layers.merge import _Merge
from keras import backend as K
import tensorflow as tf


class RandomWeightedAverage(_Merge):
    """Takes a randomly-weighted average of two tensors. In geometric terms, this outputs a random point on the line
    between each pair of input points.
    Inheriting from _Merge is a little messy but it was the quickest solution I could think of.
    Improvements appreciated."""

    def _merge_function(self, inputs):
        if len(inputs[0].shape) > 2:
            weights = K.random_uniform((K.shape(inputs[0])[0], 1, 1, 1))
        else:
            weights = K.random_uniform((K.shape(inputs[0])[0], 1))
        weighted_a = weights * inputs[0]
        weighted_b = (1 - weights) * inputs[1]
        return weighted_a + weighted_b


class MinibatchDiscrimination(Layer):
    __name__ = 'minibatch_discrimination'

    def __init__(self, kernels=50, dims=5, **kwargs):
        super(MinibatchDiscrimination, self).__init__(**kwargs)
        self.kernels = kernels
        self.dims = dims

    def build(self, input_shape):
        assert len(input_shape) == 2
        self.W = self.add_weight(name='kernel',
                                 shape=(input_shape[1], self.kernels * self.dims),
                                 initializer='random_normal',
                                 trainable=True)

    def call(self, inputs):
        Ms = K.dot(inputs, self.W)
        Ms = K.reshape(Ms, (-1, self.kernels, self.dims))
        x_i = K.reshape(Ms, (-1, self.kernels, 1, self.dims))
        x_j = K.reshape(Ms, (-1, 1, self.kernels, self.dims))
        x_i = K.repeat_elements(x_i, self.kernels, 2)
        x_j = K.repeat_elements(x_j, self.kernels, 1)
        norm = K.sum(K.abs(x_i - x_j), axis=3)
        Os = K.sum(K.exp(-norm), axis=2)
        return Os

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.kernels)


def LayerNorm():
    return Lambda(lambda x: K.tf.contrib.layers.layer_norm(x))


class dense(object):

    def __init__(self, units,
                 bnorm=False,
                 lnorm=False,
                 dropout=0.0,
                 activation='leaky_relu',
                 leaky_relu_slope=0.1,
                 residual=None, k_constraint=None,
                 reg=None):

        self.kernel_init = 'glorot_uniform'
        self.bias_init = 'glorot_uniform'
        self.layer = Dense(units=units,
                           kernel_initializer=self.kernel_init,
                           bias_initializer=self.bias_init,
                           kernel_constraint=k_constraint,
                           activity_regularizer=reg)
        self.bn = BatchNormalization()
        self.ln = LayerNorm()
        self.residual = residual
        self.bnorm = bnorm
        self.lnorm = lnorm
        self.activation = activation
        self.dropout = dropout

    def fun(self, inputs):

        x = self.layer(inputs)

        if self.residual is not None:
            x = keras.layers.Add()([x, self.residual])

        if self.bnorm:
            x = self.bn(x)
        elif self.lnorm:
            x = self.ln(x)

        if self.activation == 'leaky_relu':
            x = LeakyReLU(self.leaky_relu_slope)(x)
        elif self.activation == 'elu':
            x = ELU()(x)
        elif self.activation is not None:
            x = Activation(self.activation)(x)

        if self.dropout > 0.0:
            x = Dropout(self.dropout)(x)

        return x

    def add_to_kernel(self, additive):
        self.layer.kernel = self.layer.kernel + additive

    def add_to_bias(self, additive):
        self.layer.bias = self.layer.bias + additive

    def get_gradients(self, loss):
        return K.gradients(loss, self.layer.kernel)[0], K.gradients(loss, self.layer.bias)[0]

    def __call__(self, inputs):
        return self.fun(inputs)


class conv2d(object):

    def __init__(self, filters,
                 kernel_size=(5, 5),
                 padding='same',
                 strides=(1, 1),
                 bnorm=False,
                 lnorm=False,
                 dropout=0.0,
                 activation='leaky_relu',
                 leaky_relu_slope=0.1,
                 residual=None, k_constraint=None,
                 reg=None):

        self.kernel_init = 'glorot_uniform'
        self.bias_init = 'glorot_uniform'
        self.layer = Conv2D(filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            kernel_initializer=self.kernel_init,
                            bias_initializer=self.bias_init,
                            padding=padding,
                            kernel_constraint=k_constraint,
                            activity_regularizer=reg)
        self.bn = BatchNormalization()
        self.ln = LayerNorm()
        self.residual = residual
        self.bnorm = bnorm
        self.lnorm = lnorm
        self.activation = activation
        self.dropout = dropout

    def fun(self, inputs):

        x = self.layer(inputs)

        if self.residual is not None:
            x = keras.layers.Add()([x, self.residual])

        if self.bnorm:
            x = self.bn(x)
        elif self.lnorm:
            x = self.ln(x)

        if self.activation == 'leaky_relu':
            x = LeakyReLU(self.leaky_relu_slope)(x)
        elif self.activation == 'elu':
            x = ELU()(x)
        elif self.activation is not None:
            x = Activation(self.activation)(x)

        if self.dropout > 0.0:
            x = Dropout(self.dropout)(x)

        return x

    def add_to_kernel(self, additive):
        self.layer.kernel = self.layer.kernel + additive

    def add_to_bias(self, additive):
        self.layer.bias = self.layer.bias + additive

    def get_gradients(self, loss):
        return K.gradients(loss, self.layer.kernel)[0], K.gradients(loss, self.layer.bias)[0]

    def __call__(self, inputs):
        return self.fun(inputs)


class deconv2d(object):

    def __init__(self, filters,
                 kernel_size=(5, 5),
                 padding='same',
                 strides=(1, 1),
                 bnorm=False,
                 lnorm=False,
                 dropout=0.0,
                 activation='leaky_relu',
                 leaky_relu_slope=0.1,
                 residual=None, k_constraint=None,
                 reg=None):

        self.kernel_init = 'glorot_uniform'
        self.bias_init = 'glorot_uniform'
        self.layer = Conv2DTranspose(filters=filters,
                                     kernel_size=kernel_size,
                                     strides=strides,
                                     kernel_initializer=self.kernel_init,
                                     bias_initializer=self.bias_init,
                                     padding=padding,
                                     kernel_constraint=k_constraint,
                                     activity_regularizer=reg)
        self.bn = BatchNormalization()
        self.ln = LayerNorm()
        self.residual = residual
        self.bnorm = bnorm
        self.lnorm = lnorm
        self.activation = activation
        self.dropout = dropout

    def fun(self, inputs):

        x = self.layer(inputs)

        if self.residual is not None:
            x = keras.layers.Add()([x, self.residual])

        if self.bnorm:
            x = self.bn(x)
        elif self.lnorm:
            x = self.ln(x)

        if self.activation == 'leaky_relu':
            x = LeakyReLU(self.leaky_relu_slope)(x)
        elif self.activation == 'elu':
            x = ELU()(x)
        elif self.activation is not None:
            x = Activation(self.activation)(x)

        if self.dropout > 0.0:
            x = Dropout(self.dropout)(x)

        return x

    def add_to_kernel(self, additive):
        self.layer.kernel = self.layer.kernel + additive

    def add_to_bias(self, additive):
        self.layer.bias = self.layer.bias + additive

    def get_gradients(self, loss):
        return K.gradients(loss, self.layer.kernel)[0], K.gradients(loss, self.layer.bias)[0]

    def __call__(self, inputs):
        return self.fun(inputs)


def res(filters,
        kernel_size=(5, 5),
        padding='same',
        strides=(1, 1),
        bnorm=True,
        dropout=0.0,
        activation='leaky_relu',
        leaky_relu_slope=0.1,
        residual=None):

    kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    bias_init = keras.initializers.Zeros()
    conv = BasicConvLayer(filters, kernel_size=kernel_size,  padding=padding,  strides=strides,  bnorm=bnorm,
                          dropout=dropout,  activation=activation,  leaky_relu_slope=leaky_relu_slope,  residual=residual)
    resconv = Conv2D(filters=filters,
                     kernel_size=kernel_size,
                     strides=strides,
                     kernel_initializer=kernel_init,
                     bias_initializer=bias_init,
                     padding=padding)
    bn = BatchNormalization()

    def fun(inputs):
        residual = conv(inputs)
        x = resconv(inputs)
        x = keras.layers.Add()([x, residual])
        if bnorm:
            x = bn(x)
        if activation == 'leaky_relu':
            x = LeakyReLU(leaky_relu_slope)(x)
        elif activation == 'elu':
            x = ELU()(x)
        elif activation is not None:
            x = Activation(activation)(x)
        if dropout > 0.0:
            x = Dropout(dropout)(x)
        return x

    return fun


def rdeconv(filters,
            kernel_size=(5, 5),
            padding='valid',
            strides=(1, 1),
            bnorm=True,
            dropout=0.0,
            activation='leaky_relu',
            leaky_relu_slope=0.1,
            residual=None):

    kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.01)
    bias_init = keras.initializers.Zeros()
    conv = deconv2d(filters, kernel_size=kernel_size, padding=padding, strides=strides, bnorm=bnorm,
                    dropout=dropout, activation=activation, leaky_relu_slope=leaky_relu_slope, residual=residual)
    resconv = Conv2DTranspose(filters=filters,
                              kernel_size=kernel_size,
                              strides=strides,
                              kernel_initializer=kernel_init,
                              bias_initializer=bias_init,
                              padding=padding)
    bn = BatchNormalization()

    def fun(inputs):
        residual = conv(inputs)
        x = resconv(inputs)
        x = keras.layers.Add()([x, residual])
        if bnorm:
            x = bn(x)
        if activation == 'leaky_relu':
            x = LeakyReLU(leaky_relu_slope)(x)
        elif activation == 'elu':
            x = ELU()(x)
        elif activation is not None:
            x = Activation(activation)(x)
        if dropout > 0.0:
            x = Dropout(dropout)(x)
        return x

    return fun


def squared_pairwise_distance():
    def _squared_pairwise_distance(input):
        a, b = input
        expanded_a = K.tf.expand_dims(a, 1)
        expanded_b = K.tf.expand_dims(b, 0)
        distances = K.tf.reduce_sum(K.tf.squared_difference(expanded_a, expanded_b), 2)
        return distances
    return Lambda(_squared_pairwise_distance)


def k_largest_indexes(k, idx_dims=2, signal=1):
    def _k_largest_indexes(x):
        if idx_dims > 1:
            _, idx = K.tf.nn.top_k(K.flatten(x * signal), k=k, sorted=True)
            multi_dimensional_idx = K.tf.stack([idx // K.shape(x)[1], idx % K.shape(x)[1]], axis=1)
            # multi_dimensional_idx = K.tf.Print(multi_dimensional_idx, [multi_dimensional_idx.shape,])
            return multi_dimensional_idx
        else:
            _, idx = K.tf.nn.top_k(x * signal, k=k, sorted=True)
            return idx

    return Lambda(_k_largest_indexes)


def print_tensor_shape(input):
    def _tensor_shape(input):
        return K.tf.Print(input, [input.shape])
    return Lambda(_tensor_shape)(input)


class VAELossLayer(Layer):
    __name__ = 'vae_loss_layer'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(VAELossLayer, self).__init__(**kwargs)

    def lossfun(self, x_true, x_pred, z_avg, z_log_var):
        rec_loss = K.mean(K.square(x_true - x_pred))
        kl_loss = K.mean(-0.5 * K.sum(1.0 + z_log_var - K.square(z_avg) - K.exp(z_log_var), axis=-1))
        return rec_loss + kl_loss

    def call(self, inputs):
        x_true = inputs[0]
        x_pred = inputs[1]
        z_avg = inputs[2]
        z_log_var = inputs[3]
        loss = self.lossfun(x_true, x_pred, z_avg, z_log_var)
        self.add_loss(loss, inputs=inputs)

        return x_true


class GramMatrixLayer(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, features):
        x = K.batch_flatten(features)
        x = K.expand_dims(x, axis=-1)
        gram = K.batch_dot(x, K.permute_dimensions(x, (0, 2, 1)))
        return gram

    def compute_output_shape(self, input_shape):
        flattened_shape = input_shape[1] * input_shape[2] * input_shape[3]
        return (input_shape[0], flattened_shape, flattened_shape)


def pi_regularizer_creator(weight=1.0):
    """
    # input shapes
        2D tensor with shape (m, n)
    # output
        l*sum_k(sum_l(pi_kl-1)^2 + sum_l(pi_lk-1)^2)
    """
    def pi_regularizer(perm_matrix):
        l = K.constant(weight)
        sum_over_axis_0 = K.sum(K.sum(perm_matrix - 1, axis=0), axis=1)
        sum_over_axis_1 = K.sum(K.sum(perm_matrix - 1, axis=1), axis=0)
        return l * (sum_over_axis_0 + sum_over_axis_1)


class PermutationMatrixPiLayer(Layer):
    """
    Implements a Kernelized Sorting [1] Permutation Matrix Pi, sliced by
    indexes of the current batch (which must be provided as input)

    [1] N. Quadrianto, L. Song, and A. J. Smola. Kernelized sorting

    # Input shapes
        K: gram matrix for first domain
            2D tensor with shape: (batch, batch)
        L: gram matrix for second domain
            2D tensor with shape: (batch, batch)
        k_indexes: original indexes of first domain samples
            1D tensor with  shape (batch,)
        l_indexes: original indexes of second domain samples
            1D tensor with  shape (batch,)

    # Output shapes
        K.PI
            2D tensor with shape: (batch, batch)
        L.PI^t
            2D tensor with shape: (batch, batch)

    """

    def __init__(self, n, m, restriction_weight, **kwargs):
        """
            args:
                n: num of samples in first domain dataset
                m: num of samples in second domain dataset
                restriction_weight: lambda in formulation, how strongly we want
                    to enforce the PI.1n = 1n.PI^t = 1n restriction
        """
        self.n = n
        self.m = m
        self.restriction_weight = restriction_weight
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.pi = self.add_weight(name='pi',
                                  shape=(self.n, self.m),
                                  initializer='glorot_uniform',
                                  regularizer=pi_regularizer_creator(weight=self.restriction_weight),
                                  trainable=True)
        super().build(input_shape)

    def call(self, inputs, mask=None):
        if type(inputs) is not list or len(inputs) <= 1:
            raise Exception('Permutation matrix must be called on a list of tensors '
                            '(4). Got: ' + str(inputs))

        Kmat = inputs[0]
        Lmat = inputs[1]
        k_indexes = inputs[2]
        l_indexes = inputs[3]

        partial_pi = K.squeeze(tf.gather(K.squeeze(tf.gather(self.pi, k_indexes, axis=1), -1), l_indexes, axis=0), 1)

        K_Pi = K.dot(Kmat, partial_pi)
        L_Pi_t = K.dot(Lmat, K.transpose(partial_pi))
        return [K_Pi, L_Pi_t]

    def compute_output_shape(self, input_shape):
        assert type(input_shape) is list  # must have mutiple input shape tuples

        # all tuples in input_shapes should be the same
        return [input_shape[0], input_shape[0]]

    def compute_mask(self, inputs, mask=None):
        return [None, None]


class GradNorm(Layer):

    def __init__(self, **kwargs):
        super(GradNorm, self).__init__(**kwargs)

    def build(self, input_shapes):
        super(GradNorm, self).build(input_shapes)

    def call(self, inputs):
        target, wrt = inputs
        grads = K.gradients(target, wrt)
        assert len(grads) == 1
        grad = grads[0]
        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True))

    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


class MaximumMeanDiscrepancy(Layer):
    __name__ = 'maximum_mean_discrepancy_layer'

    def __init__(self, **kwargs):
        super(MaximumMeanDiscrepancy, self).__init__(**kwargs)

    def call(self, inputs):
        x_true = inputs[0]
        x_fake = inputs[1]
        return tf.log(mmd.rbf_mmd2(x_true, x_fake))

    def compute_output_shape(self, input_shape):
        assert type(input_shape) is list  # must have mutiple input shape tuples

        # all tuples in input_shapes should be the same
        return (1,)

    def compute_mask(self, inputs, mask=None):
        return [None, None]


# legacy
BasicConvLayer = conv2d
BasicDeconvLayer = deconv2d
ResLayer = res
ResDeconvLayer = rdeconv

import keras
from keras.engine.topology import Layer
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Conv2DTranspose
from keras.layers import Activation, ELU, LeakyReLU, Dropout, Lambda
from keras.layers.merge import _Merge
from keras import backend as K
import tensorflow as tf
from . import mmd


class SampleNormal(Layer):
    __name__ = 'sample_normal'

    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(SampleNormal, self).__init__(**kwargs)

    def _sample_normal(self, z_avg, z_log_var):
        batch_size = K.shape(z_avg)[0]
        z_dims = K.shape(z_avg)[1]
        eps = K.random_normal(shape=K.shape(z_avg), mean=0.0, stddev=1.0)
        return z_avg + K.exp(z_log_var / 2.0) * eps

    def call(self, inputs):
        z_avg = inputs[0]
        z_log_var = inputs[1]
        return self._sample_normal(z_avg, z_log_var)


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


def BasicConvLayer(filters,
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
    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  kernel_initializer=kernel_init,
                  bias_initializer=bias_init,
                  padding=padding)
    bn = BatchNormalization()

    def fun(inputs):

        x = conv(inputs)

        if residual is not None:
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


def BasicDeconvLayer(filters,
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
    conv = Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           kernel_initializer=kernel_init,
                           bias_initializer=bias_init,
                           padding=padding)
    bn = BatchNormalization()

    def fun(inputs):

        x = conv(inputs)

        if residual is not None:
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


def ResLayer(filters,
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
    conv = BasicConvLayer(filters, kernel_size=kernel_size,  padding=padding,  strides=strides,  bnorm=bnorm,  dropout=dropout,  activation=activation,  leaky_relu_slope=leaky_relu_slope,  residual=residual)
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


def ResDeconvLayer(filters,
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
    conv = BasicDeconvLayer(filters, kernel_size=kernel_size, padding=padding, strides=strides, bnorm=bnorm, dropout=dropout, activation=activation, leaky_relu_slope=leaky_relu_slope, residual=residual)
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
        flattened_shape = input_shape[1]*input_shape[2]*input_shape[3]
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

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        weights = K.random_uniform((32, 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

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
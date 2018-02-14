import keras.backend as K
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape, 
    BatchNormalization, Concatenate, Dropout, LeakyReLU, LocallyConnected2D,
    Lambda)
from keras.optimizers import Adam, SGD
import numpy as np

from models import ALI
from models.ali import (DiscriminatorLossLayer, discriminator_accuracy, 
    generator_accuracy, GeneratorLossLayer)
from models.layers import BasicConvLayer, BasicDeconvLayer, SampleNormal
from models.utils import set_trainable, zero_loss

def discriminator_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p, prediction for pairs (Gx(z), z) # label 1
    y_pred[:,1]: q, prediction for pairs (x, Gz(z)) # label 0
    """
    p = K.clip(y_pred[:,0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:,1], K.epsilon(), 1.0 - K.epsilon())
    p_true = y_true[:,0]
    q_true = y_true[:,1]

    q_error = -K.mean(K.log(K.abs(q_true - q)))
    p_error = -K.mean(K.log(K.abs(p - p_true)))

    return q_error + p_error

def generator_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p, prediction for pairs (Gx(z), z)
    y_pred[:,1]: q, prediction for pairs (x, Gz(z))
    """
    p = K.clip(y_pred[:,0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:,1], K.epsilon(), 1.0 - K.epsilon())
    p_true = y_true[:,0]
    q_true = y_true[:,1]

    q_error = -K.mean(K.log(K.abs(p_true - q)))
    p_error = -K.mean(K.log(K.abs(p - q_true)))

    return q_error + p_error

class ALIforSVHN(ALI):
    """
    Based on the original ALI paper arch. Experiment on SVHN.
    See Table 4 on the paper for details
    """
    def __init__(self, *args, **kwargs):
        kwargs['name'] = 'ali_for_svhn'
        super().__init__(*args, **kwargs)

    def build_Gz(self):
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x_input)
        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(128, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(256, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(512, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(512, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)

        x = Flatten()(x)

        # the output is an average (mu) and std variation (sigma) 
        # describing the distribution that better describes the input
        mu = Dense(self.z_dims)(x)
        mu = Activation('linear')(mu)
        sigma = Dense(self.z_dims)(x)
        sigma = Activation('linear')(sigma)

        # use the generated values to sample random z from the latent space
        concatenated = Concatenate(axis=-1)([mu, sigma])
        output = Lambda(
            function=lambda x: x[:,:self.z_dims] + (K.exp(x[:,self.z_dims:]) * (K.random_normal(shape=K.shape(x[:,self.z_dims:])))),
            output_shape=(self.z_dims, )
        )(concatenated)

        return Model(x_input, output, name="Gz")

    def build_Gx(self):
        z_input = Input(shape=(self.z_dims,))
        orig_channels = self.input_shape[2]

        x = Reshape((1, 1, -1))(z_input)

        x = BasicDeconvLayer(256, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(128, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(64, (4, 4), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(32, (4, 4), strides=(2, 2), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicDeconvLayer(32, (5, 5), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)

        x = BasicConvLayer(32, (1, 1), strides=(1, 1), bnorm=True, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(orig_channels, (1, 1), activation='sigmoid', bnorm=False)(x)

        return Model(z_input, x, name="Gx")


    def build_D(self):
        x_input = Input(shape=self.input_shape)

        x = BasicConvLayer(32, (5, 5), strides=(1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x_input)
        x = BasicConvLayer(64, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(128, (4, 4), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(256, (4, 4), strides=(2, 2), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = BasicConvLayer(512, (4, 4), strides=(1, 1), bnorm=True, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(x)
        x = Flatten()(x)

        z_input = Input(shape=(self.z_dims,))
        z = Reshape((1, 1, self.z_dims))(z_input)
        z = BasicConvLayer(512, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
        z = BasicConvLayer(512, (1, 1), bnorm=False, dropout=0.2, activation='leaky_relu', leaky_relu_slope=0.01)(z)
        z = Flatten()(z)

        xz = Concatenate(axis=-1)([x, z])

        xz = Dense(1024)(xz)
        xz = LeakyReLU(0.01)(xz)
        xz = Dropout(0.2)(xz)

        xz = Dense(1024)(xz)
        xz = LeakyReLU(0.01)(xz)
        xz = Dropout(0.2)(xz)

        xz = Dense(1)(xz)
        xz = Activation('sigmoid')(xz)

        return Model([x_input, z_input], xz, name="Discriminator")

    def build_ALI_trainer(self):
        input_x = Input(shape=self.input_shape)
        input_z = Input(shape=(self.z_dims, ))

        assert self.f_D is not None

        p = self.f_D([self.f_Gx(input_z), input_z]) # for pairs (Gx(z), z)
        q = self.f_D([input_x, self.f_Gz(input_x)]) # for pairs (x, Gz(x))

        concatenated = Concatenate(axis=-1)([p, q])
        return Model([input_x, input_z], concatenated, name='ali')

    def build_model(self):

        self.f_Gz = self.build_Gz()
        self.f_Gx = self.build_Gx()
        self.f_D = self.build_D()
        self.f_Gz.summary(); self.f_Gx.summary(); self.f_D.summary()

        opt_d = Adam(lr=1e-6, beta_1=0.5, beta_2=10e-3)
        opt_g = Adam(lr=1e-4, beta_1=0.5, beta_2=10e-3)

        # build discriminator
        self.dis_trainer = self.build_ALI_trainer()
        set_trainable(self.f_Gz, False)
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, True)
        self.dis_trainer.compile(optimizer=opt_d, loss=discriminator_lossfun)

        # build generators
        self.gen_trainer = self.build_ALI_trainer()
        set_trainable(self.f_Gz, True)
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=opt_g, loss=generator_lossfun)

        self.dis_trainer.summary(); self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_trainer')


    def train_on_batch(self, x_data, compute_grad_norms=False):
        # self.swap_weights()

        batchsize = len(x_data)

        # perform label smoothing if applicable
        y_pos, y_neg = ALI.get_labels(batchsize, self.label_smoothing)
        y = np.stack((y_neg, y_pos), axis=1)

        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        # # indicates the upper for losses of the networks, i.e. a net will be retrained (although at most max_retrains
        # # times) until the loss is lower than the bound
        # max_loss = 5.
        # # also an upper bound but only for the generator network: the ratio of losses gen/dis (generator's loss can only
        # # be max_g_2_d_loss_ratio times higher than discriminator's loss
        # max_g_2_d_loss_ratio = 4.5
        # retrained_times, max_retrains = 0, 5
        # while True:
        #     d_loss = self.dis_trainer.train_on_batch([x_data, z_latent_dis], y)

        #     if d_loss < max_loss or retrained_times >= max_retrains:
        #         break
        #     retrained_times += 1
        # if retrained_times > 0:
        #     print('Retrained Discriminator {} time(s)'.format(retrained_times))
        # while True:
        #     g_loss = self.gen_trainer.train_on_batch([x_data, z_latent_dis], y)

        #     if (g_loss < max_loss and g_loss < self.last_d_loss * max_g_2_d_loss_ratio) \
        #             or retrained_times >= max_retrains:
        #         break
        #     retrained_times += 1
        # if retrained_times > 0:
        #     print('Retrained Generator {} time(s)'.format(retrained_times))

        # retrained_times = 0

        d_loss = self.dis_trainer.train_on_batch([x_data, z_latent_dis], y)
        g_loss = self.gen_trainer.train_on_batch([x_data, z_latent_dis], y)
        self.last_d_loss = d_loss

        losses = {
            'g_loss': g_loss,
            'd_loss': d_loss
        }

        return losses
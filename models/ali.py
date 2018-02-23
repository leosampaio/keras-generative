import os
import random
from abc import ABCMeta, abstractmethod

import numpy as np

import keras
from keras.engine.topology import Layer
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape, 
    BatchNormalization, Concatenate, Dropout, LeakyReLU, LocallyConnected2D,
    Lambda)
from keras.optimizers import Adam, Adadelta
from keras import backend as K

from .base import BaseModel

from .utils import *
from .layers import *

def discriminator_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p, prediction for pairs (Gx(z), z)
    y_pred[:,1]: q, prediction for pairs (x, Gz(z))
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


class ALI(BaseModel, metaclass=ABCMeta):
    def __init__(self,
                 input_shape=(64, 64, 3),
                 z_dims=128,
                 name='ali',
                 **kwargs):
        super(ALI, self).__init__(input_shape=input_shape, name=name, **kwargs)

        self.z_dims = z_dims

        self.f_Gz = None
        self.f_Gx = None
        self.f_D = None

        self.gen_trainer = None
        self.dis_trainer = None

        self.last_d_loss = 10000000
        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):
    
        batchsize = len(x_data)

        # perform label smoothing if applicable
        y_pos, y_neg = ALI.get_labels(batchsize, self.label_smoothing)
        y = np.stack((y_neg, y_pos), axis=1)

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        # train both networks
        d_loss = self.dis_trainer.train_on_batch([x_data, z_latent_dis], y)
        g_loss = self.gen_trainer.train_on_batch([x_data, z_latent_dis], y)

        # repeat generator training if loss is too high in comparison with D
        max_loss, max_g_2_d_loss_ratio = 5., 5.
        retrained_times, max_retrains = 0, 3
        while retrained_times < max_retrains and (g_loss > max_loss or g_loss > self.last_d_loss * max_g_2_d_loss_ratio):
            g_loss = self.gen_trainer.train_on_batch([x_data, z_latent_dis], y)
            retrained_times += 1
        if retrained_times > 0:
            print('Retrained Generator {} time(s)'.format(retrained_times))

        self.last_d_loss = d_loss
        losses = {
            'g_loss': g_loss,
            'd_loss': d_loss
        }

        return losses

    def predict(self, z_samples):
        return self.f_Gx.predict(z_samples)

    def build_ALI_trainer(self):
        input_x = Input(shape=self.input_shape)
        input_z = Input(shape=(self.z_dims, ))

        assert self.f_D is not None

        p = self.f_D([self.f_Gx(input_z), input_z]) # for pairs (Gx(z), z)
        q = self.f_D([input_x, self.f_Gz(input_x)]) # for pairs (x, Gz(x))

        concatenated = Concatenate(axis=-1)([p, q])
        return Model([input_x, input_z], concatenated, name='ali')

    def build_model(self):

        self.f_Gz = self.build_Gz() # Moriarty, the encoder
        self.f_Gx = self.build_Gx() # Irene, the decoder
        self.f_D = self.build_D()   # Sherlock, the detective
        self.f_Gz.summary(); self.f_Gx.summary(); self.f_D.summary()

        opt_d, opt_g = self.build_optmizers()
        loss_d, loss_g = self.define_loss_functions()

        # build discriminator
        self.dis_trainer = self.build_ALI_trainer()
        set_trainable(self.f_Gz, False)
        set_trainable(self.f_Gx, False)
        set_trainable(self.f_D, True)
        self.dis_trainer.compile(optimizer=opt_d, loss=loss_d)

        # build generators
        self.gen_trainer = self.build_ALI_trainer()
        set_trainable(self.f_Gz, True)
        set_trainable(self.f_Gx, True)
        set_trainable(self.f_D, False)
        self.gen_trainer.compile(optimizer=opt_g, loss=loss_g)

        self.dis_trainer.summary(); self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_trainer')

    def save_model(self, out_dir, epoch):
        self.trainers['f_D'] = self.f_D
        self.trainers['f_Gz'] = self.f_Gz
        self.trainers['f_Gx'] = self.f_Gx
        super().save_model(out_dir, epoch)
        # remove f_dis from trainers to not load its weights when calling load_model()
        del self.trainers['f_D']
        del self.trainers['f_Gz']
        del self.trainers['f_Gx']

    def define_loss_functions(self):
        return discriminator_lossfun, generator_lossfun

    @abstractmethod
    def build_Gz(self):
        pass

    @abstractmethod
    def build_Gx(self):
        pass

    @abstractmethod
    def build_D(self):
        pass

    @abstractmethod
    def build_optmizers(self):
        pass

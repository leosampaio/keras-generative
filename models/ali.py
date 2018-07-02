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

from core.models import BaseModel

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

        self.is_conditional = kwargs.get('is_conditional', False)
        self.auxiliary_classifier = kwargs.get('auxiliary_classifier', False)
        self.conditional_dims = kwargs.get('conditional_dims', 0)
        self.conditionals_for_samples = kwargs.get('conditionals_for_samples', False)

        self.last_losses = {
            'g_loss': 10.,
            'd_loss': 10.
        }

        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):
        
        batchsize = len(x_data)

        # perform label smoothing if applicable
        y_pos, y_neg = ALI.get_labels(batchsize, self.label_smoothing)
        y = np.stack((y_neg, y_pos), axis=1)

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        if self.is_conditional:
            input_data = [x_data, z_latent_dis, y_batch]
        else:
            input_data = [x_data, z_latent_dis]

        # train both networks
        d_loss = self.dis_trainer.train_on_batch(input_data, y)
        g_loss = self.gen_trainer.train_on_batch(input_data, y)
        if self.last_losses['d_loss'] < self.dis_loss_control:
            g_loss = self.gen_trainer.train_on_batch(input_data, y)
        if self.last_losses['d_loss'] < self.dis_loss_control*1e-5:
            for i in range(0, 5):
                g_loss = self.gen_trainer.train_on_batch(input_data, y)


        self.last_d_loss = d_loss
        losses = {
            'g_loss': g_loss,
            'd_loss': d_loss
        }

        self.last_losses = losses
        return losses

    def predict(self, z_samples):
        return self.f_Gx.predict(z_samples)

    def build_ALI_trainer(self):
        input_x = Input(shape=self.input_shape)
        input_z = Input(shape=(self.z_dims, ))

        assert self.f_D is not None

        if self.is_conditional:
            input_conditional = Input(shape=(self.conditional_dims, ))
            p = self.f_D([self.f_Gx([input_z, input_conditional]), input_z, input_conditional])
            q = self.f_D([input_x, self.f_Gz(input_x), input_conditional])
            input = [input_x, input_z, input_conditional]
        else:
            p = self.f_D([self.f_Gx(input_z), input_z])
            q = self.f_D([input_x, self.f_Gz(input_x)])
            input = [input_x, input_z]

        concatenated = Concatenate(axis=-1)([p, q])
        return Model(input, concatenated, name='ali')

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

    def save_images(self, samples, filename, conditionals_for_samples=None):
        '''
        Save images generated from random sample numbers
        '''
        if self.is_conditional:
            imgs = self.predict([samples, conditionals_for_samples])
        else:
            imgs = self.predict(samples)
        # imgs = np.clip(imgs * 0.5 + 0.5, 0.0, 1.0)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))

        self.save_image_as_plot(imgs, filename)

    def did_collapse(self, losses):
        if losses["g_loss"] == losses["d_loss"]:
            return "G and D losses are equal"
        else: return False

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

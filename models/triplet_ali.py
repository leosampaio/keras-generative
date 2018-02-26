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
import models

from .utils import *
from .layers import *

class TripletALI(BaseModel, metaclass=ABCMeta):
    def __init__(self,
                 ali_models=['ali_SVHN_conditional', 'ali_SVHN_conditional'],
                 *args,
                 **kwargs):
        kwargs['name'] = 'triplet_ali'
        super().__init__(*args, **kwargs)

        self.ali_d1 = models.models[ali_models[0]](*args, **kwargs)
        self.ali_d2 = models.models[ali_models[1]](*args, **kwargs)

        self.last_d_loss = 10000000

        self.is_conditional = kwargs.get('is_conditional', False)
        self.auxiliary_classifier = kwargs.get('auxiliary_classifier', False)
        self.conditional_dims = kwargs.get('conditional_dims', 0)
        self.conditionals_for_samples = kwargs.get('conditionals_for_samples', False)

        self.build_model()


    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):
        
        a_x, p_x, n_x = x_data
        a_y, p_y, n_y = y_batch

        batchsize = len(a_x)

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

        losses = {
            'g_loss': g_loss,
            'd_loss': d_loss
        }

        return losses

    def predict(self, z_samples):
        return self.f_Gx.predict(z_samples)

    def make_batch(self, dataset, indx):
        data, labels = dataset.get_triplets(indx)
        return data, labels

    def build_trainers(self):
        input_a_x = Input(shape=self.input_shape)
        input_p_x = Input(shape=self.input_shape)
        input_n_x = Input(shape=self.input_shape)
        input_z = Input(shape=(self.z_dims, ))

        if self.is_conditional:
            input_conditional = Input(shape=(self.conditional_dims, ))

            # build ALI for Domain 1 (anchor)
            d1_z_a_hat = self.ali_d1.f_Gz(input_a_x)
            d1_x_hat = self.ali_d1.f_Gx([input_z, input_conditional])
            d1_p = self.ali_d1.f_D([d1_x_hat, input_z, input_conditional])
            d1_q = self.ali_d1.f_D([input_a_x, d1_z_a_hat, input_conditional])

            # build ALI for Domain 2 (using the positive samples)
            d2_z_p_hat = self.ali_d2.f_Gz(input_p_x)
            d2_x_hat = self.ali_d2.f_Gx([input_z, input_conditional])
            d2_p = self.ali_d2.f_D([d2_x_hat, input_z, input_conditional])
            d2_q = self.ali_d2.f_D([input_p_x, d2_z_p_hat, input_conditional])

            # get encoding only for Domain 2 negative samples
            d2_z_n_hat = self.ali_d2.f_Gz(input_n_x)

            input = [input_a_x, input_p_x, input_n_x, input_z, input_conditional]
        else:

            # build ALI for Domain 1 (anchor)
            d1_z_a_hat = self.ali_d1.f_Gz(input_a_x)
            d1_x_hat = self.ali_d1.f_Gx(input_z)
            d1_p = self.ali_d1.f_D([d1_x_hat, input_z])
            d1_q = self.ali_d1.f_D([input_a_x, d1_z_a_hat])

            # build ALI for Domain 2 (using the positive samples)
            d2_z_p_hat = self.ali_d2.f_Gz(input_p_x)
            d2_x_hat = self.ali_d2.f_Gx(input_z)
            d2_p = self.ali_d2.f_D([d2_x_hat, input_z])
            d2_q = self.ali_d2.f_D([input_p_x, d2_z_p_hat])

            # get encoding only for Domain 2 negative samples
            d2_z_n_hat = self.ali_d2.f_Gz(input_n_x)

            input = [input_a_x, input_p_x, input_n_x, input_z]

        concatenated_d1 = Concatenate(axis=-1)([d1_p, d1_q])
        concatenated_d2 = Concatenate(axis=-1)([d2_p, d2_q])
        concatenated_triplet_encodings = Concatenate(axis=-1)([d1_z_a_hat, d2_z_p_hat, d2_z_n_hat])
        return Model(
            input, 
            [concatenated_d1, concatenated_d2, concatenated_triplet_encodings], 
            name='triplet_ali'
        )

    def build_model(self):
        pass
        
    def save_model(self, out_dir, epoch):
        self.trainers['f_D'] = self.f_D
        self.trainers['f_Gz'] = self.f_Gz
        self.trainers['f_Gx'] = self.f_Gx
        super().save_model(out_dir, epoch)
        # remove f_dis from trainers to not load its weights when calling load_model()
        del self.trainers['f_D']
        del self.trainers['f_Gz']
        del self.trainers['f_Gx']

    def save_images(self, samples, filename, conditionals_for_samples=None):
        pass

    def build_optmizers(self):
        pass

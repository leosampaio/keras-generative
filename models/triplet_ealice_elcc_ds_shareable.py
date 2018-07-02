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
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K

from core.models import BaseModel
import models

from .utils import *
from .layers import *
from .alice import generator_lossfun, discriminator_lossfun, simple_generator_lossfun, simple_discriminator_lossfun
from .triplet_alice_elcc_ds import TripletALICEwithLCCandDS, triplet_lossfun_creator
from .triplet_ealice_elcc_ds import TripletExplicitALICEwithExplicitLCCandDS


def latent_cycle_mae_loss(y_true, y_pred):

    a, b = y_pred[..., :y_pred.shape[-1] // 2], y_pred[..., (y_pred.shape[-1] // 2):]
    return K.mean(K.abs(a - b), axis=-1)


class TripletExplicitALICEwithExplicitLCCandDSandSharedLayers(BaseModel):

    def __init__(self,
                 submodels=['ealice_shareable', 'ealice_shareable'],
                 n_layers_to_share=0,
                 *args,
                 **kwargs):
        kwargs['name'] = 'triplet_ealice_elcc_ds_shared'
        super().__init__(*args, **kwargs)

        self.alice_d1 = models.models[submodels[0]](*args, **kwargs)
        n_layers_to_share_each_section = (np.max([0, n_layers_to_share - len(self.alice_d1.s_layers[1])]), np.min([len(self.alice_d1.s_layers[1]), n_layers_to_share]))
        self.alice_d2 = models.models[submodels[1]](*args, share_with=self.alice_d1, n_layers_to_share=n_layers_to_share_each_section, **kwargs)

        # create local references to ease model saving and loading
        self.d1_f_D = self.alice_d1.f_D
        self.d1_f_Gz = self.alice_d1.f_Gz
        self.d1_f_Gx = self.alice_d1.f_Gx

        self.d2_f_D = self.alice_d2.f_D
        self.d2_f_Gz = self.alice_d2.f_Gz
        self.d2_f_Gx = self.alice_d2.f_Gx

        self.z_dims = kwargs.get('z_dims', 128)
        self.is_conditional = kwargs.get('is_conditional', False)
        self.auxiliary_classifier = kwargs.get('auxiliary_classifier', False)
        self.conditional_dims = kwargs.get('conditional_dims', 0)
        self.conditionals_for_samples = kwargs.get('conditionals_for_samples', False)
        self.triplet_margin = kwargs.get('triplet_margin', 1.0)
        self.triplet_weight = kwargs.get('triplet_weight', 1.0)
        self.submodels_weights = kwargs.get('submodels_weights', None)

        self.triplet_losses = []

        self.last_losses = {
            'g_loss': 10.,
            'd_loss': 10.,
            'domain1_g_loss': 10.,
            'domain1_d_loss': 10.,
            'domain2_g_loss': 10.,
            'domain2_d_loss': 10.,
            'lc12_loss': 10.,
            'lc21_loss': 10.,
            'triplet_loss': 10.,
        }

        self.build_model()

    def store_trainers(self):
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_trainer')
        self.store_to_save('d1_f_D')
        self.store_to_save('d1_f_Gx')
        self.store_to_save('d2_f_D')
        self.store_to_save('d2_f_Gx')

    predict = TripletALICEwithLCCandDS.__dict__['predict']
    make_batch = TripletALICEwithLCCandDS.__dict__['make_batch']
    save_images = TripletALICEwithLCCandDS.__dict__['save_images']
    predict_images = TripletALICEwithLCCandDS.__dict__['predict_images']
    did_collapse = TripletALICEwithLCCandDS.__dict__['did_collapse']
    plot_losses_hist = TripletALICEwithLCCandDS.__dict__['plot_losses_hist']
    save_losses_history = TripletALICEwithLCCandDS.__dict__['save_losses_history']
    load_model = TripletALICEwithLCCandDS.__dict__['load_model']
    train_on_batch = TripletExplicitALICEwithExplicitLCCandDS.__dict__['train_on_batch']
    build_trainer = TripletExplicitALICEwithExplicitLCCandDS.__dict__['build_trainer']
    build_model = TripletExplicitALICEwithExplicitLCCandDS.__dict__['build_model']
    define_loss_functions = TripletExplicitALICEwithExplicitLCCandDS.__dict__['define_loss_functions']
    build_optmizers = TripletExplicitALICEwithExplicitLCCandDS.__dict__['build_optmizers']

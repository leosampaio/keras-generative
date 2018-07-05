import numpy as np

from keras import Input, Model
from keras.layers import Concatenate
from keras import backend as K

from core.models import BaseModel
import models

from .utils import *
from .layers import *
from .alice import simple_generator_lossfun, simple_discriminator_lossfun
from .triplet_alice_lcc_ds import TripletALICEwithLCCandDS, triplet_lossfun_creator


def latent_cycle_mae_loss(y_true, y_pred):

    a, b = y_pred[..., :y_pred.shape[-1] // 2], y_pred[..., (y_pred.shape[-1] // 2):]
    return K.mean(K.abs(a - b), axis=-1)


class TripletExplicitALICEwithExplicitLCCandDS(BaseModel):

    def __init__(self,
                 submodels=['ealice_shared', 'ealice_shared'],
                 *args,
                 **kwargs):
        kwargs['name'] = 'triplet_ealice_elcc_ds'
        super().__init__(*args, **kwargs)

        self.alice_d1 = models.models[submodels[0]](*args, **kwargs)
        self.alice_d2 = models.models[submodels[1]](*args, **kwargs)

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

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        a_x, p_x, n_x = x_data
        a_y, p_y, n_y = y_batch

        batchsize = len(a_x)

        # perform label smoothing if applicable
        y_pos, y_neg = smooth_binary_labels(batchsize, self.label_smoothing, one_sided_smoothing=True)
        y = np.stack((y_neg, y_pos), axis=1)

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        input_data = [a_x, p_x, n_x, z_latent_dis]

        # train both networks
        d_loss = self.dis_trainer.train_on_batch(input_data, [y, a_x, y, p_x, y, y, y])
        g_loss = self.gen_trainer.train_on_batch(input_data, [y, a_x, y, p_x, y, y, y])
        if self.last_losses['domain1_d_loss'] < self.dis_loss_control or self.last_losses['domain2_d_loss'] < self.dis_loss_control:
            g_loss = self.gen_trainer.train_on_batch(input_data, [y, a_x, y, p_x, y, y, y])

        self.last_losses = {
            'g_loss': g_loss[1] + g_loss[3],
            'd_loss': d_loss[0],
            'domain1_g_loss': g_loss[1],
            'domain1_d_loss': d_loss[1],
            'domain1_c_loss': g_loss[2],
            'domain2_g_loss': g_loss[3],
            'domain2_d_loss': d_loss[3],
            'domain2_c_loss': g_loss[4],
            'lc12_loss': g_loss[5],
            'lc21_loss': g_loss[6],
            'triplet_loss': g_loss[7],
        }

        return self.last_losses

    def build_trainer(self):

        input_a_x = Input(shape=self.input_shape)
        input_p_x = Input(shape=self.input_shape)
        input_n_x = Input(shape=self.input_shape)
        input_z = Input(shape=(self.z_dims, ))

        d1_z_hat = self.alice_d1.f_Gz(input_a_x)
        d2_z_hat = self.alice_d2.f_Gz(input_p_x)

        # build ALICE for Domain 1 (anchor)
        d1_x_hat = self.alice_d1.f_Gx(input_z)
        d1_x_reconstructed = Activation('linear', name='d1_cycled')(self.alice_d1.f_Gx(d1_z_hat))
        d1_p = self.alice_d1.f_D([d1_x_hat, input_z])
        d1_q = self.alice_d1.f_D([input_a_x, d1_z_hat])

        # build ALICE for Domain 2 (using the positive samples)
        d2_x_hat = self.alice_d2.f_Gx(input_z)
        d2_x_reconstructed = Activation('linear', name='d2_cycled')(self.alice_d2.f_Gx(d2_z_hat))
        d2_p = self.alice_d2.f_D([d2_x_hat, input_z])
        d2_q = self.alice_d2.f_D([input_p_x, d2_z_hat])

        input = [input_a_x, input_p_x, input_n_x, input_z]

        # get reconstructed latent variables for latent cycle consistency
        slice_g_lambda = Lambda(lambda x: x[:, :self.z_dims // 2], output_shape=(self.z_dims // 2, ))
        latent_cycle_12 = slice_g_lambda(self.alice_d2.f_Gz(self.alice_d2.f_Gx(d1_z_hat)))
        latent_cycle_21 = slice_g_lambda(self.alice_d1.f_Gz(self.alice_d1.f_Gx(d2_z_hat)))
        sliced_d1_z_hat = slice_g_lambda(d1_z_hat)
        sliced_d2_z_hat = slice_g_lambda(d2_z_hat)

        # get only encoding for Domain 2 negative samples
        d2_z_n_hat = self.alice_d2.f_Gz(input_n_x)

        concatenated_d1 = Concatenate(axis=-1, name="d1_discriminator")([d1_p, d1_q])
        concatenated_d2 = Concatenate(axis=-1, name="d2_discriminator")([d2_p, d2_q])
        concatenated_lat_cycle_12 = Concatenate(axis=-1, name="lat_cycle_12")([latent_cycle_12, sliced_d1_z_hat])
        concatenated_lat_cycle_21 = Concatenate(axis=-1, name="lat_cycle_21")([latent_cycle_21, sliced_d2_z_hat])
        concatenated_triplet_enc = Concatenate(axis=-1, name="triplet_encoding")([d1_z_hat, d2_z_hat, d2_z_n_hat])
        return Model(
            input,
            [concatenated_d1, d1_x_reconstructed, concatenated_d2, d2_x_reconstructed, concatenated_lat_cycle_12, concatenated_lat_cycle_21, concatenated_triplet_enc],
            name='triplet_ali'
        )

    def build_model(self):

        # get loss functions and optmizers
        loss_d, loss_g, loss_triplet = self.define_loss_functions()
        opt_d, opt_g = self.build_optmizers()

        # build the discriminators trainer
        self.dis_trainer = self.build_trainer()
        set_trainable(
            [self.alice_d1.f_Gx, self.alice_d1.f_Gz,
             self.alice_d2.f_Gx, self.alice_d2.f_Gz], False)
        set_trainable([self.alice_d1.f_D, self.alice_d2.f_D], True)
        self.dis_trainer.compile(optimizer=opt_d,
                                 loss={
                                     "d1_discriminator": loss_d,
                                     "d1_cycled": "mae",
                                     "d2_discriminator": loss_d,
                                     "d2_cycled": "mae",
                                     "lat_cycle_12": latent_cycle_mae_loss,
                                     "lat_cycle_21": latent_cycle_mae_loss,
                                     "triplet_encoding": loss_triplet
                                 },
                                 loss_weights=[1., 0., 1., 0., 0., 0., 0.])

        # build the generators trainer
        self.gen_trainer = self.build_trainer()
        set_trainable(
            [self.alice_d1.f_Gx, self.alice_d1.f_Gz,
             self.alice_d2.f_Gx, self.alice_d2.f_Gz], True)
        set_trainable([self.alice_d1.f_D, self.alice_d2.f_D], False)
        self.gen_trainer.compile(optimizer=opt_g,
                                 loss={
                                     "d1_discriminator": loss_g,
                                     "d1_cycled": "mae",
                                     "d2_discriminator": loss_g,
                                     "d2_cycled": "mae",
                                     "lat_cycle_12": latent_cycle_mae_loss,
                                     "lat_cycle_21": latent_cycle_mae_loss,
                                     "triplet_encoding": loss_triplet
                                 },
                                 loss_weights=[1., 1., 1., 1., 1., 1., self.triplet_weight])

        self.dis_trainer.summary()
        self.gen_trainer.summary()

        # Store trainers
        self.store_trainers()

    def store_trainers(self):
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_trainer')
        self.store_to_save('d1_f_D')
        self.store_to_save('d1_f_Gz')
        self.store_to_save('d1_f_Gx')
        self.store_to_save('d2_f_D')
        self.store_to_save('d2_f_Gz')
        self.store_to_save('d2_f_Gx')

    def define_loss_functions(self):
        return simple_discriminator_lossfun, simple_generator_lossfun, triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.z_dims)

    def build_optmizers(self):
        opt_d = Adam(lr=self.lr, clipnorm=5.)
        opt_g = Adam(lr=self.lr, clipnorm=5.)
        return opt_d, opt_g

    predict = TripletALICEwithLCCandDS.__dict__['predict']
    make_batch = TripletALICEwithLCCandDS.__dict__['make_batch']
    save_images = TripletALICEwithLCCandDS.__dict__['save_images']
    predict_images = TripletALICEwithLCCandDS.__dict__['predict_images']
    did_collapse = TripletALICEwithLCCandDS.__dict__['did_collapse']
    plot_losses_hist = TripletALICEwithLCCandDS.__dict__['plot_losses_hist']
    save_losses_history = TripletALICEwithLCCandDS.__dict__['save_losses_history']
    load_model = TripletALICEwithLCCandDS.__dict__['load_model']

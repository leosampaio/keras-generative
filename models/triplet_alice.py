import os

import numpy as np

from keras import Input, Model
from keras.layers import Concatenate
from keras.optimizers import RMSprop
from keras import backend as K

from core.models import BaseModel
import models

from .utils import *
from .layers import *
from .alice import generator_lossfun, discriminator_lossfun


def triplet_lossfun_creator(margin=1., zdims=256):
    def triplet_lossfun(_, y_pred):

        m = K.constant(margin)
        zero = K.constant(0.)
        a, p, n = [y_pred[..., i:i + zdims] for i in range(0, y_pred.shape[-1], zdims)]
        return K.maximum(zero, m + K.sqrt(K.sum(K.square(a - p))) - K.sqrt(K.sum(K.square(a - n))))

    return triplet_lossfun


class TripletALICE(BaseModel):

    def __init__(self,
                 submodels=['alice_mnist', 'alice_svhn'],
                 *args,
                 **kwargs):
        kwargs['name'] = 'triplet_alice'
        super().__init__(*args, **kwargs)

        self.alice_d1 = models.models[submodels[0]](*args, **kwargs)
        self.alice_d2 = models.models[submodels[1]](*args, **kwargs)

        # create local references to ease model saving and loading
        self.d1_f_D = self.alice_d1.f_D
        self.d1_f_Gz = self.alice_d1.f_Gz
        self.d1_f_Gx = self.alice_d1.f_Gx
        self.d1_f_D_cycle = self.alice_d1.f_D_cycle

        self.d2_f_D = self.alice_d2.f_D
        self.d2_f_Gz = self.alice_d2.f_Gz
        self.d2_f_Gx = self.alice_d2.f_Gx
        self.d2_f_D_cycle = self.alice_d2.f_D_cycle

        self.last_d_loss = 10000000

        self.z_dims = kwargs.get('z_dims', 128)
        self.is_conditional = kwargs.get('is_conditional', False)
        self.auxiliary_classifier = kwargs.get('auxiliary_classifier', False)
        self.conditional_dims = kwargs.get('conditional_dims', 0)
        self.conditionals_for_samples = kwargs.get('conditionals_for_samples', False)
        self.triplet_margin = kwargs.get('triplet_margin', 1.0)
        self.triplet_weight = kwargs.get('triplet_weight', 1.0)
        self.submodels_weights = kwargs.get('submodels_weights', None)

        self.triplet_losses = []
        self.recompiled = False

        self.build_model()

    def train_on_batch(self, x_data, y_batch=None, compute_grad_norms=False):

        a_x, p_x, n_x = x_data
        a_y, p_y, n_y = y_batch

        batchsize = len(a_x)

        # perform label smoothing if applicable
        y_pos, y_neg = smooth_binary_labels(batchsize, self.label_smoothing, one_sided_smoothing=True)
        y = np.stack((y_neg, y_pos), axis=1)
        y = [y] * 3

        # get real latent variables distribution
        z_latent_dis = np.random.normal(size=(batchsize, self.z_dims))

        if self.is_conditional:
            input_data = [a_x, p_x, n_x, z_latent_dis, a_y]
        else:
            input_data = [a_x, p_x, n_x, z_latent_dis]

        # train both networks
        d_loss = self.dis_trainer.train_on_batch(input_data, y)
        g_loss = self.gen_trainer.train_on_batch(input_data, y)

        gen_loss = g_loss[1] + g_loss[2]
        dis_loss = d_loss[0]

        self.last_d_loss = dis_loss

        losses = {
            'g_loss': gen_loss,
            'd_loss': d_loss[0],
            'domain1_g_loss': g_loss[1],
            'domain1_d_loss': d_loss[1],
            'domain2_g_loss': g_loss[2],
            'domain2_d_loss': d_loss[2],
            'triplet_loss': g_loss[3],
        }

        return losses

    def predict(self, z_samples, domain=1):
        if domain == 1:
            return self.alice_d1.f_Gx.predict(z_samples)
        elif domain == 2:
            return self.alice_d2.f_Gx.predict(z_samples)

    def make_batch(self, dataset, indx):
        data, labels = dataset.get_triplets(indx)
        return data, labels

    def build_trainer(self):
        input_a_x = Input(shape=self.input_shape)
        input_p_x = Input(shape=self.input_shape)
        input_n_x = Input(shape=self.input_shape)
        input_z = Input(shape=(self.z_dims, ))

        d1_z_a_hat = self.alice_d1.f_Gz(input_a_x)
        d2_z_p_hat = self.alice_d2.f_Gz(input_p_x)

        if self.is_conditional:
            input_conditional = Input(shape=(self.conditional_dims, ))

            # build ALICE for Domain 1 (anchor)
            d1_x_hat = self.alice_d1.f_Gx([input_z, input_conditional])
            d1_x_reconstructed = self.alice_d1.f_Gx([d1_z_a_hat, input_conditional])
            d1_p = self.alice_d1.f_D([d1_x_hat, input_z, input_conditional])
            d1_q = self.alice_d1.f_D([input_a_x, d1_z_a_hat, input_conditional])

            # build ALICE for Domain 2 (using the positive samples)
            d2_x_hat = self.alice_d2.f_Gx([input_z, input_conditional])
            d2_x_reconstructed = self.alice_d2.f_Gx([d2_z_p_hat, input_conditional])
            d2_p = self.alice_d2.f_D([d2_x_hat, input_z, input_conditional])
            d2_q = self.alice_d2.f_D([input_p_x, d2_z_p_hat, input_conditional])

            input = [input_a_x, input_p_x, input_n_x, input_z, input_conditional]
        else:

            # build ALICE for Domain 1 (anchor)
            d1_x_hat = self.alice_d1.f_Gx(input_z)
            d1_x_reconstructed = self.alice_d1.f_Gx(d1_z_a_hat)
            d1_p = self.alice_d1.f_D([d1_x_hat, input_z])
            d1_q = self.alice_d1.f_D([input_a_x, d1_z_a_hat])

            # build ALICE for Domain 2 (using the positive samples)
            d2_x_hat = self.alice_d2.f_Gx(input_z)
            d2_x_reconstructed = self.alice_d2.f_Gx(d2_z_p_hat)
            d2_p = self.alice_d2.f_D([d2_x_hat, input_z])
            d2_q = self.alice_d2.f_D([input_p_x, d2_z_p_hat])

            input = [input_a_x, input_p_x, input_n_x, input_z]

        # set cycle consistency discriminators
        d1_p_cycle = self.alice_d1.f_D_cycle([input_a_x, input_a_x])
        d1_q_cycle = self.alice_d1.f_D_cycle([input_a_x, d1_x_reconstructed])
        d2_p_cycle = self.alice_d2.f_D_cycle([input_p_x, input_p_x])
        d2_q_cycle = self.alice_d2.f_D_cycle([input_p_x, d2_x_reconstructed])

        # get only encoding for Domain 2 negative samples
        d2_z_n_hat = self.alice_d2.f_Gz(input_n_x)

        concatenated_d1 = Concatenate(axis=-1, name="d1_discriminator")([d1_p, d1_q, d1_p_cycle, d1_q_cycle])
        concatenated_d2 = Concatenate(axis=-1, name="d2_discriminator")([d2_p, d2_q, d2_p_cycle, d2_q_cycle])
        concatenated_triplet_enc = Concatenate(axis=-1, name="triplet_encoding")([d1_z_a_hat, d2_z_p_hat, d2_z_n_hat])
        return Model(
            input,
            [concatenated_d1, concatenated_d2, concatenated_triplet_enc],
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
        set_trainable([self.alice_d1.f_D, self.alice_d2.f_D,
                       self.alice_d1.f_D_cycle, self.alice_d2.f_D_cycle], True)
        self.dis_trainer.compile(optimizer=opt_d,
                                 loss={
                                     "d1_discriminator": loss_d,
                                     "d2_discriminator": loss_d,
                                     "triplet_encoding": loss_triplet
                                 },
                                 loss_weights=[1., 1., 0.])

        # build the generators trainer
        self.gen_trainer = self.build_trainer()
        set_trainable(
            [self.alice_d1.f_Gx, self.alice_d1.f_Gz,
             self.alice_d2.f_Gx, self.alice_d2.f_Gz], True)
        set_trainable([self.alice_d1.f_D, self.alice_d2.f_D,
                       self.alice_d1.f_D_cycle, self.alice_d2.f_D_cycle], False)
        self.gen_trainer.compile(optimizer=opt_g,
                                 loss={
                                     "d1_discriminator": loss_g,
                                     "d2_discriminator": loss_g,
                                     "triplet_encoding": loss_triplet
                                 },
                                 loss_weights=[1., 1., self.triplet_weight])

        self.dis_trainer.summary()
        self.gen_trainer.summary()

        # Store trainers
        self.store_to_save('dis_trainer')
        self.store_to_save('gen_trainer')
        self.store_to_save('d1_f_D')
        self.store_to_save('d1_f_D_cycle')
        self.store_to_save('d1_f_Gz')
        self.store_to_save('d1_f_Gx')
        self.store_to_save('d2_f_D')
        self.store_to_save('d2_f_D_cycle')
        self.store_to_save('d2_f_Gz')
        self.store_to_save('d2_f_Gx')

    def save_model(self, out_dir, epoch):
        super().save_model(out_dir, epoch)

    def define_loss_functions(self):
        return discriminator_lossfun, generator_lossfun, triplet_lossfun_creator(margin=self.triplet_margin, zdims=self.z_dims)

    def save_images(self, samples, filename, conditionals_for_samples=None):
        if self.is_conditional:
            d1_imgs = self.alice_d1.f_Gx.predict([samples, conditionals_for_samples])
            d2_imgs = self.alice_d2.f_Gx.predict([samples, conditionals_for_samples])
        else:
            d1_imgs = self.alice_d1.f_Gx.predict(samples)
            d2_imgs = self.alice_d2.f_Gx.predict(samples)
        imgs = np.empty((2 * len(samples)) + tuple(d1_imgs.shape[1:]), dtype=d1_imgs.dtype)
        imgs[0::2] = d1_imgs
        imgs[1::2] = d2_imgs
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))

        self.save_image_as_plot(imgs[:len(samples)], filename)

    def build_optmizers(self):
        opt_d = RMSprop(lr=self.lr)
        opt_g = RMSprop(lr=self.lr)
        return opt_d, opt_g

    def predict_images(self, z_sample, domain=1):
        images = self.predict(z_sample, domain=domain)
        if images.shape[3] == 1:
            images = np.squeeze(imgs, axis=(3,))
        return images

    def did_collapse(self, losses):
        if losses["g_loss"] == losses["d_loss"]:
            return "G and D losses are equal"
        elif losses["domain1_g_loss"] == losses["domain1_d_loss"]:
            return "Domain 1 G and D losses are equal"
        elif losses["domain2_g_loss"] == losses["domain2_d_loss"]:
            return "Domain 2 G and D losses are equal"
        else:
            return False

    def plot_losses_hist(self, out_dir):
        plt.plot(self.g_losses, label='Gen')
        plt.plot(self.d_losses, label='Dis')
        plt.plot(self.triplet_losses, label='Triplet')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'loss_hist.png'))
        plt.close()

        plt.plot(self.losses_ratio, label='G / D')
        plt.legend()
        plt.savefig(os.path.join(out_dir, 'losses_ratio.png'))
        plt.close()

    def save_losses_history(self, losses):
        self.g_losses.append(losses['g_loss'])
        self.d_losses.append(losses['d_loss'])
        self.triplet_losses.append(losses["triplet_loss"])
        self.losses_ratio.append(losses['g_loss'] / losses['d_loss'])

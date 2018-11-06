from keras import backend as K
import numpy as np


def triplet_distances(a, p, n, metric='l2', nn=None):
    if metric == 'l2':
        # d_p = K.tf.norm((a - p), axis=1, ord=2)
        # d_n = K.tf.norm((a - n), axis=1, ord=2)
        d_p = K.sqrt(K.maximum(K.epsilon(), K.sum(K.square(a - p), axis=1)))
        d_n = K.sqrt(K.maximum(K.epsilon(), K.sum(K.square(a - n), axis=1)))
        if nn is not None:
            d_nn = K.sqrt(K.maximum(K.epsilon(), K.sum(K.square(n - nn), axis=1)))
        # d_p = K.tf.Print(d_p, [d_p, d_n, K.shape(d_p)], message="[dp, dn]:")
    elif metric == 'l1':
        d_p = K.tf.norm((a - p), axis=1, ord=1)
        d_n = K.tf.norm((a - n), axis=1, ord=1)
        if nn is not None:
            d_nn = K.tf.norm((n - nn), axis=1, ord=1)
    elif metric == 'inf':
        d_p = K.tf.norm((a - p), axis=1, ord=np.inf)
        d_n = K.tf.norm((a - n), axis=1, ord=np.inf)
        if nn is not None:
            d_nn = K.tf.norm((n - nn), axis=1, ord=np.inf)
    elif metric == 'cos':
        norm_a, norm_p, norm_n = K.tf.nn.l2_normalize(a, axis=1), K.tf.nn.l2_normalize(p, axis=1), K.tf.nn.l2_normalize(n, axis=1)
        d_p = -K.sum(norm_a * norm_p, axis=1)
        d_n = -K.sum(norm_a * norm_n, axis=1)
        if nn is not None:
            norm_nn = K.tf.nn.l2_normalize(nn, axis=1)
            d_nn = -K.sum(norm_nn * norm_n, axis=1)

    if nn is not None:
        return d_p, d_n, d_nn
    else:
        return d_p, d_n


def triplet_lossfun_creator(margin=1., zdims=256, distance_metric='l2', ttype='normal'):
    def triplet_lossfun(_, y_pred):

        m = K.ones((K.shape(y_pred)[0],)) * margin
        zero = K.zeros((K.shape(y_pred)[0],))
        a, p, n = [y_pred[..., i:i + zdims] for i in range(0, y_pred.shape[-1], zdims)]
        d_p, d_n = triplet_distances(a, p, n, metric=distance_metric)
        if ttype == 'inverted':
            return K.mean(K.maximum(zero, - d_p + d_n))
        elif ttype == 'simplified':
            return K.mean(K.maximum(zero, m - d_p))
        elif ttype == 'normal':
            return K.mean(K.maximum(zero, m + d_p - d_n))

    return triplet_lossfun


def quadruplet_lossfun_creator(margin=1., zdims=256, distance_metric='l2', ttype='normal'):
    def quadruplet_lossfun(_, y_pred):

        m = K.ones((K.shape(y_pred)[0],)) * margin
        zero = K.zeros((K.shape(y_pred)[0],))
        a, p, n, nn = [y_pred[..., i:i + zdims] for i in range(0, y_pred.shape[-1], zdims)]
        d_p, d_n, d_nn = triplet_distances(a, p, n, nn=nn, metric=distance_metric)
        if ttype == 'inverted':
            return K.mean(K.maximum(zero, - d_p + d_n) + K.maximum(zero, d_n - d_nn))
        elif ttype == 'simplified':
            return K.mean(K.maximum(zero, m - d_p))
        elif ttype == 'normal':
            return K.mean(K.maximum(zero, m + d_p - d_n))

    return quadruplet_lossfun


def generic_triplet_lossfun_creator(margin=1., ttype='normal', distance_metric='l2'):
    def triplet_lossfun(_, y_pred):

        m = K.ones((K.shape(y_pred)[0],)) * margin
        zero = K.zeros((K.shape(y_pred)[0],))
        a, p, n = y_pred[..., 0], y_pred[..., 1], y_pred[..., 2]
        d_p, d_n = triplet_distances(a, p, n, metric=distance_metric)
        if ttype == 'inverted':
            return K.mean(K.maximum(zero, - d_p + d_n))
        elif ttype == 'simplified':
            return K.mean(K.maximum(zero, m - d_p))
        else:
            return K.mean(K.maximum(zero, m + d_p - d_n))

    return triplet_lossfun


def triplet_balance_creator(margin=1., zdims=256, gamma=1., distance_metric='l2'):
    def triplet_balance(_, y_pred):
        a, p, n = [y_pred[..., i:i + zdims] for i in range(0, y_pred.shape[-1], zdims)]
        d_p, d_n = triplet_distances(a, p, n, metric=distance_metric)
        return - d_p + d_n

    return triplet_balance


def triplet_std_creator(margin=1., zdims=256, distance_metric='l2'):
    def triplet_std(_, y_pred):
        a, p, n = [y_pred[..., i:i + zdims] for i in range(0, y_pred.shape[-1], zdims)]
        d_p, d_n = triplet_distances(a, p, n, metric=distance_metric)
        return K.std(d_p - d_n)

    return triplet_std


def topgan_magan_equilibrium_creator(zdims=256):
    def topgan_magan_equilibrium(_, y_pred):
        a, p, n = [y_pred[..., i:i + zdims] for i in range(0, y_pred.shape[-1], zdims)]
        return K.sqrt(K.sum(K.square(a - p)))
    return topgan_magan_equilibrium


def eq_triplet_lossfun_creator(margin=1., zdims=256, k=1, simplified=False):
    def triplet_lossfun(_, y_pred):

        m = margin
        zero = K.constant(0.)
        a, p, n = [y_pred[..., i:i + zdims] for i in range(0, y_pred.shape[-1], zdims)]
        if simplified:
            return K.maximum(zero, m - k * K.sqrt(K.sum(K.square(a - n))))
        else:
            return K.maximum(zero, m + K.sqrt(K.sum(K.square(a - p))) - k * K.sqrt(K.sum(K.square(a - n))))

    return triplet_lossfun


def discriminator_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p, D(Gx(z))
    y_pred[:,1]: q, D(x)
    """
    p = K.clip(y_pred[:, 0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:, 1], K.epsilon(), 1.0 - K.epsilon())
    # p_true = y_true[:, 0] # 0
    # q_true = y_true[:, 1] # 1

    q_error = -K.mean(K.log(K.abs(0 - q)))
    p_error = -K.mean(K.log(K.abs(p - 1)))

    return q_error + p_error


def generator_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p, D(Gx(z))
    y_pred[:,1]: q, D(x)
    """
    p = K.clip(y_pred[:, 0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:, 1], K.epsilon(), 1.0 - K.epsilon())
    # p_true = y_true[:, 0]
    # q_true = y_true[:, 1]

    q_error = -K.mean(K.log(K.abs(1 - q)))
    p_error = -K.mean(K.log(K.abs(p - 0)))

    return q_error + p_error


def wasserstein_dis_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p, D(Gx(z))
    y_pred[:,1]: q, D(x)
    """
    p = K.clip(y_pred[:, 0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:, 1], K.epsilon(), 1.0 - K.epsilon())
    return K.mean(q) - K.mean(p)


def wasserstein_gen_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: p, D(Gx(z))
    y_pred[:,1]: q, D(x)
    """
    p = K.clip(y_pred[:, 0], K.epsilon(), 1.0 - K.epsilon())
    q = K.clip(y_pred[:, 1], K.epsilon(), 1.0 - K.epsilon())
    return K.mean(p)


def wgan_gradient_penalty_lossfun(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """
    https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py

    Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the loss function
    that penalizes the network if the gradient norm moves away from 1. However, it is impossible to evaluate
    this function at all points in the input space. The compromise used in the paper is to choose random points
    on the lines between real and generated samples, and check the gradients at these points. Note that it is the
    gradient w.r.t. the input averaged samples, not the weights of the discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator and evaluate the loss.
    Then we get the gradients of the discriminator w.r.t. the input averaged samples.
    The l2 norm and penalty can then be calculated for this gradient.
    Note that this loss function requires the original averaged samples as input, but Keras only supports passing
    y_true and y_pred to loss functions. To get around this, we make a partial() of the function with the
    averaged_samples argument, and use that for model training."""
    # first get the gradients:
    #   assuming: - that y_pred has dimensions (batch_size, 1)
    #             - averaged_samples has dimensions (batch_size, nbr_features)
    # gradients afterwards has dimension (batch_size, nbr_features), basically
    # a list of nbr_features-dimensional gradient vectors
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def began_gen_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: (Gx(z))
    y_pred[:,1]: D(Gx(z))
    y_pred[:,2]: D(x)
    y_true:      x
    """
    x_hat = y_pred[..., 0]
    x_hat_reconstructed = y_pred[..., 1]
    ae_loss = K.mean(K.abs(x_hat - x_hat_reconstructed))
    return ae_loss


def ae_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: x
    y_pred[:,1]: D(x)
    """
    x = y_pred[..., 0]
    x_reconstructed = y_pred[..., 1]
    ae_loss = K.mean(K.square(x - x_reconstructed))
    return ae_loss


def began_dis_lossfun(y_true, y_pred, k_gd_ratio):
    """
    y_pred[:,0]: (Gx(z))
    y_pred[:,1]: D(Gx(z))
    y_pred[:,2]: D(x)
    y_true:      x
    """
    x_hat = y_pred[..., 0]
    x_hat_reconstructed = y_pred[..., 1]
    x_real = y_true[..., 0]
    x_real_reconstructed = y_pred[..., 2]
    fake_ae_loss = K.mean(K.abs(x_hat - x_hat_reconstructed))
    real_ae_loss = K.mean(K.abs(x_real - x_real_reconstructed))
    return real_ae_loss - k_gd_ratio * fake_ae_loss


def began_convergence_lossfun(y_true, y_pred, gamma):
    """
    y_pred[:,0]: (Gx(z))
    y_pred[:,1]: D(Gx(z))
    y_pred[:,2]: D(x)
    y_true:      x
    """
    x_hat = y_pred[..., 0]
    x_hat_reconstructed = y_pred[..., 1]
    x_real = y_true[..., 0]
    x_real_reconstructed = y_pred[..., 2]
    fake_ae_loss = K.mean(K.abs(x_hat - x_hat_reconstructed))
    real_ae_loss = K.mean(K.abs(x_real - x_real_reconstructed))
    return real_ae_loss + K.abs(gamma * real_ae_loss - fake_ae_loss)


def ganomaly_latent_ae_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: Gz(x)
    y_pred[:,1]: Dec(Enc(Gz(x)))
    """
    z = y_pred[:, 0]
    z_hat = y_pred[:, 1]
    return K.mean(K.abs(z - z_hat))


def feat_matching_lossfun(_, y_pred):
    a, n = y_pred[:, 0], y_pred[:, 1]
    return K.sqrt(K.sum(K.square(a - n)))


def con_began_gen_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: half1 (Gx(z))
    y_pred[:,1]: half2 (Gx(z))
    y_pred[:,2]: D(half1 Gx(z))
    y_pred[:,3]: half1 x
    y_pred[:,4]: half2 x
    y_pred[:,5]: D(half1 x)
    """
    half_x_hat = y_pred[..., 0]
    con_half_x_hat = y_pred[..., 1]
    half_x_hat_reconstructed = y_pred[..., 2]
    con_ae_loss = K.mean(K.abs(half_x_hat - half_x_hat_reconstructed)) - K.mean(K.abs(half_x_hat - con_half_x_hat))
    return con_ae_loss


def con_began_dis_lossfun_creator(k_gd_ratio):
    """
    y_pred[:,0]: half1 (Gx(z))
    y_pred[:,1]: half2 (Gx(z))
    y_pred[:,2]: D(half1 Gx(z))
    y_pred[:,3]: half1 x
    y_pred[:,4]: half2 x
    y_pred[:,5]: D(half1 x)
    """
    def con_began_dis_lossfun(y_true, y_pred):
        half_x_hat = y_pred[..., 0]
        con_half_x_hat = y_pred[..., 1]
        half_x_hat_reconstructed = y_pred[..., 2]
        half_x = y_pred[..., 3]
        con_half_x = y_pred[..., 4]
        half_x_reconstructed = y_pred[..., 5]
        con_ae_loss = K.mean(K.abs(half_x - half_x_reconstructed)) - K.mean(K.abs(half_x - con_half_x))
        fake_con_ae_loss = K.mean(K.abs(half_x_hat - half_x_hat_reconstructed)) - K.mean(K.abs(half_x_hat - con_half_x_hat))
        return con_ae_loss - k_gd_ratio * fake_con_ae_loss
    return con_began_dis_lossfun


def con_began_convergence_lossfun_creator(gamma):
    """
    y_pred[:,0]: half1 (Gx(z))
    y_pred[:,1]: half2 (Gx(z))
    y_pred[:,2]: D(half1 Gx(z))
    y_pred[:,3]: half1 x
    y_pred[:,4]: half2 x
    y_pred[:,5]: D(half1 x)
    """
    def con_began_convergence_lossfun(y_true, y_pred):
        half_x_hat = y_pred[..., 0]
        con_half_x_hat = y_pred[..., 1]
        half_x_hat_reconstructed = y_pred[..., 2]
        half_x = y_pred[..., 3]
        con_half_x = y_pred[..., 4]
        half_x_reconstructed = y_pred[..., 5]
        con_ae_loss = K.mean(K.abs(half_x - half_x_reconstructed)) - K.mean(K.abs(half_x - con_half_x))
        fake_con_ae_loss = K.mean(K.abs(half_x_hat - half_x_hat_reconstructed)) - K.mean(K.abs(half_x_hat - con_half_x_hat))
        return con_ae_loss + K.abs(gamma * con_ae_loss - fake_con_ae_loss)
    return con_began_convergence_lossfun


def con_began_ae_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: half1 (Gx(z))
    y_pred[:,1]: half2 (Gx(z))
    y_pred[:,2]: D(half1 Gx(z))
    y_pred[:,3]: half1 x
    y_pred[:,4]: half2 x
    y_pred[:,5]: D(half1 x)
    """
    half_x = y_pred[..., 3]
    con_half_x = y_pred[..., 4]
    half_x_reconstructed = y_pred[..., 5]
    con_ae_loss = K.mean(K.abs(half_x - half_x_reconstructed)) - K.mean(K.abs(half_x - con_half_x))
    return con_ae_loss


def topgan_began_gen_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: (Gx(z))
    y_pred[:,1]: D(Gx(z))
    y_pred[:,2]: D(x)
    y_pred[:,3]: x
    """
    x_hat = y_pred[..., 0]
    x_hat_reconstructed = y_pred[..., 1]
    ae_loss = K.mean(K.abs(x_hat - x_hat_reconstructed))
    return ae_loss


def topgan_began_dis_lossfun(y_true, y_pred):
    """
    y_pred[:,0]: (Gx(z))
    y_pred[:,1]: D(Gx(z))
    y_pred[:,2]: D(x)
    y_pred[:,3]: x
    """
    x_hat = y_pred[..., 0]
    x_hat_reconstructed = y_pred[..., 1]
    x_real = y_pred[..., 3]
    x_real_reconstructed = y_pred[..., 2]
    # y_pred = K.tf.Print(y_pred, [y_pred.shape])
    # a = K.tf.Print(a, [a.shape])
    # p = K.tf.Print(p, [p.shape])
    fake_ae_loss = K.mean(K.abs(x_hat - x_hat_reconstructed))
    real_ae_loss = K.mean(K.abs(x_real - x_real_reconstructed))
    return real_ae_loss - fake_ae_loss

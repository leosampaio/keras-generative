from keras import backend as K
from keras.callbacks import Callback
import os
import sys
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches


def set_trainable(model, train):
    """
    Enable or disable training for the model
    """
    if type(model) != list:
        mlist = [model]
    else:
        mlist = model
    for model in mlist:
        model.trainable = train
        for l in model.layers:
            l.trainable = train


def zero_loss(y_true, y_pred):
    return K.zeros_like(y_true)


def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def interpolate_vectors(a, b, steps=10):
    interp = np.zeros((steps,) + a.shape)
    for i in range(a.shape[0]):
        for step in range(steps - 1):
            interp[step][i] = a[i] + (b[i] - a[i]) / steps * step
    interp[steps - 1] = b
    return interp


class PlotReconstructions(Callback):

    def __init__(self, encoder, decoder, z_dim=256, examples_to_plot=10):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.examples_to_plot = examples_to_plot
        self.z_dim = z_dim

        if not os.path.isdir('aae'):
            os.makedirs('aae')

    def on_epoch_end(self, epoch, logs=None):
        data = self.validation_data[0][:self.examples_to_plot]
        decoded_imgs = self.model.predict(data)
        random_z = np.random.normal(size=(self.examples_to_plot, self.z_dim))
        random_imgs = self.decoder.predict(random_z)

        # interpolations
        a = data[0]
        b = data[1]
        encodings = self.encoder.predict(np.array([a, b]).reshape((2,) + a.shape))
        interp = interpolate_vectors(encodings[0], encodings[1])
        decoded_interp = self.decoder.predict(interp)

        plot_reconstructions(data, decoded_imgs, random_imgs, decoded_interp, 'aae/aae_{}.png'.format(epoch))


def plot_reconstructions(x, y, random_imgs, interpolations, plotname):
    n = x.shape[0]
    img_size = x.shape[1]
    plt.figure(figsize=(15, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(4, n, i + 1)
        plt.imshow(x[i].reshape((img_size, img_size, 3)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(4, n, i + 1 + n)
        plt.imshow(y[i].reshape((img_size, img_size, 3)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display random images
        ax = plt.subplot(4, n, i + 1 + 2 * n)
        plt.imshow(random_imgs[i].reshape((img_size, img_size, 3)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # interpolation
        ax = plt.subplot(4, n, i + 1 + 3 * n)
        plt.imshow(interpolations[i].reshape((img_size, img_size, 3)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(plotname)
    plt.close()


def get_gradient_norm_func(model):
    # https://github.com/keras-team/keras/issues/2226
    weights = model.trainable_weights  # weight tensors
    #weights = [weight for weight in weights if model.get_layer(weight.name[:-2]).trainable]
    gradients = model.optimizer.get_gradients(model.total_loss, weights)  # gradient tensors
    summed_squares = [K.sum(K.square(g)) for g in gradients]
    norm = K.sqrt(sum(summed_squares))
    input_tensors = [model.inputs[0],  # input data
                     model.inputs[1],
                     model.sample_weights[0],  # how much to weight each sample by
                     model.targets[0],  # labels
                     K.learning_phase(),  # train or test mode
                     ]

    func = K.function(inputs=input_tensors, outputs=[norm])
    return func


def time_format(t):
    m, s = divmod(t, 60)
    m = int(m)
    s = int(s)
    if m == 0:
        return '%d sec' % s
    else:
        return '%d min %d sec' % (m, s)


def print_current_progress(current_epoch, current_batch_index, batch_size, dataset_length, losses, elapsed_time):
    ratio = 100.0 * (current_batch_index + batch_size) / dataset_length
    status_string = "Epoch #{:04.0f} | {:06.0f} / {:06.0f} ({:6.2f}% )".format(current_epoch + 1, current_batch_index + batch_size, dataset_length, ratio)

    for k, l in losses.items():
        status_string = "{} | {} = {:8.6f}".format(status_string, k, l)
        assert not np.isnan(l)

    # compute ETA
    eta = elapsed_time / (current_batch_index + batch_size) * (dataset_length - (current_batch_index + batch_size))
    status_string = "{} | ETA: {}".format(status_string, time_format(eta))
    print(status_string)
    sys.stdout.flush()


def add_input_noise(x_batch, curr_epoch, total_epochs, start_noise):
    if start_noise == 0.0:
        return x_batch

    if type(x_batch) == tuple:
        batchsize = x_batch[0].shape
    else:
        batchsize = x_batch.shape
    noise = np.random.normal(size=batchsize)

    noise_factor = curr_epoch / total_epochs
    if noise_factor < 0.02:
        return x_batch

    if type(x_batch) == tuple:
        noised_batch = tuple([X + noise * noise_factor for X in x_batch])
    else:
        noised_batch = x_batch + noise * noise_factor
    return noised_batch


def add_input_noise_to_autoencoder(x_batch):
    batchsize = x_batch.shape
    noise = np.random.normal(size=batchsize)
    return x_batch + noise * 0.0


def smooth_binary_labels(batchsize, smoothing=0.0, one_sided_smoothing=True):
    if smoothing > 0.0:
        y_pos = 1. - np.random.random((batchsize, )) * smoothing
        if one_sided_smoothing:
            y_neg = np.zeros(batchsize, dtype='float32')
        else:
            y_neg = np.random.random((batchsize, )) * smoothing
    else:
        y_pos = np.ones(batchsize, dtype='float32')
        y_neg = np.zeros(batchsize, dtype='float32')

    return y_pos, y_neg


def plot_metrics(outfile, metrics_list, iterations_list, types,
                 metric_names=None, n_cols=2, legend=False, x_label=None,
                 y_label=None, wspace=None, hspace=None, figsize=8):

    assert isinstance(metrics_list, (list, tuple)) and \
        not isinstance(metrics_list, str)

    total_n_plots = len(metrics_list)
    if total_n_plots == 1:
        grid_cols, grid_rows = 1, 1
    elif total_n_plots == 2:
        grid_cols, grid_rows = 2, 1
    elif total_n_plots == 3 or total_n_plots == 4:
        grid_cols, grid_rows = 2, 2
    elif total_n_plots == 5 or total_n_plots == 6:
        grid_cols, grid_rows = 2, 3
    elif total_n_plots == 7 or total_n_plots == 8 or total_n_plots == 9:
        grid_cols, grid_rows = 3, 3
    elif total_n_plots == 10 or total_n_plots == 11 or total_n_plots == 12:
        grid_cols, grid_rows = 3, 4
    elif total_n_plots == 13 or total_n_plots == 14 or total_n_plots == 15:
        grid_cols, grid_rows = 3, 5
    elif total_n_plots == 16:
        grid_cols, grid_rows = 4, 4
    elif total_n_plots == 17 or total_n_plots == 18 or total_n_plots == 19 or total_n_plots == 19:
        grid_cols, grid_rows = 4, 5

    fig_w, fig_h = figsize * grid_cols, figsize * grid_rows

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(grid_rows, grid_cols)
    if wspace is not None and hspace is not None:
        gs.update(wspace=wspace, hspace=hspace)
    elif wspace is not None:
        gs.update(wspace=wspace)
    elif hspace is not None:
        gs.update(hspace=hspace)

    argsort_of_metric_names = np.argsort(metric_names) # keeps order between runs
    for ii in argsort_of_metric_names:

        metric = metrics_list[ii]
        current_cell = gs[ii // grid_cols, ii % grid_cols]
        ax = None

        if types[ii] == 'lines':
            ax = plt.subplot(current_cell)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            if isinstance(metric[0], (list, tuple, np.ndarray)):
                lines = []
                for jj, submetric in enumerate(metric):
                    if metric_names is not None:
                        label = metric_names[ii][jj]
                    else:
                        label = "line_%01d" % jj
                    line, = ax.plot(iterations_list[ii], submetric,
                                    color='C%d' % jj,
                                    label=label)
                    lines.append(line)
            else:
                if metric_names is not None:
                    label = metric_names[ii]
                else:
                    label = "line_01"
                line, = ax.plot(iterations_list[ii], metric, color='C0',
                                label=label)
                lines = [line]
            if ((not isinstance(legend, (list, tuple)) and legend) or
                    (isinstance(legend, (list, tuple)) and legend[ii])):
                lg = ax.legend(handles=lines, prop={'size': 16})

        elif types[ii] == 'scatter':
            ax = plt.subplot(current_cell)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            cmap = cm.tab10
            category_labels = metric[-1][..., 2]
            norm = colors.Normalize(vmin=np.min(category_labels), vmax=np.max(category_labels))
            cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            mapped_colors = cmapper.to_rgba(category_labels)
            unique_labels = list(set(category_labels))
            lines = ax.scatter(metric[-1][..., 0], metric[-1][..., 1],
                               color=mapped_colors,
                               label=unique_labels)
            patch = mpatches.Patch(color='silver', label=metric_names[ii])
            ax.legend(handles=[patch], prop={'size': 20})

        elif types[ii] == 'image-grid':
            imgs = metric[-1]
            n_images = len(imgs)
            inner_grid_width = int(np.sqrt(n_images))
            inner_grid = GridSpecFromSubplotSpec(inner_grid_width, inner_grid_width, current_cell, wspace=0.1, hspace=0.1)
            for i in range(n_images):
                inner_ax = plt.subplot(inner_grid[i])
                if imgs.ndim == 4:
                    inner_ax.imshow(imgs[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
                else:
                    inner_ax.imshow(imgs[i, :, :], cmap='gray', interpolation='none', vmin=0.0, vmax=1.0)
                inner_ax.axis('off')

        
        if ax is not None:
            if x_label is not None and not isinstance(x_label, (list, tuple)):
                ax.set_xlabel(x_label, color='k')
            elif isinstance(x_label, (list, tuple)) and ax is not None:
                ax.set_xlabel(x_label[ii], color='k')

            # Make the y-axis label, ticks and tick labels
            # match the line color.
            if y_label is not None and not isinstance(y_label, (list, tuple)):
                ax.set_ylabel(y_label, color='k')
            elif isinstance(y_label, (list, tuple)):
                ax.set_ylabel(y_label[ii], color='k')
            ax.tick_params('y', colors='k')

    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close(fig)


def change_losses_weight_over_time(current_epoch, max_epoch, type, scalar_loss):
    weighting_factor = np.min((1, (current_epoch) / max_epoch))
    if type == 'inc':
        return scalar_loss * weighting_factor
    elif type == 'dec':
        return scalar_loss * (1 - weighting_factor)
    elif type == 'hold':
        if current_epoch >= max_epoch:
            return scalar_loss
        else:
            return 0.0
    elif type == 'hold-inc':
        if current_epoch >= max_epoch:
            weighting_factor = np.min((1, (current_epoch - max_epoch) / max_epoch))
            return scalar_loss * weighting_factor
        else:
            return 0.0
    elif type == 'hold-dec':
        if current_epoch >= max_epoch:
            weighting_factor = np.min((1, (current_epoch - max_epoch) / max_epoch))
            return scalar_loss * (1 - weighting_factor)
        else:
            return 0.0
    elif type == 'halt':
        if current_epoch >= max_epoch:
            return 0.0
        else:
            return scalar_loss
    elif type == 'none':
        return scalar_loss
    else:
        raise ValueError("type must be 'inc', 'dec', 'hold', 'halt' or 'none'")

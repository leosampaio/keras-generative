import os
import sys
import time
import threading
import queue
import numpy as np
import h5py

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.models import load_model
from abc import ABCMeta, abstractmethod

from .utils import *
from core.losses import Loss

try:
    from .notifyier import *
except ImportError as e:
    print(e)
    print("You did not set a notifyier. Notifications will not be sent anywhere")


class BaseModel(metaclass=ABCMeta):
    '''
    Base class for non-conditional generative networks
    '''

    def __init__(self, **kwargs):

        if not hasattr(self, 'name'):
            raise Exception('You must give your model a reference name')

        if not hasattr(self, 'loss_names'):
            raise Exception("You must define your model's expected losses "
                            "in a loss_names attribute")
        if not hasattr(self, 'loss_names'):
            raise Exception("You must define your model's loss plot"
                            "organization in a loss_plot_organization attribute ")
        self.n_losses = len(self.loss_names)

        self.current_epoch = 0
        self.run_id = kwargs.get('run_id', 0)
        self.name = "{}_r{}".format(self.name, self.run_id)

        if 'input_shape' not in kwargs:
            raise Exception('Please specify input shape!')
        self.input_shape = kwargs['input_shape']

        self.output = kwargs.get('output', 'output')
        self.test_mode = kwargs.get('test_mode', False)

        self.trainers = {}
        self.last_epoch = 0  # recalculated from weights filename if it is loaded later
        self.label_smoothing = kwargs.get('label_smoothing', 0.0)
        self.input_noise = kwargs.get('input_noise', 0.0)

        self.checkpoint_every = kwargs.get('checkpoint_every', 1)
        self.notify_every = kwargs.get('notify_every', self.checkpoint_every)
        self.lr = kwargs.get('lr', 1e-4)

        # generic loss setup - start
        controlled_losses = kwargs.get('controlled_losses', [])
        self.losses = {}
        for loss_name in self.loss_names:

            # get correct string if it exists
            control_string = next((l for l in controlled_losses if l.startswith(loss_name)), False)
            if control_string:
                self.losses[loss_name] = Loss.from_control_string(control_string)
            else:
                self.losses[loss_name] = Loss()
        self.opt_states = None
        self.optimizers = None
        self.update_loss_weights()
        # generic loss setup - end

        # generic metric setup - start
        if kwargs.get('metrics') is not None:
            metrics = kwargs.get('metrics')
            try:
                self.metric_calculators = {m: getattr(self, "calculate_{}".format(m)) for m in metrics}
                self.metrics = {m: [] for m in metrics}
                self.metrics_calc_threads = {m: threading.Thread() for m in metrics}
                self.metric_types = {m: getattr(self, "{}_metric_type".format(m)) for m in metrics}
                for t in self.metrics_calc_threads.values():
                    t.start()
            except AttributeError:
                raise AttributeError("You must define calculate_ methods for "
                                     "all your metrics")
        else:
            self.metrics = None
        # generic metric setup - end

    def get_experiment_id(self):
        id = "{}_zdim{}_edim{}".format(self.name, self.z_dims, self.embedding_size)
        for l, loss in self.losses.items():
            id = "{}_L{}{}{}{}".format(id, l.replace('_', ''), loss.weight,
                                       loss.weight_control_type.replace('-', ''),
                                       loss.pivot_control_epoch)
        return id.replace('.', '')

    def _get_experiment_id(self):
        return self.get_experiment_id()
    experiment_id = property(_get_experiment_id)

    def main_loop(self, dataset, epochs=100, batchsize=100):
        '''
        Main learning loop
        '''

        # Create output directories if not exist
        self.dataset = dataset
        out_dir = os.path.join(self.output, self.experiment_id)
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        self.res_out_dir = os.path.join(out_dir, 'results')
        if not os.path.isdir(self.res_out_dir):
            os.mkdir(self.res_out_dir)

        self.wgt_out_dir = os.path.join(out_dir, 'weights')
        if not os.path.isdir(self.wgt_out_dir):
            os.mkdir(self.wgt_out_dir)

        self.tmp_out_dir = os.path.join(out_dir, 'tmp')
        if not os.path.isdir(self.tmp_out_dir):
            os.mkdir(self.tmp_out_dir)

        # Start training
        print('\n\n--- START TRAINING ---\n')
        num_data = len(dataset)
        self.batchsize = batchsize
        self.make_predict_functions()
        for e in range(self.last_epoch, epochs):
            perm = np.random.permutation(num_data)
            start_time = time.time()
            self.current_epoch = e + 1
            for b in range(0, num_data, batchsize):

                # account for division of data into batch size
                if batchsize > num_data - b:
                    continue
                bsize = batchsize
                indx = perm[b:b + bsize]

                # slice batch out of dataset
                x_batch, y_batch = self.make_batch(dataset, indx)

                # add optional input noise that decreases over time
                x_batch = add_input_noise(x_batch, curr_epoch=e, total_epochs=epochs, start_noise=self.input_noise)

                # finally, train and report status
                losses = self.train_on_batch(x_batch, y_batch=y_batch)
                self.update_loss_history(losses)

                print_current_progress(e, b, bsize, len(dataset), losses, elapsed_time=time.time() - start_time)

                # check for collapse scenario where G and D losses are equal
                did_collapse = self.did_collapse(losses)
                if did_collapse:
                    message = "[{}] {}. Stopped at Epoch #{}".format(self.experiment_id, did_collapse, self.current_epoch)
                    try:
                        notify_with_message(message, experiment_id=self.experiment_id)
                    except NameError:
                        pass
                    print(message)
                    exit()

                if self.test_mode:
                    print('\nFinish testing: %s' % self.experiment_id)
                    return

            # plot samples and losses and send notification if it's checkpoint time
            if ((self.current_epoch) % self.checkpoint_every) == 0:
                self.save_model(self.wgt_out_dir, self.current_epoch)
            if ((self.current_epoch) % self.notify_every) == 0:
                self.send_metrics_notification()

            elapsed_time = time.time() - start_time
            print('Took: {}s\n'.format(elapsed_time))
            self.did_train_over_an_epoch()
            self.update_loss_weights()

    def update_loss_weights(self):
        log_message = "New loss weights: "
        weight_delta = 0.
        for l, loss in self.losses.items():
            new_loss, delta = loss.update_weight_based_on_time(self.current_epoch)
            log_message = "{}{}:{}, ".format(log_message, l, new_loss)
            weight_delta = np.max((weight_delta, delta))

        if weight_delta > 0.1:
            for l in self.losses.values():
                l.reset_weight_from_last_significant_change()
            self.reset_optimizers()
            log_message = "{}. Did reset optmizers".format(log_message)
        print(log_message)

    def reset_optimizers(self):
        if self.opt_states is None:
            if self.optimizers is not None:
                self.opt_states = {k: opt.get_weights() for k, opt in self.optimizers.items()}
            return
        for k, opt in self.optimizers.items():
            opt.set_weights(self.opt_states[k])

    def update_loss_history(self, new_losses):
        for l, loss in self.losses.items():
            loss.update_history(new_losses[l])

    def make_predict_functions(self):
        """
        implement `_make_predict_function()` calls here if you plan on making
        predictions on multiple threads
        """
        pass

    def make_batch(self, dataset, indx):
        '''
        Get batch from dataset
        '''
        data = dataset.images[indx]
        labels = dataset.attrs[indx]
        return data, labels

    def did_collapse(self, losses):
        return False

    def save_model(self, out_dir, epoch):
        folder = os.path.join(out_dir, 'epoch_%05d' % epoch)
        if not os.path.isdir(folder):
            os.mkdir(folder)

        for k, v in self.trainers.items():
            filename = os.path.join(folder, '%s.hdf5' % (k))
            v.save_weights(filename)

    def store_to_save(self, name):
        self.trainers[name] = getattr(self, name)

    def load_model(self, folder):
        for k, v in self.trainers.items():
            try:
                filename = os.path.join(folder, "{}.hdf5".format(k))
                getattr(self, k).load_weights(filename)
            except OSError as e:
                print(e)
                print("Couldn't load {}. Starting from scratch".format(filename))
            except ValueError as e:
                print(e)
                print("Couldn't load {}. Starting from scratch".format(filename))

        # load epoch number
        epoch = int(folder.split('_')[-1].replace('/', ''))
        self.last_epoch = epoch

    def did_train_over_an_epoch(self):
        pass

    @abstractmethod
    def train_on_batch(self, x_batch, y_batch=None, compute_grad_norms=False):
        '''
        Plase override "train_on_batch" method in the derived model!
        '''
        pass

    def predict_images(self, z_sample):
        images = self.predict(z_sample)
        if images.shape[3] == 1:
            images = np.squeeze(imgs, axis=(3,))
        # images = np.clip(predictions * 0.5 + 0.5, 0.0, 1.0)
        return images

    @staticmethod
    def get_labels(batchsize, smoothing=0.0, one_sided_smoothing=True):
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

    def calculate_all_metrics(self):
        log_message = "Metrics for Epoch #{}: ".format(np.max((self.current_epoch - self.notify_every, 1)))
        self.did_go_through_all_metrics = threading.Event()
        for m, calculation_fun in self.metric_calculators.items():
            self.metrics_calc_threads[m].join()
            if not self.metrics[m]:
                log_message = "{} {}: {},".format(log_message, m, "waiting")
            elif self.metric_types[m] == 'lines':
                log_message = "{} {}: {},".format(log_message, m, self.metrics[m][-1])
            else:
                log_message = "{} {}: plotted,".format(log_message, m)
            finished_cgraph_use_event = threading.Event()

            self.metrics_calc_threads[m] = threading.Thread(target=self.metrics_calculation_worker,
                                                            args=(calculation_fun, self.metrics[m],
                                                                  self.metric_types[m],
                                                                  finished_cgraph_use_event,
                                                                  self.did_go_through_all_metrics))
            self.metrics_calc_threads[m].start()
            finished_cgraph_use_event.wait()
        return log_message

    def metrics_calculation_worker(self, calculation_fun, results, mtype,
                                   finished_cgraph_use_event, did_go_through_all_metrics):
        try:
            result = calculation_fun(finished_cgraph_use_event)
        except Exception as e:
            print("Exception while computing metrics: {}".format(repr(e)))
            if mtype == 'lines':
                if not results:
                    result = 0
                else:
                    result = results[-1]
        # wait for all metrics to finish
        # before appending the results
        did_go_through_all_metrics.wait()
        if mtype == 'lines':
            results.append(result)
        elif mtype == 'scatter' or mtype == 'image-grid':
            if not results:
                results.append(result)
            else:
                results[-1] = result

    def calculate_samples(self, finished_cgraph_use_event):
        imgs = self.predict(self.samples)
        finished_cgraph_use_event.set()
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs
    samples_metric_type = 'image-grid'

    def plot_all_metrics(self, outfile):
        if all(d for d in self.metrics.values()):
            l_metrics, l_iters, l_names, l_types = self.get_loss_metrics_for_plot(self.loss_plot_organization)
            lw_metrics, lw_iters, lw_names, lw_types = self.get_loss_weight_metrics_for_plot()
            e_metrics, e_iters, e_names, e_types = self.get_extra_metrics_for_plot()
            metrics, names = list(self.metrics.values()), list(self.metrics.keys())
            iters = [list(range(self.notify_every, self.current_epoch, self.notify_every))] * len(self.metrics)
            types = list(self.metric_types[k] for k in self.metrics.keys())
            plot_metrics(outfile,
                         metrics_list=metrics + e_metrics + l_metrics + lw_metrics,
                         iterations_list=iters + e_iters + l_iters + lw_iters,
                         metric_names=names + e_names + l_names + lw_names,
                         types=types + e_types + l_types + lw_types,
                         legend=True,
                         figsize=8,
                         wspace=0.15)
            self.did_go_through_all_metrics.set()
            return True
        else:
            self.did_go_through_all_metrics.set()
            return False

    def get_loss_metrics_for_plot(self, plot_organization):
        metrics = []
        iters = []
        names = []
        for l in plot_organization:
            if isinstance(l, (tuple, list)):
                submetrics, subiters, subnames, _ = self.get_loss_metrics_for_plot(l)
                metrics.append(submetrics)
                iters.append(subiters[0])
                names.append(subnames)
            else:
                metrics.append(self.losses[l].history)
                iters.append(list(range(len(self.losses[l].history))))
                names.append(l)
        return metrics, iters, names, ['lines'] * len(metrics)

    def get_loss_weight_metrics_for_plot(self):
        metrics = []
        names = []
        contrast_increment = 0  # increment to help loss weight visualization
        for l in self.loss_names:
            metrics.append(np.array(self.losses[l].weight_history) + contrast_increment)
            names.append(l)
            contrast_increment += 0.005
        iters = list(range(len(metrics[0])))
        return [metrics], [iters], [names], ['lines']

    def get_extra_metrics_for_plot(self):
        """
        should return any metrics you want to plot together with the 
        frequent ones

        returns 5 lists: 
            metrics_list, iterations_list, metric_names, metric_types
        """
        return [[]] * 4

    def load_precalculated_features_if_they_exist(self, feature_type, has_labels=True):
        filename = os.path.join(
            self.tmp_out_dir,
            "precalculated_features_{}_e{}.h5".format(feature_type, self.current_epoch))
        if os.path.exists(filename):
            with h5py.File(filename, 'r') as hf:
                x = hf['feats'][:]
                if has_labels:
                    y = hf['labels'][:]
                    return x, y
                else:
                    return x
        else:
            if has_labels:
                return False, False
            else:
                return False

    def save_precalculated_features(self, feature_type, X, Y=None):
        start = time.time()
        filename = os.path.join(
            self.tmp_out_dir,
            "precalculated_features_{}_e{}.h5".format(feature_type, self.current_epoch))
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("feats",  data=X)
            if Y is not None:
                hf.create_dataset("labels",  data=Y)
        print("[Precalc] Saving {} took {}s".format(feature_type, time.time() - start))

    def send_metrics_notification(self):
        outfile = os.path.join(self.res_out_dir, "epoch_{:04}_metrics.png".format(self.current_epoch))
        log_message = self.calculate_all_metrics()
        did_plot = self.plot_all_metrics(outfile)
        print(log_message)
        try:
            message = "[{}] Epoch #{:04}".format(self.experiment_id, self.current_epoch)
            notify_with_message(log_message, self.experiment_id)
            if did_plot:
                notify_with_image(outfile, experiment_id=self.experiment_id, message=message)
        except NameError as e:
            print(e)

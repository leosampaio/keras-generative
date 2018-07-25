import os
import sys
import time
import threading
import queue
import numpy as np
import h5py

from abc import ABCMeta, abstractmethod

from models.utils import print_current_progress, plot_metrics
from core.losses import Loss
import metrics

try:
    from core.notifyier import notify_with_message, notify_with_image
except ImportError as e:
    print(repr(e))
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
        self.current_fract_epoch = 0.
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

        self.checkpoint_every = kwargs.get('checkpoint_every', "1")
        self.notify_every = kwargs.get('notify_every', self.checkpoint_every)
        self.lr = kwargs.get('lr', 1e-4)
        self.z_dims = kwargs.get('z_dims', 100)

        # generic loss setup - start
        controlled_losses = kwargs.get('controlled_losses', [])
        if controlled_losses is None:
            controlled_losses = []
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
            desired_metrics = kwargs.get('metrics')
            self.metrics = {m: metrics.build_metric_by_name(m, experiment_id=self.experiment_id, **kwargs) for m in desired_metrics}
        else:
            self.metrics = None
        # generic metric setup - end

    def get_experiment_id(self):
        id = "{}_zdim{}".format(self.name, self.z_dims)
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

        # convert checkpoints to img-processed counts if not already
        # floats are interpreted as epoch fractions
        try:
            self.notify_every = int(self.notify_every)
        except ValueError:
            self.notify_every = int(float(self.notify_every) * len(dataset))
        try:
            self.checkpoint_every = int(self.checkpoint_every)
        except ValueError:
            self.checkpoint_every = int(float(self.checkpoint_every) * len(dataset))

        # Create output directories if not exist
        self.dataset = dataset
        self.batchsize = batchsize
        self.processed_images = 0
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
        for e in range(self.last_epoch, epochs):
            start_time = time.time()
            self.current_epoch = e + 1
            for x_batch, y_batch, batch_index in dataset.generator(batchsize=self.batchsize):

                # finally, train and report status
                losses = self.train_on_batch(x_batch, y_batch=y_batch)
                self.update_loss_history(losses)
                self.processed_images += self.batchsize
                self.current_fract_epoch = self.processed_images / len(self.dataset)

                print_current_progress(e, batch_index,
                                       batch_size=self.batchsize,
                                       dataset_length=len(self.dataset),
                                       losses=self.losses,
                                       elapsed_time=time.time() - start_time)

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
                if self.processed_images % self.checkpoint_every < (self.processed_images - self.batchsize) % self.checkpoint_every:
                    self.save_model(self.wgt_out_dir, self.processed_images)
                if self.processed_images % self.notify_every < (self.processed_images - self.batchsize) % self.notify_every:
                    self.send_metrics_notification()

                self.update_loss_weights()

            elapsed_time = time.time() - start_time
            print('Took: {}s\n'.format(elapsed_time))
            self.did_train_over_an_epoch()

    def update_loss_weights(self):
        weight_delta = 0.
        for l, loss in self.losses.items():
            new_loss, delta = loss.update_weight_based_on_time(self.current_fract_epoch)
            weight_delta = np.max((weight_delta, delta))

        if weight_delta > 0.1:
            for l in self.losses.values():
                l.reset_weight_from_last_significant_change()
            self.reset_optimizers()
            print("[TRAINING] Did reset optmizers.")

    def reset_optimizers(self):
        if self.opt_states is None:
            if self.optimizers is not None:
                self.opt_states = {k: opt.get_weights() for k, opt in self.optimizers.items()}
            return
        for k, opt in self.optimizers.items():
            opt.set_weights(self.opt_states[k])

    def update_loss_history(self, new_losses):
        for l, loss in self.losses.items():
            loss.update_history(new_losses[l] / self.batchsize)

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

    def gather_data_for_metric(self, data_type):
        data = self.load_precomputed_features_if_they_exist(data_type)
        if not data:
            try:
                gather_func = getattr(self, "compute_{}".format(data_type))
            except AttributeError:
                raise AttributeError("One of your metrics requires a "
                                     "compute_{} method".format(data_type))
            data = gather_func()
        return data

    def compute_all_metrics(self):
        log_message = "[Metrics] Image #{} Epoch #{}: ".format(self.processed_images, self.current_fract_epoch)
        for m, metric in self.metrics.items():
            input_data = self.gather_data_for_metric(metric.input_type)
            metric.compute_in_parallel(input_data)
            log_message = "{} {}: {},".format(log_message, m, metric.last_value_repr)
        return log_message

    def plot_all_metrics(self, outfile):
        if all(d.is_ready_for_plot() for d in self.metrics.values()):
            metrics, iters, names, types = self.get_metrics_for_plot()
            l_metrics, l_iters, l_names, l_types = self.get_loss_metrics_for_plot(self.loss_plot_organization)
            lw_metrics, lw_iters, lw_names, lw_types = self.get_loss_weight_metrics_for_plot()
            e_metrics, e_iters, e_names, e_types = self.get_extra_metrics_for_plot()
            plot_metrics(outfile,
                         metrics_list=metrics + e_metrics + l_metrics + lw_metrics,
                         iterations_list=iters + e_iters + l_iters + lw_iters,
                         metric_names=names + e_names + l_names + lw_names,
                         types=types + e_types + l_types + lw_types,
                         legend=True,
                         figsize=8,
                         wspace=0.15)
            return True
        else:
            return False

    def get_loss_metrics_for_plot(self, plot_organization):
        metrics, iters, names = [], [], []
        for l in plot_organization:
            if isinstance(l, (tuple, list)):
                submetrics, subiters, subnames, _ = self.get_loss_metrics_for_plot(l)
                metrics.append(submetrics)
                iters.append(subiters[0])
                names.append(subnames)
            else:
                metrics.append(self.losses[l].history)
                iters.append(1)
                names.append(l)
        return metrics, iters, names, ['lines'] * len(metrics)

    def get_loss_weight_metrics_for_plot(self):
        metrics, names = [], []
        contrast_increment = 0  # increment to help loss weight visualization
        for l, loss in self.losses.items():
            metrics.append(np.array(loss.weight_history) + contrast_increment)
            names.append(l)
            contrast_increment += 0.005
        return [metrics], [1], [names], ['lines']

    def get_metrics_for_plot(self):
        metrics, iters, names, types = [], [], [], []
        for m, metric in self.metrics.items():
            data = metric.get_data_for_plot()
            metrics.append(data)
            names.append(m)
            iters.append(self.notify_every)
            types.append(metric.plot_type)
        return metrics, iters, names, types

    def get_extra_metrics_for_plot(self):
        """
        returns 4 lists: 
            metrics_list, iterations_list, metric_names, metric_types
        """
        return [[]] * 4

    def load_precomputed_features_if_they_exist(self, feature_type):
        filename = os.path.join(
            self.tmp_out_dir,
            "precomputed_{}_pi{}.h5".format(feature_type, self.processed_images))
        if os.path.exists(filename):
            with h5py.File(filename, 'r') as hf:
                x = hf['feats'][:]
                data = [x]
                if 'labels' in hf:
                    y = hf['labels'][:]
                    data.append(y)
                if 'x_test' in hf:
                    x_test, y_test = hf['x_test'][:], hf['y_test'][:]
                    data += [x_test, y_test]
                return data
        else:
            return False

    def save_precomputed_features(self, feature_type, X, Y=None, test_set=None):
        start = time.time()
        filename = os.path.join(
            self.tmp_out_dir,
            "precomputed_{}_pi{}.h5".format(feature_type, self.processed_images))
        with h5py.File(filename, 'w') as hf:
            hf.create_dataset("feats",  data=X)
            if Y is not None:
                hf.create_dataset("labels",  data=Y)
            if test_set is not None:
                x_test, y_test = test_set
                hf.create_dataset("x_test",  data=x_test)
                hf.create_dataset("y_test",  data=y_test)
        print("[Precalc] Saving {} took {}s".format(feature_type, time.time() - start))

    def send_metrics_notification(self):
        outfile = os.path.join(self.res_out_dir, "pi_{:04}_metrics.png".format(self.processed_images))
        log_message = self.compute_all_metrics()
        did_plot = self.plot_all_metrics(outfile)
        print(log_message)
        try:
            message = "[{}] ProcImgs #{:04} Epoch #{}".format(self.experiment_id, self.processed_images, self.current_fract_epoch)
            notify_with_message(log_message, self.experiment_id)
            if did_plot:
                notify_with_image(outfile, experiment_id=self.experiment_id, message=message)
        except NameError as e:
            print(repr(e))

import h5py
import numpy as np
import itertools
import glob
import threading
import os

from datasets import svhn
from datasets import mnist
from datasets import moving_mnist
from datasets import lsun
from datasets import cifar10
from datasets import celeba
from datasets import mixture

import sklearn as sk
import keras


class Dataset(object):

    def __init__(self, name):
        self.name = name
        self.images = None

    def __len__(self):
        return len(self.images)

    def _get_shape(self):
        return self.images.shape

    shape = property(_get_shape)

    def get_random_fixed_batch(self, n=32):
        np.random.seed(14)
        perm = np.random.permutation(len(self.images))
        x_data = self.images[perm[:n]]
        y_data = np.argmax(self.attrs[perm[:n]], axis=1)
        np.random.seed()
        return x_data, y_data

    def get_test_set(self):
        if hasattr(self, 'x_test') and hasattr(self, 'y_test'):
            return self.x_test, self.y_test
        else:
            raise KeyError("This dataset does not have a test set")

    def get_random_perm_of_test_set(self, n):
        if hasattr(self, 'x_test') and hasattr(self, 'y_test'):
            np.random.seed(14)
            perm = np.random.permutation(len(self.x_test))
            np.random.seed()
            return self.x_test[perm[:n]], self.y_test[perm[:n]]
        else:
            raise KeyError("This dataset does not have a test set")

    def has_test_set(self):
        return hasattr(self, 'x_test')

    def generator(self, batchsize):
        general_cursor = 0
        perm = np.random.permutation(len(self.images))
        n_data = len(self.images)
        for b in range(0, n_data, batchsize):
            if batchsize > n_data - b:
                continue
            general_cursor += batchsize
            indx = perm[b:b + batchsize]
            x_data, y_data = self.images[indx], np.zeros((batchsize,), dtype=np.uint8)
            yield x_data, y_data, general_cursor


class ConditionalDataset(Dataset):

    def __init__(self, name):
        super(ConditionalDataset, self).__init__(name)
        self.attrs = None
        self.attr_names = None

    def get_random_fixed_batch(self, n=32):
        if len(self.attrs.shape) == 2:
            y_categorical = np.argmax(self.attrs, axis=1)
            _, x_data, _, y_data = sk.model_selection.train_test_split(self.images,
                                                                       y_categorical,
                                                                       stratify=y_categorical,
                                                                       test_size=n,
                                                                       random_state=14)
        else:
            perm = np.random.permutation(len(self.images))
            x_data, y_data = self.images[perm[:n]], self.attrs[perm[:n]]

        return x_data, y_data

    def generator(self, batchsize):
        general_cursor = 0
        n_data = len(self.images)
        perm = np.random.permutation(n_data)
        for b in range(0, n_data, batchsize):
            if batchsize > n_data - b:
                continue
            general_cursor += batchsize
            indx = perm[b:b + batchsize]
            x_data, y_data = self.images[indx], self.attrs[indx]
            yield x_data, y_data, general_cursor


class StackedDataset(Dataset):

    def _get_shape(self):
        return list(self.images.shape[:-1]) + [3]

    shape = property(_get_shape)

    def get_random_fixed_batch(self, n=32):
        np.random.seed(14)
        perm = np.random.permutation(len(self.images))
        perm2 = np.random.permutation(len(self.images))
        perm3 = np.random.permutation(len(self.images))
        x_data = np.concatenate([self.images[perm[:n]], self.images[perm2[:n]], self.images[perm3[:n]]], axis=-1)
        y_data = np.vstack((np.argmax(self.attrs[perm[:n]], axis=1), np.argmax(self.attrs[perm2[:n]], axis=1), np.argmax(self.attrs[perm3[:n]], axis=1)))
        y_data = np.sum(y_data*np.transpose([[100, 10, 1]]), axis=0)
        np.random.seed()
        return x_data, y_data

    def get_test_set(self):
        if hasattr(self, 'x_test') and hasattr(self, 'y_test'):

            np.random.seed(14)
            perm = np.random.permutation(len(self.x_test))
            np.random.seed(15)
            perm2 = np.random.permutation(len(self.x_test))
            np.random.seed()
            x_test = np.concatenate((self.x_test, self.x_test[perm], self.x_test[perm2]), axis=-1)
            y_test = np.argmax(self.y_test, axis=1)
            y_test = np.sum((y_test*100, y_test[perm]*10, y_test[perm2]), axis=0)
            return x_test, self.y_test
        else:
            raise KeyError("This dataset does not have a test set")

    def get_random_perm_of_test_set(self, n):
        if hasattr(self, 'x_test') and hasattr(self, 'y_test'):
            np.random.seed(14)
            perm = np.random.permutation(len(self.x_test))
            np.random.seed()
            x_data = np.concatenate([self.x_test[perm[:(n*3)]][i:i+n] for i in range(0, n*3, n)], axis=-1)
            y_data = [self.y_test[perm[:(n*3)]][i:i+n] for i in range(0, n*3, n)]
            y_data = np.sum(y_data*np.transpose([[100, 10, 1]]), axis=0)
            return x_data, y_data
        else:
            raise KeyError("This dataset does not have a test set")

    def has_test_set(self):
        return hasattr(self, 'x_test')

    def generator(self, batchsize):
        general_cursor = 0
        perm = np.random.permutation(len(self.images))
        n_data = len(self.images)
        for b in range(0, n_data, batchsize):
            if batchsize*3 > n_data - b:
                continue
            general_cursor += batchsize
            indx = perm[b:b + batchsize]
            indx2 = perm[b + batchsize:(b + batchsize*2)%n_data]
            indx3 = perm[(b + batchsize*2)%n_data:(b + batchsize*3)%n_data]
            x_data = np.concatenate((self.images[indx], self.images[indx2], self.images[indx3]), axis=-1)
            y_data = np.vstack((np.argmax(self.attrs[indx], axis=1), np.argmax(self.attrs[indx2], axis=1), np.argmax(self.attrs[indx3], axis=1)))
            y_data = np.sum(y_data*np.transpose([[100, 10, 1]]), axis=0)
            y_data = keras.utils.to_categorical(y_data, num_classes=1000)
            yield x_data, y_data, general_cursor


class SimulatedAnomalyDetectionDataset(Dataset):

    def __init__(self, name, x, y, anomaly_class=0, test_set=None):
        super().__init__(name)
        self.full_set = x
        self.y = y

        # speedup future lookups by prepreparing without the anomaly
        ncols = y.shape[1]
        dtype = y.dtype.descr * ncols
        struct = y.view(dtype)
        self.uniq = np.unique(struct)
        anom_indx = np.where((self.y == tuple(self.uniq[anomaly_class])).all(axis=1))[0]
        norm_indx = np.where(~(self.y == tuple(self.uniq[anomaly_class])).all(axis=1))[0]
        self.anom_x, self.anom_y = self.full_set[anom_indx], self.y[anom_indx]
        self.norm_x, self.norm_y = self.full_set[norm_indx], self.y[norm_indx]
        self.anomaly_label = np.ones((len(y),), dtype=np.float32)
        self.anomaly_label[anom_indx] = -1.

        if test_set is not None:
            self.x_test, y = test_set
            self.y_test = np.ones((len(y),), dtype=np.float32)
            self.y_test[np.where(y == anomaly_class)[0]] = -1.

    def get_random_fixed_batch(self, n=32):
        np.random.seed(14)
        perm = np.random.permutation(len(self.norm_x))
        np.random.seed()
        x_data = self.norm_x[perm[:n]]
        y_data = np.ones((n,))
        return x_data, y_data

    def generator(self, batchsize):
        general_cursor = 0
        n_data = len(self.norm_x)
        perm = np.random.permutation(n_data)
        for b in range(0, n_data, batchsize):
            if batchsize > n_data - b:
                continue
            general_cursor += batchsize
            indx = perm[b:b + batchsize]
            x_data, y_data = self.norm_x[indx], self.norm_y[indx]
            yield x_data, y_data, general_cursor

    def __len__(self):
        return len(self.norm_x)

    def _get_shape(self):
        return self.norm_x.shape

    shape = property(_get_shape)


class CrossDomainDatasets(object):

    def __init__(self, name, anchor_dataset, mirror_dataset):
        self.name = name
        assert len(anchor_dataset.attr_names) == len(mirror_dataset.attr_names)
        self.anchor = anchor_dataset
        self.mirror = mirror_dataset
        self.counter = itertools.count(0)

        # speedup future lookups by prepreparing slices
        labels = self.anchor.attrs
        ncols = labels.shape[1]
        dtype = labels.dtype.descr * ncols
        struct = labels.view(dtype)
        uniq = np.unique(struct)
        self.uniq_y = uniq.view(labels.dtype).reshape(-1, ncols)
        self.slices_p = {tuple(m): np.where((self.mirror.attrs == tuple(m)).all(axis=1))[0] for m in self.uniq_y}
        self.slices_n = {tuple(m): np.where(~(self.mirror.attrs == tuple(m)).all(axis=1))[0] for m in self.uniq_y}
        self.shuffle_p_n_samples()

        # the mirror permutation allows us to keep the model API the same
        # while providing good sampling across both datasets
        self.mirror_permutation = np.random.permutation(len(self.mirror))
        self.current_m_index = 0

    def generator(self, batchsize):
        general_cursor = 0
        n_data = len(self.anchor)
        perm = np.random.permutation(n_data)
        for b in range(0, n_data, batchsize):
            if batchsize > n_data - b:
                continue
            general_cursor += batchsize
            indx = perm[b:b + batchsize]
            x_data, y_data = self.get_triplets(indx)
            yield x_data, y_data, general_cursor

    def get_unlalabeled_pairs(self, idx, b_idx=None):
        a_x = self.anchor.images[idx]
        if b_idx is None:
            b_idx = self.get_perm_mirror_indices(len(idx))
        b_x = self.mirror.images[b_idx]

        return (a_x, b_x), (idx, b_idx)

    def get_triplets(self, idx):
        a_x, a_y = self.anchor.images[idx], self.anchor.attrs[idx]
        p_idx = [self.slices_p[tuple(y)][self.slices_p_perm[tuple(y)][next(self.counter) % len(self.slices_p_perm[tuple(y)])]] for y in a_y]
        n_idx = [self.slices_n[tuple(y)][self.slices_n_perm[tuple(y)][next(self.counter) % len(self.slices_n_perm[tuple(y)])]] for y in a_y]
        p_x, p_y = self.mirror.images[p_idx], self.mirror.attrs[p_idx]
        n_x, n_y = self.mirror.images[n_idx], self.mirror.attrs[n_idx]

        if next(self.counter) > 2 * len(self.mirror):
            self.shuffle_p_n_samples()

        return (a_x, p_x, n_x), (a_y, p_y, n_y)

    def get_positive_pairs(self, idx):
        a_x, a_y = self.anchor.images[idx], self.anchor.attrs[idx]
        p_idx = [self.slices_p[tuple(y)][self.slices_p_perm[tuple(y)][next(self.counter) % len(self.slices_p_perm[tuple(y)])]] for y in a_y]
        p_x, p_y = self.mirror.images[p_idx], self.mirror.attrs[p_idx]

        return (a_x, p_x), (a_y, p_y)

    def get_negative_pairs(self, idx):
        a_x, a_y = self.anchor.images[idx], self.anchor.attrs[idx]
        n_idx = [self.slices_n[tuple(y)][self.slices_n_perm[tuple(y)][next(self.counter) % len(self.slices_n_perm[tuple(y)])]] for y in a_y]
        n_x, n_y = self.mirror.images[n_idx], self.mirror.attrs[n_idx]

        return (a_x, n_x), (a_y, n_y)

    def shuffle_p_n_samples(self):
        self.slices_p_perm = {tuple(y): np.random.permutation(np.arange(self.slices_p[tuple(y)].shape[0])) for y in self.uniq_y}
        self.slices_n_perm = {tuple(y): np.random.permutation(np.arange(self.slices_n[tuple(y)].shape[0])) for y in self.uniq_y}

    def get_perm_mirror_indices(self, bsize):
        size = min(bsize, len(self.mirror) - self.current_m_index)
        idx = self.mirror_permutation[self.current_m_index:self.current_m_index + size]

        self.current_m_index = self.current_m_index + size

        # if we reached the end, repermute and complete the batch
        if size < bsize:
            remaining_size = bsize - size
            self.mirror_permutation = np.random.permutation(len(self.mirror))
            remaining_idx = self.mirror_permutation[0:remaining_size]
            if remaining_idx:
                idx.append(remaining_idx)
            self.current_m_index = remaining_size

        return idx

    def __len__(self):
        return len(self.anchor)

    def _get_mirror_len(self):
        return len(self.mirror)

    def _get_shape(self):
        return self.anchor.shape

    def _get_attr_names(self):
        return self.anchor.attr_names

    attr_names = property(_get_attr_names)
    shape = property(_get_shape)
    mirror_len = property(_get_mirror_len)


class TimeCorelatedDataset(Dataset):

    def __init__(self, name, x_data, input_n_frames=4):
        self.name = name
        self.data = x_data
        self.input_n_frames = input_n_frames

    def __len__(self):
        return len(self.data)

    def _get_shape(self):
        return (self.data.shape[0],) + self.data.shape[2:4] + (self.data.shape[4] * self.input_n_frames,)

    def get_pairs(self, idx):
        selected_videos = self.data[idx, ...]
        num_frames = [len(x) for x in selected_videos]
        max_starting_frames_allowed = [n - (self.input_n_frames * 2) for n in num_frames]
        starting_frames = [np.random.randint(0, m) for m in max_starting_frames_allowed]

        input_frames = [x[start:(start + self.input_n_frames)] for x, start in zip(selected_videos, starting_frames)]
        prediction_ground_truth = [x[(start + self.input_n_frames):(start + self.input_n_frames * 2)] for x, start in zip(selected_videos, starting_frames)]

        input_frames = [self.concatenate_frames_over_channels(x) for x in input_frames]
        prediction_ground_truth = [self.concatenate_frames_over_channels(x) for x in prediction_ground_truth]
        return np.array(input_frames), np.array(prediction_ground_truth)

    def concatenate_frames_over_channels(self, x):
        t = np.transpose(x, (1, 2, 0, 3))
        return np.array(np.reshape(t, t.shape[0:2] + (t.shape[2] * t.shape[3],)))

    def undo_concatenated_frames_over_channel(self, x):
        t = np.reshape(x, x.shape[0:2] + (self.input_n_frames, x.shape[2] // self.input_n_frames))
        t = np.transpose(t, (2, 0, 1, 3))
        return t

    def get_original_frames_from_processed_samples(self, X):
        orig_x = [self.undo_concatenated_frames_over_channel(x) for x in X]
        return np.array(orig_x)

    def get_some_random_samples(self):
        idx = np.random.randint(0, len(self.data), 4)
        a_data, b_data = self.get_pairs(idx)
        return a_data, b_data

    shape = property(_get_shape)


class BufferedDataset(object):

    def __init__(self, datapath):
        path = os.path.join(datapath, "*.npy")
        self.filepaths = glob.glob(path)
        self.n_buffers = len(self.filepaths)
        self.buffer = np.load(self.filepaths[0])
        self.current_buffer = 0
        self.current_mirror_buffer = 1

        self.load_mirror_buffer_in_background()

    def load_mirror_buffer_in_background(self):
        self.loading_thread = threading.Thread(target=self._load_mirror_buffer_worker)
        self.loading_thread.start()

    def swap_buffers(self):
        self.loading_thread.join()
        self.buffer = self.mirror_buffer
        self.mirror_buffer = None
        self.current_buffer = (self.current_buffer + 1) % self.n_buffers
        self.current_mirror_buffer = (self.current_buffer + 1) % self.n_buffers
        self.load_mirror_buffer_in_background()
        print("Swapped buffer {} in. Loading buffer {}...".format(self.current_buffer, self.current_mirror_buffer))
        did_go_around = self.current_buffer % self.n_buffers == 0
        return did_go_around

    def get_random_fixed_batch(self, n=32):
        np.random.seed(14)
        perm = np.random.permutation(len(self.buffer))
        x_data = self.buffer[perm[:n]]
        np.random.seed()
        return x_data, np.zeros((n,))

    def generator(self, batchsize):
        did_finish_an_epoch = False
        general_cursor = 0
        while not did_finish_an_epoch:
            perm = np.random.permutation(len(self.buffer))
            n_data = len(self.buffer)
            for b in range(0, n_data, batchsize):
                if batchsize > n_data - b:
                    continue
                general_cursor += batchsize
                indx = perm[b:b + batchsize]
                x_data, y_data = self.buffer[indx], np.zeros((batchsize,), dtype=np.uint8)
                yield x_data, y_data, general_cursor
            did_finish_an_epoch = self.swap_buffers()

    def _load_mirror_buffer_worker(self):
        self.mirror_buffer = np.load(self.filepaths[self.current_mirror_buffer])

    def __len__(self):
        return len(self.buffer) * self.n_buffers

    def _get_shape(self):
        return [len(self)] + list(self.buffer.shape[1:])

    shape = property(_get_shape)


class LargeDataset(object):

    def __init__(self, datapath, buffer_size=50000):

        self.datapath = datapath
        print("Reading H5 file... ", end="")
        self.file = h5py.File(self.datapath, 'r')
        print("Done!")
        self.data_ref = self.file['data']
        self.n_buffers = len(self.data_ref) // buffer_size
        self.buffer_size = buffer_size
        self.buffer = self.data_ref[0:self.buffer_size, ...]
        self.current_buffer = 0
        self.current_mirror_buffer = 1

        self.load_mirror_buffer_in_background()

    def get_test_set(self):
        if hasattr(self, 'x_test') and hasattr(self, 'y_test'):
            return self.x_test, self.y_test
        else:
            raise KeyError("This dataset does not have a test set")

    def has_test_set(self):
        return hasattr(self, 'x_test')

    def load_mirror_buffer_in_background(self):
        self.loading_thread = threading.Thread(target=self._load_mirror_buffer_worker)
        self.loading_thread.start()

    def swap_buffers(self):
        self.loading_thread.join()
        self.buffer = self.mirror_buffer
        self.mirror_buffer = None
        self.current_buffer = (self.current_buffer + 1) % self.n_buffers
        self.current_mirror_buffer = (self.current_buffer + 1) % self.n_buffers
        self.load_mirror_buffer_in_background()
        print("Swapped buffer {} in. Loading buffer {}...".format(self.current_buffer, self.current_mirror_buffer))
        did_go_around = self.current_buffer % self.n_buffers == 0
        return did_go_around

    def get_random_fixed_batch(self, n=32):
        np.random.seed(14)
        perm = np.random.permutation(len(self.buffer))
        x_data = self.buffer[perm[:n]]
        np.random.seed()
        return x_data, np.zeros((n,))

    def generator(self, batchsize):
        did_finish_an_epoch = False
        general_cursor = 0
        while not did_finish_an_epoch:
            perm = np.random.permutation(len(self.buffer))
            n_data = len(self.buffer)
            for b in range(0, n_data, batchsize):
                if batchsize > n_data - b:
                    continue
                general_cursor += batchsize
                indx = perm[b:b + batchsize]
                x_data, y_data = self.buffer[indx], np.zeros((batchsize,), dtype=np.uint8)
                yield x_data, y_data, general_cursor
            did_finish_an_epoch = self.swap_buffers()

    def _load_mirror_buffer_worker(self):
        current_pointer = self.buffer_size * self.current_mirror_buffer
        next_stop = np.min((current_pointer + self.buffer_size, len(self.data_ref)))
        self.mirror_buffer = self.data_ref[current_pointer:next_stop, ...]

    def __len__(self):
        return len(self.data_ref)

    def _get_shape(self):
        return [len(self)] + list(self.buffer.shape[1:])

    shape = property(_get_shape)


def load_dataset(dataset_name):
    if dataset_name == 'mnist':
        dataset = ConditionalDataset(name=dataset_name.replace('-', ''))
        dataset.images, dataset.attrs, dataset.x_test, dataset.y_test, dataset.attr_names = mnist.load_data()
    elif dataset_name == 'stacked-mnist':
        dataset = StackedDataset(name=dataset_name.replace('-', ''))
        dataset.images, dataset.attrs, dataset.x_test, dataset.y_test, dataset.attr_names = mnist.load_data()
    elif dataset_name == 'mnist-original':
        dataset = ConditionalDataset(name=dataset_name.replace('-', ''))
        dataset.images, dataset.attrs, dataset.x_test, dataset.y_test, dataset.attr_names = mnist.load_data(original=True)
    elif dataset_name == 'mnist-rgb':
        dataset = ConditionalDataset(name=dataset_name.replace('-', ''))
        dataset.images, dataset.attrs, dataset.x_test, dataset.y_test, dataset.attr_names = mnist.load_data(use_rgb=True)
    elif dataset_name == 'svhn':
        dataset = ConditionalDataset(name=dataset_name.replace('-', ''))
        dataset.images, dataset.attrs, dataset.x_test, dataset.y_test, dataset.attr_names = svhn.load_data()
    elif dataset_name == 'svhn-extra':
        dataset = ConditionalDataset(name=dataset_name.replace('-', ''))
        dataset.images, dataset.attrs, dataset.x_test, dataset.y_test, dataset.attr_names = svhn.load_data(include_extra=True)
    elif dataset_name == 'mnist-svhn':
        anchor = load_dataset('mnist-rgb')
        mirror = load_dataset('svhn')
        dataset = CrossDomainDatasets(dataset_name.replace('-', ''), anchor, mirror)
    elif dataset_name == 'cifar10':
        dataset = ConditionalDataset(name=dataset_name.replace('-', ''))
        dataset.images, dataset.attrs, dataset.x_test, dataset.y_test, dataset.attr_names = cifar10.load_data()
    elif dataset_name == 'moving-mnist':
        data = moving_mnist.load_data()
        dataset = TimeCorelatedDataset(dataset_name.replace('-', ''), data)
    elif dataset_name == 'lsun-bedroom':
        datapath = lsun.load_data()
        dataset = LargeDataset(datapath)
    elif dataset_name == 'celeba':
        datapath = celeba.load_data()
        dataset = LargeDataset(datapath, buffer_size=20000)
    elif dataset_name == 'celeba-128':
        datapath = celeba.load_data(image_size=128)
        dataset = LargeDataset(datapath, buffer_size=5000)
    elif dataset_name == 'celeba-crop-128':
        datapath = celeba.load_data(image_size=128, center_crop=True)
        dataset = LargeDataset(datapath, buffer_size=5000)
    elif dataset_name == 'celeba-crop-64':
        datapath = celeba.load_data(image_size=64, center_crop=True)
        dataset = LargeDataset(datapath, buffer_size=20000)
    elif dataset_name == 'synthetic-8ring':
        dataset = ConditionalDataset(name=dataset_name.replace('-', ''))
        dataset.images, dataset.attrs, dataset.attr_names = mixture.load_data(type="ring", n=8, std=.05, r=1, density=5000)
    elif dataset_name == 'synthetic-25grid':
        dataset = ConditionalDataset(name=dataset_name.replace('-', ''))
        dataset.images, dataset.attrs, dataset.attr_names = mixture.load_data(type="grid", n=25, std=.05, density=2500)
    elif dataset_name == 'synthetic-high-dim':
        dataset = ConditionalDataset(name=dataset_name.replace('-', ''))
        dataset.images, dataset.attrs, dataset.attr_names = mixture.load_data(type="high-dim", n=10, d=1200, std=1., density=5000)
    elif dataset_name == 'mnist-anomaly':
        x, y, x_t, y_t, y_names = mnist.load_data()
        dataset = SimulatedAnomalyDetectionDataset(dataset_name.replace('-', ''),
                                                   x, y, anomaly_class=0, test_set=(x_t, y_t))
    elif dataset_name == 'svhn-anomaly':
        x, y, x_t, y_t, y_names = svhn.load_data()
        dataset = SimulatedAnomalyDetectionDataset(dataset_name.replace('-', ''),
                                                   x, y, anomaly_class=0, test_set=(x_t, y_t))
    elif dataset_name == 'cifar10-anomaly':
        x, y, x_t, y_t, y_names = cifar10.load_data()
        dataset = SimulatedAnomalyDetectionDataset(dataset_name.replace('-', ''),
                                                   x, y, anomaly_class=0, test_set=(x_t, y_t))
    else:
        raise KeyError("Dataset not implemented")

    return dataset


if __name__ == '__main__':
    datapath = lsun.load_data()
    dataset = BufferedDataset(datapath)
    for x_batch, y_batch, batch_index in dataset.generator(batchsize=32):
        print("{}: {}".format(batch_index, x_batch.shape))

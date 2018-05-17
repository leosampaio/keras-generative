import h5py
import numpy as np
import itertools

from . import svhn
from . import mnist
from . import moving_mnist


class Dataset(object):

    def __init__(self):

        self.images = None

    def __len__(self):
        return len(self.images)

    def _get_shape(self):
        return self.images.shape

    shape = property(_get_shape)


class ConditionalDataset(Dataset):

    def __init__(self):
        super(ConditionalDataset, self).__init__()
        self.attrs = None
        self.attr_names = None

class PairwiseDataset(object):

    def __init__(self, x_data, y_data):
        assert x_data.shape[1] == y_data.shape[1]
        assert x_data.shape[2] == y_data.shape[2]
        assert x_data.shape[3] == 1 or y_data.shape[3] == 1 or \
            x_data.shape[3] == y_data.shape[3]

        if x_data.shape[3] != y_data.shape[3]:
            d = max(x_data.shape[3], y_data.shape[3])
            if x_data.shape[3] != d:
                x_data = np.tile(x_data, [1, 1, 1, d])
            if y_data.shape[3] != d:
                y_Data = np.tile(y_data, [1, 1, 1, d])

        x_len = len(x_data)
        y_len = len(y_data)
        l = min(x_len, y_len)

        self.x_data = x_data[:l]
        self.y_data = y_data[:l]

    def __len__(self):
        return len(self.x_data)

    def _get_shape(self):
        return self.x_data.shape

    shape = property(_get_shape)

class CrossDomainDatasets(object):

    def __init__(self, anchor_dataset, mirror_dataset):
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

        if next(self.counter) > 2*len(self.mirror):
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
            if remaining_idx: idx.append(remaining_idx)
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
    def __init__(self, x_data, input_n_frames=4):
        self.data = x_data
        self.input_n_frames = input_n_frames

    def __len__(self):
        return len(self.data)

    def _get_shape(self):
        return (self.data.shape[0],) + self.data.shape[2:4] + (self.data.shape[4]*self.input_n_frames,)

    def get_pairs(self, idx):
        selected_videos = self.data[idx, ...]
        num_frames = [len(x) for x in selected_videos]
        max_starting_frames_allowed = [n-(self.input_n_frames*2) for n in num_frames]
        starting_frames = [np.random.randint(0, m) for m in max_starting_frames_allowed]

        input_frames = [x[start:(start+self.input_n_frames)] for x, start in zip(selected_videos, starting_frames)]
        prediction_ground_truth = [x[(start+self.input_n_frames):(start+self.input_n_frames*2)] for x, start in zip(selected_videos, starting_frames)]

        input_frames = [self.concatenate_frames_over_channels(x) for x in input_frames]
        prediction_ground_truth = [self.concatenate_frames_over_channels(x) for x in prediction_ground_truth]
        return np.array(input_frames), np.array(prediction_ground_truth)

    def concatenate_frames_over_channels(self, x):
        t = np.transpose(x, (1, 2, 0, 3))
        return np.array(np.reshape(t, t.shape[0:2] + (t.shape[2]*t.shape[3],)))

    def undo_concatenated_frames_over_channel(self, x):
        t = np.reshape(x, x.shape[0:2]+(self.input_n_frames, x.shape[2]//self.input_n_frames))
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

def load_dataset(dataset_name):
    if dataset_name == 'mnist':
        dataset = ConditionalDataset()
        dataset.images, dataset.attrs, dataset.attr_names = mnist.load_data()
    elif dataset_name == 'mnist-original':
        dataset = ConditionalDataset()
        dataset.images, dataset.attrs, dataset.attr_names = mnist.load_data(original=True)
    elif dataset_name == 'mnist-rgb':
        dataset = ConditionalDataset()
        dataset.images, dataset.attrs, dataset.attr_names = mnist.load_data(use_rgb=True)
    elif dataset_name == 'svhn':
        dataset = ConditionalDataset()
        dataset.images, dataset.attrs, dataset.attr_names = svhn.load_data()
    elif dataset_name == 'mnist-svhn':
        anchor = load_dataset('mnist-rgb')
        mirror = load_dataset('svhn')
        dataset = CrossDomainDatasets(anchor, mirror)
    elif dataset_name == 'moving-mnist':
        data = moving_mnist.load_data()
        dataset = TimeCorelatedDataset(data)
    else:
        dataset = Dataset()
        dataset.images, dataset.attrs = load_general_dataset(dataset_name)

    return dataset


def display_random():
    dset = load_data('/home/alex/datasets/multi_330k_r_pr.hdf5')
    imgs = []
    import random
    for idx in [random.randint(0, 330000) for _ in range(5)]:
        imgs.append(dset.images[idx])

    imgs = np.asarray(imgs, 'float32')
    imgs = imgs * 0.5 + 0.5
    imgs = np.clip(imgs, 0.0, 1.0)
    import matplotlib.pyplot as plt
    for idx, img in enumerate(imgs):
        plt.imsave('{}.png'.format(idx), img)


def load_general_dataset(filepath):
    try:
        with h5py.File(filepath, 'r') as hf:
            labels = hf['labels']
            feats = hf['feats'][:]
    except Exception as e:
        hf = loadmat(filepath)
        labels = np.array(hf['labels']).flatten()
        feats = hf['feats']

    return feats, labels

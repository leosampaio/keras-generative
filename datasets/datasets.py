import h5py
import numpy as np
import itertools

from . import svhn
from . import mnist


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


    def __len__(self):
        return len(self.anchor)

    def _get_shape(self):
        return self.anchor.shape

    def _get_attr_names(self):
        return self.anchor.attr_names

    attr_names = property(_get_attr_names)
    shape = property(_get_shape)

def load_dataset(dataset_name):
    if dataset_name == 'mnist':
        dataset = ConditionalDataset()
        dataset.images, dataset.attrs, dataset.attr_names = mnist.load_data()
    if dataset_name == 'mnist-rgb':
        dataset = ConditionalDataset()
        dataset.images, dataset.attrs, dataset.attr_names = mnist.load_data(use_rgb=True)
    elif dataset_name == 'svhn':
        dataset = ConditionalDataset()
        dataset.images, dataset.attrs, dataset.attr_names = svhn.load_data()
    elif dataset_name == 'mnist-svhn':
        anchor = load_dataset('mnist-rgb')
        mirror = load_dataset('svhn')
        dataset = CrossDomainDatasets(anchor, mirror)
    else:
        dataset = ConditionalDataset()
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

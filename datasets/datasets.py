import h5py
import numpy as np

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


def load_dataset(dataset_name):
    if dataset_name == 'mnist':
        dataset = ConditionalDataset()
        dataset.images, dataset.attrs, dataset.attr_names = mnist.load_data()
    elif dataset_name == 'svhn':
        dataset = ConditionalDataset()
        dataset.images, dataset.attrs, dataset.attr_names = svhn.load_data()
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

import random

import h5py
import sys
import numpy as np

from tqdm import tqdm


def main(filename):
    f = h5py.File(filename)
    num_images = f['images'].shape[0]

    batch_size = 20 * 1000

    with h5py.File(filename.replace('.hdf5', '_pr.hdf5')) as hdf5_file:
        dset = hdf5_file.create_dataset('images', dtype='float32', shape=(0, 64, 64, 3),
                                        maxshape=(num_images, 64, 64, 3))

        image_data = []

        # shuffle the dataset
        idxs = [idx for idx in range(num_images)]
        random.shuffle(idxs)
        for idx in tqdm(idxs):
            image = f['images'][idx]
            image_data.append(image)

            if len(image_data) == batch_size:
                dset.resize(dset.shape[0] + batch_size, axis=0)
                image_data = np.asarray(image_data, 'float32') / 255.
                image_data = image_data * 2.0 - 1.0
                dset[-batch_size:] = image_data
                image_data = []

        if len(image_data) != 0:
            # append the last, not full batch
            rows_in_batch = len(image_data)
            dset.resize(dset.shape[0] + rows_in_batch, axis=0)
            image_data = np.asarray(image_data, 'float32') / 255.
            image_data = image_data * 2.0 - 1.0
            dset[-rows_in_batch:] = image_data

        print(dset.shape)


if __name__ == '__main__':
    main(sys.argv[1])

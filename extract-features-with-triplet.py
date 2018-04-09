import argparse
import os
import random
import numpy as np
from scipy import misc
from scipy.misc import imresize
import h5py


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

from keras import backend as K

from models import models
from datasets import load_dataset

from models.notifyier import *


def main():
    parser = argparse.ArgumentParser(description='Sanity Check: x -> z -> x\'')
    parser.add_argument('--model', type=str, default='ALI', required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--zdims', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--domain-separated', action='store_true')
    parser.add_argument('--submodels', nargs=2,
                        help="Submodels used to build the bigger one",
                        required=True)
    parser.add_argument('--datasets', nargs=2, required=True)
    args = parser.parse_args()

    # select gpu and limit resources if applicable
    if 'tensorflow' == K.backend():
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(args.gpu)
        set_session(tf.Session(config=config))

    dataset_a = load_dataset(args.datasets[0])
    dataset_b = load_dataset(args.datasets[1])

    random.seed(14)
    perm_a = np.random.permutation(len(dataset_a))
    perm_b = np.random.permutation(len(dataset_b))
    a_x, a_y = dataset_a.images[perm_a], np.argmax(dataset_a.attrs[perm_a], axis=1)
    b_x, b_y = dataset_b.images[perm_b], np.argmax(dataset_b.attrs[perm_b], axis=1)

    model = models[args.model](
        input_shape=dataset_a.shape[1:],
        z_dims=args.zdims,
        output='',
        submodels=args.submodels
    )

    model.load_model(args.weights)

    a_feat = model.alice_d1.f_Gz.predict(a_x)
    b_feat = model.alice_d2.f_Gz.predict(b_x)

    if args.domain_separated:
        a_feat = a_feat[..., :args.zdims//2]
        b_feat = b_feat[..., :args.zdims//2]

    output_a = "{}_a.h5".format(args.output)
    output_b = "{}_b.h5".format(args.output)
    with h5py.File(output_a, 'w') as hf:
        hf.create_dataset("labels",  data=a_y)
        hf.create_dataset("feats",  data=a_feat)
    with h5py.File(output_b, 'w') as hf:
        hf.create_dataset("labels",  data=b_y)
        hf.create_dataset("feats",  data=b_feat)

if __name__ == '__main__':
    main()

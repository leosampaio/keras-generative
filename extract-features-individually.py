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
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    # select gpu and limit resources if applicable
    if 'tensorflow' == K.backend():
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(args.gpu)
        set_session(tf.Session(config=config))

    dataset_a = load_dataset(args.dataset)

    random.seed(14)
    perm_a = np.random.permutation(len(dataset_a))
    a_x, a_y = dataset_a.images[perm_a], np.argmax(dataset_a.attrs[perm_a], axis=1)

    model = models[args.model](
        input_shape=dataset_a.shape[1:],
        z_dims=args.zdims,
        output=''
    )

    model.load_model(args.weights)

    a_feat = model.f_Gz.predict(a_x)

    if args.domain_separated:
        a_feat = a_feat[..., :args.zdims//2]

    output_a = "{}.h5".format(args.output)
    with h5py.File(output_a, 'w') as hf:
        hf.create_dataset("labels",  data=a_y)
        hf.create_dataset("feats",  data=a_feat)

if __name__ == '__main__':
    main()

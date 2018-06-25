import os
import sys
import math
import argparse

from keras import backend as K
import numpy as np
from sklearn.preprocessing import LabelBinarizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use('Agg')

from models import models
from datasets import load_dataset


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Training GANs or VAEs')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batchsize', type=int, default=50)
    parser.add_argument('--output', default='output')
    parser.add_argument('--z-dims', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--test-mode', action='store_true')
    parser.add_argument('--is-conditional', action='store_true')
    parser.add_argument('--aux-classifier', action='store_true')
    parser.add_argument('--label-smoothing', default=0.0, type=float)
    parser.add_argument('--input-noise', default=0.0, type=float)
    parser.add_argument('--run-id', '-r', required=True)
    parser.add_argument('--checkpoint-every', default=1, type=int)
    parser.add_argument('--notify-every', default=1, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--dis-loss-control', default=1., type=float)
    parser.add_argument('--triplet-weight', default=1., type=float)
    parser.add_argument('--embedding-dim', default=256, type=int)
    parser.add_argument('--isolate-d-classifier', action='store_true')
    parser.add_argument('--loss-weights', type=float, nargs='+',
                        help="weights for each loss function")
    parser.add_argument('--loss-control', type=str, nargs='+',
                        help="one of 'inc', 'dec', 'hold', 'halt' or "
                        "'none' for each loss function")
    parser.add_argument('--loss-control-epoch', type=int, nargs='+',
                        help="epoch selected to work with loss-control method, "
                        "one for each loss function")
    parser.add_argument('--metrics', type=str, nargs='+',
                        help="selection of metrics you want to calculate")
    parser.add_argument('--metrics-every', type=int,
                        help="metrics frequency (by epoch)")

    args = parser.parse_args()

    # select gpu and limit resources if applicable
    if 'tensorflow' == K.backend():
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(args.gpu)
        set_session(tf.Session(config=config))

    # make output directory if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # load datasets
    dataset = load_dataset(args.dataset)

    # Construct model
    if args.model not in models:
        raise Exception('Unknown model:', args.model)

    model = models[args.model](
        input_shape=dataset.shape[1:],
        **vars(args)
    )

    if args.resume:
        model.load_model(args.resume)

    model.main_loop(dataset, epochs=args.epoch, batchsize=args.batchsize)


if __name__ == '__main__':
    main()

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
    parser.add_argument('--zdims', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--testmode', action='store_true')
    parser.add_argument('--conditional', action='store_true')
    parser.add_argument('--aux-classifier', action='store_true')
    parser.add_argument('--label-smoothing', default=0.0, type=float)
    parser.add_argument('--input-noise', default=0.0, type=float)
    parser.add_argument('--run-id', '-r', default=1, type=int)
    parser.add_argument('--checkpoint-every', default=1, type=int)
    parser.add_argument('--notify-every', default=1, type=int)
    parser.add_argument('--triplet-margin', default=1., type=float)
    parser.add_argument('--triplet-weight', default=1., type=float)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--n-layers-to-share', default=0, type=int)
    parser.add_argument('--submodels', nargs=2,
                        help="Submodels used to build the bigger one",
                        required=True)
    parser.add_argument('--resume-submodels', nargs=2,
                        help="Submodels pretrained weights")
    parser.add_argument('--dis-loss-control', default=1., type=float)

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
        z_dims=args.zdims,
        output=args.output,
        label_smoothing=args.label_smoothing,
        input_noise=args.input_noise,
        run_id=args.run_id,
        test_mode=args.testmode,
        checkpoint_every=args.checkpoint_every,
        notify_every=args.notify_every,
        aux_classifier=args.aux_classifier,
        is_conditional=args.conditional,
        conditional_dims=len(dataset.attr_names),
        triplet_margin=args.triplet_margin,
        triplet_weight=args.triplet_weight,
        lr=args.lr,
        submodels=args.submodels,
        dis_loss_control=args.dis_loss_control,
        submodels_weights=args.resume_submodels,
        permutation_matrix_shape=(len(dataset), dataset.mirror_len)
    )

    if args.resume or args.resume_submodels:
        model.load_model(args.resume)

    # generate random samples to evaluate generated results over time
    # use the same samples for all trainings - useful when resuming training
    np.random.seed(14)
    samples = np.random.normal(size=(100, args.zdims)).astype(np.float32)
    conditionals_for_samples = np.array(
        [LabelBinarizer().fit_transform(
            range(0, len(dataset.attr_names)))
         [i % len(dataset.attr_names)] for i in range(100)])
    np.random.seed()

    model.main_loop(dataset, samples,
                    samples_conditionals=conditionals_for_samples,
                    epochs=args.epoch,
                    batchsize=args.batchsize)


if __name__ == '__main__':
    main()

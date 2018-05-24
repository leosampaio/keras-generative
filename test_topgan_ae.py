import argparse
import os
import random
import numpy as np
from scipy import misc
from scipy.misc import imresize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

from keras import backend as K

from models import models
from datasets import load_dataset

from models.notifyier import *

def experiment_mapping(model, data, direction, signature):

    a_x, p_x, n_x = data    

    if direction == '1-2':
        originals = a_x
        encodings = model.alice_d1.f_Gz.predict(a_x)
        reconstructions = model.alice_d2.f_Gx.predict(encodings)
    elif direction == '2-1':
        originals = p_x
        encodings = model.alice_d2.f_Gz.predict(p_x)
        reconstructions = model.alice_d1.f_Gx.predict(encodings)
    elif direction == '1-1':
        originals = a_x
        encodings = model.alice_d1.f_Gz.predict(a_x)
        reconstructions = model.alice_d1.f_Gx.predict(encodings)
    elif direction == '2-2':
        originals = p_x
        encodings = model.alice_d2.f_Gz.predict(p_x)
        reconstructions = model.alice_d2.f_Gx.predict(encodings)
    elif direction == 'all':
        experiment_mapping(model, data, '1-2', "{}_12".format(signature))
        experiment_mapping(model, data, '2-1', "{}_21".format(signature))
        experiment_mapping(model, data, '1-1', "{}_11".format(signature))
        experiment_mapping(model, data, '2-2', "{}_22".format(signature))
        return

    weaved_imgs = np.empty((2*len(originals), *originals.shape[1:]), dtype=originals.dtype)
    weaved_imgs[0::2] = originals
    weaved_imgs[1::2] = reconstructions

    model.save_image_as_plot(weaved_imgs, "output/test1.jpg")
    notify_with_message("[{}] Triplet ALI Cycle Test".format(signature))
    notify_with_image("output/test1.jpg")

def experiment_distances(model, data, domain_separated, zdims, signature):

    a_x, p_x, n_x = data

    encodings_a = model.alice_d1.f_Gz.predict(a_x)
    encodings_p = model.alice_d2.f_Gz.predict(p_x)
    encodings_n = model.alice_d2.f_Gz.predict(n_x)

    if domain_separated:
        encodings_a = encodings_a[..., :zdims//2]
        encodings_p = encodings_p[..., :zdims//2]
        encodings_n = encodings_n[..., :zdims//2]

    mean_p = np.mean(np.linalg.norm(encodings_a-encodings_p, axis=1))
    mean_n = np.mean(np.linalg.norm(encodings_a-encodings_n, axis=1))

    print("Mean Positive Pair Distance: {}".format(mean_p))
    print("Mean Negative Pair Distance: {}".format(mean_n))

def main():
    parser = argparse.ArgumentParser(description='Sanity Check: x -> z -> x\'')
    parser.add_argument('--model', type=str, default='ALI', required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--zdims', type=int, default=256)
    parser.add_argument('--dataset', default=256, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--samples', type=int, default=50)
    parser.add_argument('--test-signature', default="Test")
    parser.add_argument('--direction', choices=['1-2', '2-1', '1-1', '2-2', 'all'], default='1-2', const='1-2', nargs='?')
    parser.add_argument('--experiment', choices=['mapping', 'distances'], required=True)
    parser.add_argument('--triplet-margin', default=1., type=float)
    parser.add_argument('--triplet-weight', default=1., type=float)
    parser.add_argument('--domain-separated', action='store_true')
    parser.add_argument('--submodels', nargs=2,
                        help="Submodels used to build the bigger one",
                        required=True)
    args = parser.parse_args()

    # select gpu and limit resources if applicable
    if 'tensorflow' == K.backend():
        import tensorflow as tf
        from keras.backend.tensorflow_backend import set_session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(args.gpu)
        set_session(tf.Session(config=config))

    dataset = load_dataset(args.dataset)
    random.seed(14)
    perm = np.random.permutation(len(dataset))
    data, labels = dataset.get_triplets(perm[:args.samples])

    model = models[args.model](
        input_shape=dataset.shape[1:],
        z_dims=args.zdims,
        output='',
        triplet_margin=args.triplet_margin,
        triplet_weight=args.triplet_weight,
        submodels=args.submodels
    )

    model.load_model(args.weights)

    if args.experiment == 'mapping' or args.direction == 'all': experiment_mapping(model, data, args.direction, args.test_signature)
    elif args.experiment == 'distances' or args.direction == 'all': experiment_distances(model, data, args.domain_separated, args.zdims, args.test_signature)

if __name__ == '__main__':
    main()

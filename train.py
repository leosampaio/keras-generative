import os
import sys
import math
import argparse

from models.hacked import VeryDcgan, DropoutDcgan, DropoutVae, DropoutALI, DropoutImprovedGAN, VeryDeepImprovedGAN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib

matplotlib.use('Agg')

from models import VAE, DCGAN, ImprovedGAN, EBGAN, BEGAN, ALI, AAE, BinAAE, AAE2, DrAAE2, PaperALI, WiderALI, DeeperALI, \
    LocallyConnALI, MobileNetALI
from datasets import load_data, mnist
from datasets.datasets import load_data

models = {
    'vae': VAE,
    'dcgan': DCGAN,
    'improvedgan': ImprovedGAN,
    'ebgan': EBGAN,
    'began': BEGAN,
    'ali': ALI,
    'paper_ali': PaperALI,
    'wider_ali': WiderALI,
    'deeper_ali': DeeperALI,
    'local_conn_ali': LocallyConnALI,
    'mobile_ali': MobileNetALI,
    'vdcgan': VeryDcgan,
    'drdcgan': DropoutDcgan,
    'drvae': DropoutVae,
    'drimprovedgan': DropoutImprovedGAN,
    'drali': DropoutALI,
    'aae': AAE,
    'binaae': BinAAE,
    'aae2': AAE2,
    'draae2': DrAAE2,
    'vdimprovedgan': VeryDeepImprovedGAN,
}


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
    parser.add_argument('--label_smoothing', default=0.0, type=float)
    parser.add_argument('--input_noise', default=0.0, type=float)

    args = parser.parse_args()

    # select gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Make output direcotiry if not exists
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    # Load datasets
    if args.dataset == 'mnist':
        dataset = mnist.load_data()
    elif args.dataset == 'svhn':
        dataset = svhn.load_data()
    else:
        dataset = load_data(args.dataset)

    # Construct model
    if args.model not in models:
        raise Exception('Unknown model:', args.model)

    model = models[args.model](
        input_shape=dataset.shape[1:],
        z_dims=args.zdims,
        output=args.output,
        label_smoothing=args.label_smoothing,
        input_noise=args.input_noise,
    )

    if args.testmode:
        model.test_mode = True

    if args.resume is not None:
        model.load_model(args.resume)

    dataset = dataset.images

    # Use the same samples for all trainings - useful when resuming training
    np.random.seed(123)
    samples = np.random.normal(size=(100, args.zdims)).astype(np.float32)
    np.random.seed()
    # Training loop
    model.main_loop(dataset, samples,
                    epochs=args.epoch,
                    batchsize=args.batchsize,
                    reporter=['loss', 'g_loss', 'd_loss', 'g_acc', 'd_acc'])


if __name__ == '__main__':
    main()

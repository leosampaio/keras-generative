import os
import sys
import math
import argparse

from models.hacked import VeryDcgan, DropoutDcgan, DropoutVae, DropoutALI, DropoutImprovedGAN, VeryDeepImprovedGAN
from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
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
    parser.add_argument('--label_smoothing', default=0.0, type=float)
    parser.add_argument('--input_noise', default=0.0, type=float)
    parser.add_argument('--swap_prob', default=0.1, type=float)
    parser.add_argument('--run_id', '-r', default=1, type=int)

    args = parser.parse_args()

    # select gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

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
        swap_prob=args.swap_prob,
        run_id=args.run_id
    )

    if args.testmode:
        model.test_mode = True

    if args.resume is not None:
        model.load_model(args.resume)

    # Use the same samples for all trainings - useful when resuming training
    np.random.seed(14)
    samples = np.random.normal(size=(100, args.zdims)).astype(np.float32)
    np.random.seed()
    model.main_loop(dataset, samples,
                    epochs=args.epoch,
                    batchsize=args.batchsize,
                    reporter=['loss', 'g_loss', 'd_loss', 'g_acc', 'd_acc'])


if __name__ == '__main__':
    main()

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

from models import models
from datasets import load_dataset
from models.notifyier import *

def main():
    parser = argparse.ArgumentParser(description='Sanity Check: x -> z -> x\'')
    parser.add_argument('--model', type=str, default='ALI', required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--zdims', type=int, default=256)
    parser.add_argument('--dataset', default=256, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    dataset = load_dataset(args.dataset)
    random.seed(14)
    random.shuffle(dataset.images)
    images = dataset.images[:50]

    model = models[args.model](
        input_shape=dataset.shape[1:],
        z_dims=args.zdims,
        output='',
    )

    model.load_model(args.weights)
    encodings = model.f_Gz.predict(images)
    reconstructions = model.predict_images(encodings)

    weaved_imgs = np.empty((2*len(images), *images.shape[1:]), dtype=images.dtype)
    weaved_imgs[0::2] = images
    weaved_imgs[1::2] = reconstructions

    model.save_image_as_plot(weaved_imgs, "output/tmp.jpg")
    epoch = int(args.weights.split('_')[-1].replace('/', ''))
    signature = "{}/epoch{}".format(args.model, epoch)
    notify_with_message("[{}] ALI Cycle Test".format(signature))
    notify_with_image("output/tmp.jpg")

if __name__ == '__main__':
    main()
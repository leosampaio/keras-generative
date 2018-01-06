import argparse
import os
import numpy as np
from tqdm import tqdm
from collections import Counter

import models
from analysis.interpolations import interpolations_walk, plot_images

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='interpolations')
    parser.add_argument('--model', type=str, default='ALI')
    parser.add_argument('--weights', type=str,
                        default='/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc/ali/weights/epoch_00005')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--z_dims', type=int, default=256)
    parser.add_argument('--method', type=str, default='slerp',
                        help='Interpolation method. slerp - great circle, other - linear')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    args = parser.parse_args()

    output_dir = '{}_{}'.format(args.output, args.model)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    model = getattr(models, args.model)(
        input_shape=(64, 64, 3),
        z_dims=args.z_dims,
        output=''
    )
    model.load_model(args.weights)

    # walk along a dimension
    interpolations = interpolations_walk(args.z_dims, num_steps=args.num_steps, method=args.method)
    images = model.predict_images(interpolations)
    # find the ratios of absolute change in pixel values to change in latent space values
    ratios = np.zeros((args.z_dims, args.num_steps - 1,))
    for dim_id in tqdm(range(args.z_dims)):
        # compute the differences in values in the latent space for each consecutive step
        z_vals = interpolations[dim_id * args.num_steps:(dim_id + 1) * args.num_steps, dim_id]
        z_diffs = np.ediff1d(z_vals)

        # compute the differences in values in pixel space for each consecutive step
        p_vals = images[dim_id * args.num_steps:(dim_id + 1) * args.num_steps, :, :, :]
        p_diffs = np.abs(np.diff(p_vals, axis=0))
        p_diffs = np.sum(p_diffs, axis=(1, 2, 3,))

        # compute the ratio of pixel space to latent space values difference for each step
        ratio_vector = np.divide(p_diffs, z_diffs)
        ratios[dim_id, :] = ratio_vector

    plt.hist(ratios, bins='auto')
    plt.savefig(os.path.join(output_dir, 'ratio_hist.png'))

    outliers_ids = np.argwhere(np.abs(ratios - ratios.mean()) > 2 * ratios.std())
    dim_id_counts = Counter([idx[0] for idx in outliers_ids])
    count_border_val = args.num_steps // 2 + 1
    # dimensions where a big change in pixel values occured in less than half steps
    sudden_change_ids = [dim_id for dim_id, dim_count in dim_id_counts.items() if dim_count < count_border_val]
    # dimensions where a big change in pixel values occured in more than half steps
    smooth_change_ids = [dim_id for dim_id, dim_count in dim_id_counts.items() if dim_count >= count_border_val]

    num_samples_to_plot = 10
    sudden_images = np.zeros(
        (len(sudden_change_ids) + len(sudden_change_ids) % num_samples_to_plot, args.num_steps, 64, 64, 3))
    smooth_images = np.zeros(
        (len(smooth_change_ids) + len(smooth_change_ids) % num_samples_to_plot, args.num_steps, 64, 64, 3))

    for idx, dim_id in enumerate(sudden_change_ids):
        sudden_images[idx] = images[dim_id * args.num_steps:(dim_id + 1) * args.num_steps]
    for idx, dim_id in enumerate(smooth_change_ids):
        smooth_images[idx] = images[dim_id * args.num_steps:(dim_id + 1) * args.num_steps]

    for plot_id in range(sudden_images.shape[0] // num_samples_to_plot):
        filename = os.path.join(output_dir, 'sudden_{}.png'.format(plot_id))
        images = sudden_images[plot_id * args.num_steps:(plot_id + 1) * args.num_steps].reshape(
            (args.num_steps * num_samples_to_plot), 64, 64, 3)
        plot_images(images=images, filename=filename, num_samples=num_samples_to_plot, num_steps=args.num_steps)
    for plot_id in range(smooth_images.shape[0] // num_samples_to_plot):
        filename = os.path.join(output_dir, 'smooth{}.png'.format(plot_id))
        images = smooth_images[plot_id * args.num_steps:(plot_id + 1) * args.num_steps].reshape(
            (args.num_steps * num_samples_to_plot), 64, 64, 3)
        plot_images(images=images, filename=filename, num_samples=num_samples_to_plot, num_steps=args.num_steps)


if __name__ == '__main__':
    main()

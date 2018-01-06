import argparse
import os

from tqdm import tqdm

import models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def interpolate_vectors(a, b, steps=10):
    interp = np.zeros((steps,) + a.shape)
    interp[0] = a
    for dim_id in range(a.shape[0]):
        for step in range(1, steps - 1):
            interp[step][dim_id] = a[dim_id] + (b[dim_id] - a[dim_id]) / (steps - 1) * step
    interp[steps - 1] = b
    return interp


def slerp(val, low, high):
    """
    :param val: [0., 1.] - approx. fraction of distance between points to calculate
    :param low: first point
    :param high: second point
    :return:
    """
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def interpolate_vectors_slerp(a, b, steps=10):
    interp = np.zeros((steps,) + a.shape)
    interp[0] = a
    for step in range(1, steps - 1):
        val = step / (steps - 1)
        interp[step] = slerp(val=val, low=a, high=b)
    interp[steps - 1] = b
    return interp


def plot_images(images, filename, num_samples=10, num_steps=10):
    """
    :param images: numpy array of shape (None, None, None ,3)
    :param filename:
    :return:
    """
    fig = plt.figure(figsize=(8, 8))
    grid = gridspec.GridSpec(num_samples, num_steps, wspace=0.1, hspace=0.1)
    for i in range(num_samples * num_steps):
        ax = plt.Subplot(fig, grid[i])
        ax.imshow(images[i, :, :, :], interpolation='none', vmin=0.0, vmax=1.0)
        ax.axis('off')
        fig.add_subplot(ax)
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def interpolations_point2point(z_dims, num_samples=10, num_steps=10, method='slerp'):
    points_a = np.random.normal(size=(num_samples, z_dims,)).astype(np.float32)
    points_b = np.random.normal(size=(num_samples, z_dims,)).astype(np.float32)
    if method == 'slerp':
        # great circle interpolation
        interpolations = np.zeros((num_samples, num_steps, z_dims))
        for sample_id in range(num_samples):
            interpolations[sample_id] = interpolate_vectors_slerp(points_a[sample_id], points_b[sample_id],
                                                                  steps=num_steps)
    else:
        # linear
        interpolations = interpolate_vectors(points_a, points_b, steps=num_steps)

    interpolations = interpolations.reshape((num_samples * num_steps, z_dims))
    if method != 'slerp':
        interpolations = np.rot90(interpolations.reshape((num_samples, num_steps, z_dims))).reshape(
            (num_samples * num_steps, z_dims))
    return interpolations


def interpolations_walk(z_dims, num_steps=10, method='slerp'):
    points = np.random.normal(size=(z_dims,)).astype(np.float32)
    interpolations = np.zeros((z_dims, num_steps, z_dims))
    for dim_id in range(z_dims):
        points_a = points.copy()
        points_b = points.copy()

        points_a[dim_id] = -1
        points_b[dim_id] = 1

        if method == 'slerp':
            # great circle interpolation
            interpolations[dim_id, :, :] = interpolate_vectors_slerp(points_a, points_b, num_steps)
        else:
            # linear
            interpolations[dim_id, :, :] = interpolate_vectors(points_a, points_b, num_steps)

    interpolations = interpolations.reshape((num_steps * z_dims, z_dims))
    return interpolations


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

    # point to point interpolations
    interpolations = interpolations_point2point(args.z_dims, args.num_samples, args.num_steps, method=args.method)
    images = model.predict_images(interpolations)
    plot_images(images, os.path.join(output_dir, 'p2p.png'), args.num_samples, args.num_steps)

    # walk along a dimension
    interpolations = interpolations_walk(args.z_dims, num_steps=args.num_steps, method=args.method)
    images = model.predict_images(interpolations)
    step = 10
    for dim_id in tqdm(range(0, args.z_dims - step, step)):
        # plot interpolations for 10 next dimensions
        plot_images(images[dim_id * args.num_steps:(dim_id + step) * args.num_steps, :, :, :],
                    os.path.join(output_dir, 'walk_{}.png'.format(dim_id)), step, args.num_steps)
    # plot the last number of dimensions (as 10 - the number of steps - isn't a divider of 128, 256 etc.)
    plot_images(images[-step * args.num_steps:, :, :, :],
                os.path.join(output_dir, 'walk_{}.png'.format(dim_id + step)), step, args.num_steps)


if __name__ == '__main__':
    main()

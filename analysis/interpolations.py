import argparse

from models import ALI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


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


def interpolate_vectors(a, b, steps=10):
    interp = np.zeros((steps,) + a.shape)
    interp[0] = a
    for i in range(a.shape[0]):
        for step in range(1, steps - 1):
            interp[step][i] = a[i] + (b[i] - a[i]) / (steps - 1) * step
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


def random_interpolations(z_dims, num_samples=10, num_steps=10):
    point_a = np.random.normal(size=(num_samples, z_dims,)).astype(np.float32)
    point_b = np.random.normal(size=(num_samples, z_dims,)).astype(np.float32)
    interpolations = interpolate_vectors(point_a, point_b, steps=num_steps)
    interpolations = interpolations.reshape((num_samples * num_steps, z_dims))
    interpolations = np.rot90(interpolations.reshape((num_samples, num_steps, z_dims))).reshape(
        (num_samples * num_steps, z_dims))
    return interpolations


def main():
    parser = argparse.ArgumentParser(description='interpolations')
    parser.add_argument('--weights', type=str,
                        default='/home/alex/Desktop/proj/sign-lang/keras-generative/out_bbc/ali/weights/epoch_00005')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--z_dims', type=int, default=256)

    args = parser.parse_args()

    model = ALI(
        input_shape=(64, 64, 3),
        z_dims=args.z_dims,
        output=''
    )
    model.load_model(args.weights)

    interpolations = random_interpolations(args.z_dims, args.num_samples, args.num_steps)
    images = model.predict_images(interpolations)
    plot_images(images, 'test.png', args.num_samples, args.num_steps)


if __name__ == '__main__':
    main()

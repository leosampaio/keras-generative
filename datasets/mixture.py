import numpy as np
import itertools
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import keras
from sklearn.preprocessing import minmax_scale


def load_data(type='ring', n=10, d=2, r=10, std=1, density=50):
    if type == 'ring':
        theta = np.linspace(0, 2 * np.pi, n + 1)
        a, b = r * np.cos(theta), r * np.sin(theta)
        means = np.column_stack((np.expand_dims(a[:-1], -1), np.expand_dims(b[:-1], -1)))
        sigma = np.ones((d,)) * std
        x = np.concatenate([np.random.normal(loc=mean, scale=sigma, size=(density, d)) for mean in means], axis=0)
        y = np.concatenate(np.array([[(means[i], sigma) for i in range(0, n)] for _ in range(density)]), axis=0)
    elif type == 'grid':
        means = [np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                                range(-4, 5, 2))]
        sigma = np.ones((d,)) * std
        x = np.concatenate([np.random.normal(loc=mean, scale=sigma, size=(density, d)) for mean in means], axis=0)
        y = np.concatenate(np.array([[(means[i], sigma) for i in range(0, n)] for _ in range(density)]), axis=0)
    elif type == 'high-dim':
        shape = (d,)
        means = [-1 * np.ones(shape, dtype="float32"), -.5 * np.ones(shape, dtype="float32"),
                 0 * np.ones(shape, dtype="float32"), .5 * np.ones(shape, dtype="float32"),
                 -2 * np.ones(shape, dtype="float32"), -2.5 * np.ones(shape, dtype="float32"),
                 10 * np.ones(shape, dtype="float32"), .25 * np.ones(shape, dtype="float32"),
                 -13 * np.ones(shape, dtype="float32"), -5.5 * np.ones(shape, dtype="float32")]
        sigma = np.concatenate([std * np.ones((d - 500,), dtype="float32"),
                                np.zeros((500), dtype="float32")], axis=0)
        x = np.concatenate([np.random.normal(loc=mean, scale=sigma, size=(density, d)) for mean in means], axis=0)
        y = np.concatenate(np.array([[(means[i], sigma) for i in range(0, n)] for _ in range(density)]), axis=0)

    # y = keras.utils.to_categorical(y)
    # x = minmax_scale(x, axis=0)
    # means = minmax_scale(means, axis=0)
    mixture_identifiers = np.array([(means[i], sigma) for i in range(0, n)])
    return x, y, mixture_identifiers

if __name__ == "__main__":
    x, y = load_data(type="grid", n=25, std=.05, r=1)
    cmap = plt.cm.gnuplot
    norm = matplotlib.colors.Normalize(vmin=np.min(y), vmax=np.max(y))
    cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    mapped_colors = cmapper.to_rgba(y)
    unique_labels = list(set(y))
    plt.figure(figsize=(7, 6))
    plt.scatter(x[:, 0], x[:, 1], color=mapped_colors)
    plt.show(block=True)

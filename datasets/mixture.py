import numpy as np
import itertools
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import keras


def load_data(type='ring', n=10, d=2, r=10, std=1, density=50):
    if type == 'ring':
        theta = np.linspace(0, 2 * np.pi, n + 1)
        a, b = r * np.cos(theta), r * np.sin(theta)
        means = np.column_stack((np.expand_dims(a[:-1], -1), np.expand_dims(b[:-1], -1)))
        x = np.concatenate([np.random.normal(loc=mean, scale=std, size=(density, d)) for mean in means], axis=0)
        y = np.repeat(list(range(0, n)), density)
    elif type == 'grid':
        means = [np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                                range(-4, 5, 2))]
        x = np.concatenate([np.random.normal(loc=mean, scale=std, size=(density, d)) for mean in means], axis=0)
        y = np.repeat(list(range(0, n)), density)
    y = keras.utils.to_categorical(y)
    return x, y, [str(i) for i in range(n)]

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

import numpy as np
import keras

def load_data(use_rgb=False):
    (x_train, y_train), _ = keras.datasets.mnist.load_data()

    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
    if use_rgb:
        x_train = np.stack((x_train,)*3, -1)
    else:
        x_train = (x_train[:, :, :, np.newaxis])
    x_train = x_train.astype('float32')/255.
    y_train = keras.utils.to_categorical(y_train)
    y_train = y_train.astype('float32')

    return x_train, y_train, [str(i) for i in range(10)]

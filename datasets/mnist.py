import numpy as np
import keras

def load_data():
    (x_train, y_train), _ = keras.datasets.mnist.load_data()

    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'constant', constant_values=0)
    x_train = (x_train[:, :, :, np.newaxis] / 255.0).astype('float32')
    y_train = keras.utils.to_categorical(y_train)
    y_train = y_train.astype('float32')

    return x_train, y_train, [str(i) for i in range(10)]

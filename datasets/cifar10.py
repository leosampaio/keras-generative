import keras
import numpy as np


def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = keras.utils.to_categorical(y_train)
    y_train = y_train.astype('float32')
    y_test = np.squeeze(y_test)

    return x_train, y_train, x_test, y_test, [str(i) for i in range(10)]

import argparse
import os
import h5py
from keras import Model, Input
from keras.callbacks import Callback
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Reshape

from models.layers import BasicConvLayer, BasicDeconvLayer
import matplotlib.pyplot as plt
import numpy as np


def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
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
    for i in range(a.shape[0]):
        for step in range(steps - 1):
            interp[step][i] = a[i] + (b[i] - a[i]) / steps * step
    interp[steps - 1] = b
    return interp


class PlotReconstructions(Callback):
    def __init__(self, encoder, decoder, z_dim=256, examples_to_plot=10):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.examples_to_plot = examples_to_plot
        self.z_dim = z_dim

        if not os.path.isdir('aae'):
            os.makedirs('aae')

    def on_epoch_end(self, epoch, logs=None):
        data = self.validation_data[0][:self.examples_to_plot]
        decoded_imgs = self.model.predict(data)
        random_z = np.random.normal(size=(self.examples_to_plot, self.z_dim))
        random_imgs = self.decoder.predict(random_z)

        # interpolations
        a = data[0]
        b = data[1]
        encodings = self.encoder.predict(np.array([a, b]).reshape((2,) + a.shape))
        interp = interpolate_vectors(encodings[0], encodings[1])
        decoded_interp = self.decoder.predict(interp)

        plot_reconstructions(data, decoded_imgs, random_imgs, decoded_interp, 'aae/aae_{}.png'.format(epoch))


def plot_reconstructions(x, y, random_imgs, interpolations, plotname):
    n = x.shape[0]
    img_size = x.shape[1]
    plt.figure(figsize=(15, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(4, n, i + 1)
        plt.imshow(x[i].reshape((img_size, img_size, 3)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(4, n, i + 1 + n)
        plt.imshow(y[i].reshape((img_size, img_size, 3)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display random images
        ax = plt.subplot(4, n, i + 1 + 2 * n)
        plt.imshow(random_imgs[i].reshape((img_size, img_size, 3)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # interpolation
        ax = plt.subplot(4, n, i + 1 + 3 * n)
        plt.imshow(interpolations[i].reshape((img_size, img_size, 3)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig(plotname)
    plt.close()


def load_data(train_cnt=10000, val_cnt=500):
    ds_path = '/home/alex/datasets/bbc_full_r.hdf5'
    offset = 10000
    with h5py.File(ds_path) as h5ds:
        x_train = h5ds['images'][offset:train_cnt + offset].astype('float32') / 255.
        x_val = h5ds['images'][train_cnt + offset:train_cnt + val_cnt + offset].astype('float32') / 255.

        return x_train, x_val


def make_decoder(img_size=64, z_dim=256):
    inputs = Input(shape=(z_dim,))
    w = img_size // (2 ** 3)
    x = Dense(w * w * 256)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((w, w, 256))(x)

    x = BasicDeconvLayer(filters=256, strides=(2, 2))(x)
    x = BasicDeconvLayer(filters=128, strides=(2, 2))(x)
    x = BasicDeconvLayer(filters=img_size, strides=(2, 2))(x)
    x = BasicDeconvLayer(filters=3, strides=(1, 1), bnorm=False, activation='sigmoid')(x)  # or tanh

    return Model(inputs, x)


def make_model(img_size=64, z_dim=256):
    inputs = Input(shape=(img_size, img_size, 3))

    x = BasicConvLayer(filters=64, strides=(2, 2))(inputs)
    x = BasicConvLayer(filters=128, strides=(2, 2))(x)
    x = BasicConvLayer(filters=256, strides=(2, 2))(x)
    x = BasicConvLayer(filters=512, strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    encoded = Dense(z_dim)(x)

    decoder = make_decoder(img_size, z_dim)
    reconstruction = decoder(encoded)

    autoencoder = Model(inputs, reconstruction)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    encoder = Model(inputs, encoded)
    encoded_input = Input(shape=(z_dim,))
    decoder = Model(encoded_input, decoder(encoded_input))

    return autoencoder, encoder, decoder


def main():
    parser = argparse.ArgumentParser(description='Adversarial Autoencoder')
    parser.add_argument('-e', '--epochs', default=50)
    parser.add_argument('--batch_size', default=256)
    parser.add_argument('--data_size', default=10000)
    parser.add_argument('--val_data_size', default=500)
    args = parser.parse_args()

    autoencoder, encoder, decoder = make_model()
    print(autoencoder.summary())
    print('Loading data.')
    x_train, x_val = load_data(train_cnt=args.data_size, val_cnt=args.val_data_size)
    autoencoder.fit(x_train, x_train,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    shuffle=True,
                    validation_data=(x_val, x_val),
                    callbacks=[PlotReconstructions(encoder, decoder)]
                    )


if __name__ == '__main__':
    main()

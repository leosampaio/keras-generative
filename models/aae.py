import argparse
import h5py
from keras import Model, Input
from keras.layers import Flatten, Dense, BatchNormalization, Activation, Reshape

from models.layers import BasicConvLayer, BasicDeconvLayer
import numpy as np

from models.utils import PlotReconstructions


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

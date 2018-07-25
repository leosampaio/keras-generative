import os
import sys
import requests

import numpy as np
import scipy as sp
import scipy.io

import keras

url_train = 'http://ufldl.stanford.edu/housenumbers/train_32x32.mat'
url_extra = 'http://ufldl.stanford.edu/housenumbers/extra_32x32.mat'
url_test = 'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
curdir = os.path.abspath(os.path.dirname(__file__))
outdir = os.path.join(curdir, 'files')
outfile_train = os.path.join(outdir, 'svhn.mat')
outfile_extra = os.path.join(outdir, 'svhn_extra.mat')
outfile_test = os.path.join(outdir, 'svhn_test.mat')

CHUNK_SIZE = 32768


def download_svhn(url, outfile):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    session = requests.Session()
    response = session.get(url, stream=True)
    with open(outfile, 'wb') as fp:
        dl = 0
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                dl += len(chunk)
                fp.write(chunk)

                mb = dl / 1.0e6
                sys.stdout.write('\r%.2f MB downloaded...' % (mb))
                sys.stdout.flush()

        sys.stdout.write('\nFinish!\n')
        sys.stdout.flush()


def download_svhn_extra():
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    session = requests.Session()
    response = session.get(url_extra, stream=True)
    with open(outfile_extra, 'wb') as fp:
        dl = 0
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                dl += len(chunk)
                fp.write(chunk)

                mb = dl / 1.0e6
                sys.stdout.write('\r%.2f MB downloaded...' % (mb))
                sys.stdout.flush()

        sys.stdout.write('\nFinish!\n')
        sys.stdout.flush()


def preprocess(X):
    X = np.transpose(X, axes=[3, 0, 1, 2])
    X = (X / 255.0).astype('float32')
    return X


def load_data(include_extra=False):
    """
    Load and return dataset as tuple (data, label, label_strings)
    """

    if not os.path.exists(outfile_train):
        download_svhn(url_train, outfile_train)

    if not os.path.exists(outfile_test):
        download_svhn(url_test, outfile_test)

    mat = sp.io.loadmat(outfile_train)
    x_train = mat['X']
    y_train = mat['y']

    mat = sp.io.loadmat(outfile_test)
    x_test = mat['X']
    y_test = mat['y']

    if include_extra:
        if not os.path.isfile(outfile_extra):
            download_svhn_extra()
        mat_e = sp.io.loadmat(outfile_extra)
        x_train = np.concatenate((x_train, mat_e['X']), axis=-1)
        y_train = np.concatenate((y_train, mat_e['y']), axis=0)

    x_train = preprocess(x_train)
    x_test = preprocess(x_test)
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0
    y_test = np.squeeze(y_test)
    y_train = keras.utils.to_categorical(y_train)
    y_train = y_train.astype('float32')

    return x_train, y_train, x_test, y_test, [str(i) for i in range(10)]

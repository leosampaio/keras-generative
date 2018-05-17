import os
import requests
import sys

import numpy as np
import scipy as sp
import scipy.io

import keras

url = 'http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy'
curdir = os.path.abspath(os.path.dirname(__file__))
outdir = os.path.join(curdir, 'files')
outfile = os.path.join(outdir, 'moving_mnist.npy')

CHUNK_SIZE = 32768

def download_moving_mnist():
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

def load_data():
    """
    Load and return dataset as tuple (data, label, label_strings)
    """

    if not os.path.exists(outfile):
        download_moving_mnist()

    data = np.load(outfile)
    data = np.moveaxis(data, 1, 0) # fix inverted axis setup
    data = (data / 255.0).astype('float32')
    data = np.expand_dims(data, -1)

    return data

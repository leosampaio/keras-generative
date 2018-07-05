from flask import Flask, request, abort
import tensorflow as tf
import pickle
import json
import numpy as np
import time
import h5py

from metrics import inception_score

app = Flask(__name__)  # create the Flask app


def to_rgb(x):
    x = x * 255.
    if x.shape[3] == 1:
        n, w, h, _ = x.shape
        ret = np.empty((n, w, h, 3), dtype=np.uint8)
        ret[:, :, :, 2] = ret[:, :, :, 1] = ret[:, :, :, 0] = x[:, :, :, 0]
    else:
        ret = x
    return ret


@app.route('/inception-score', methods=['POST'])
def compute_inception_score():
    start = time.time()
    req_data = request.get_json()
    filename = req_data['filename']
    with h5py.File(filename, 'r') as hf:
        data = hf['feats'][:]
        mean, std = inception_score.get_inception_score(to_rgb(data))
    return json.dumps({'mean': repr(mean),
                       'std': repr(std),
                       'computation_time': str(time.time() - start)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

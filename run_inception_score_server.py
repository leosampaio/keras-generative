from flask import Flask, request, abort
import tensorflow as tf
import pickle
import json
import numpy as np
import time
import h5py

from models import inception_score

app = Flask(__name__)  # create the Flask app

@app.route('/inception-score', methods=['POST'])
def compute_inception_score():
    start = time.time()
    req_data = request.get_json()
    filename = req_data['filename']
    with h5py.File(filename, 'r') as hf:
        data = hf['feats'][:]
        mean, std = inception_score.get_inception_score(data)
    return json.dumps({'mean': repr(mean),
                       'std': repr(std),
                       'computation_time': str(time.time() - start)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

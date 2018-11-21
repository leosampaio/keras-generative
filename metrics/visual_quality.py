import json
import requests
import time
import logging
import os
import glob

import numpy as np
import tensorflow as tf
import keras.backend as K
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

from core.metrics import HistoryMetric, ProjectionMetric
from metrics import inception_score
from metrics import mmd


def to_rgb(x):
    x = x * 255.
    if x.shape[3] == 1:
        n, w, h, _ = x.shape
        ret = np.empty((n, w, h, 3), dtype=np.uint8)
        ret[:, :, :, 2] = ret[:, :, :, 1] = ret[:, :, :, 0] = x[:, :, :, 0]
    else:
        ret = x
    return ret


def get_latest_file_from_path(path):
    list_of_files = glob.glob(path)  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


class InceptionScore(HistoryMetric):
    name = 'inception_score'
    input_type = 'generated_and_real_samples'

    def compute(self, input_data):
        x_hat, _ = input_data
        mean, std = inception_score.get_inception_score(to_rgb(x_hat))
        return mean


class RemoteInceptionScore(HistoryMetric):
    name = 'r_inception_score'
    input_type = 'generated_and_real_samples'

    def __init__(self, experiment_id, output='output', **kwargs):
        super().__init__()
        self.precomputed_file_path = os.path.join(
            output, experiment_id,
            'tmp/precomputed_generated_and_real_samples_*')

    def get_server_ip(self):
        server = 'localhost'
        with open("server_for_inception_score.config") as f:
            server = f.readline()
            server = server.rstrip()
        return server

    def compute(self, input_data):
        server = self.get_server_ip()
        filename = get_latest_file_from_path(self.precomputed_file_path)
        json_data = {'filename': filename}
        start = time.time()
        headers = {'Content-Type': 'application/json'}
        logging.getLogger("requests").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        r = requests.post("http://{}:{}/inception-score".format(server, 5000),
                          headers=headers,
                          data=json.dumps(json_data))
        print("[Remote IS] Total request took {}s".format(time.time() - start))
        response_data = json.loads(r.text)
        print("[Remote IS] Computation took {}s".format(response_data['computation_time']))
        return float(response_data['mean'])


class MultiprocessInceptionScore(HistoryMetric):
    name = 'm_inception_score'
    input_type = 'generated_and_real_samples'

    def __init__(self, experiment_id, output='output', **kwargs):
        super().__init__()
        self.precomputed_file_path = os.path.join(
            output, experiment_id,
            'tmp/precomputed_generated_and_real_samples_*')

    def get_server_ip(self):
        return'localhost'

    compute = RemoteInceptionScore.__dict__['compute']


class MaximumMeanDiscrepancy(HistoryMetric):
    name = 'mmd'
    input_type = 'generated_and_real_samples'

    def __init__(self, input_shape=(32, 32, 1), **kwargs):
        super().__init__()
        x_ph = tf.placeholder(tf.float32, shape=[None] + list(input_shape), name='mmd_x')
        x_hat_ph = tf.placeholder(tf.float32, shape=[None] + list(input_shape), name='mmd_x_hat')
        x_flat = K.batch_flatten(x_ph)
        x_hat_flat = K.batch_flatten(x_hat_ph)
        self.mmd_computer = tf.log(mmd.rbf_mmd2(x_flat, x_hat_flat))

    def compute(self, input_data):
        x_hat, x = input_data
        mmd = K.get_session().run(self.mmd_computer, feed_dict={'mmd_x:0': x, 'mmd_x_hat:0': x_hat})
        return mmd


class SyntheticDatasetVis(ProjectionMetric):
    name = 'synthetic-data-vis'
    input_type = 'generated_and_real_samples'

    def compute(self, input_data):
        x_hat, x = input_data
        y = np.concatenate((np.ones((len(x_hat),)) * -1, np.ones((len(x),))))
        x_all = np.concatenate((x, x_hat), axis=0)

        return np.concatenate((x_all, np.expand_dims(y, axis=1)),
                              axis=1)


class SyntheticDatasetVis(ProjectionMetric):
    name = 'synthetic-data-vis-generated'
    input_type = 'generated_and_real_samples'

    def compute(self, input_data):
        x_hat, x = input_data
        y = np.concatenate((np.ones((1,)) * -1, np.ones((len(x),))))
        x_all = np.concatenate((x[:1], x_hat), axis=0)

        return np.concatenate((x_all, np.expand_dims(y, axis=1)),
                              axis=1)


class SyntheticDatasetModeCoverage(HistoryMetric):
    name = 'mode-coverage'
    input_type = 'generated_samples_and_possible_labels'

    def compute(self, input_data):
        x, distributions = input_data

        means = [mu for mu, _ in distributions]

        nn = NearestNeighbors(n_neighbors=1, metric='l2')
        nn.fit(means)
        distances, modes = nn.kneighbors(x, return_distance=True)

        std = distributions[0][1][0]
        if distributions.shape[2] > 2:
            limit = 10 * std
        else:
            limit = 3 * std
        covered_modes = np.unique(modes[np.where(distances < limit)[0]])

        return len(covered_modes)


class SyntheticDatasetHighQualitySamples(HistoryMetric):
    name = 'high-quality-ratio'
    input_type = 'generated_samples_and_possible_labels'

    def compute(self, input_data):
        x, distributions = input_data

        means = [mu for mu, _ in distributions]

        nn = NearestNeighbors(n_neighbors=1, metric='l2')
        nn.fit(means)
        distances, modes = nn.kneighbors(x, return_distance=True)

        std = distributions[0][1][0]
        if distributions.shape[2] > 2:
            limit = 10 * std
        else:
            limit = 3 * std
        high_quality_samples_ratio = np.mean(distances < limit)

        return high_quality_samples_ratio


class StackedMNISTModeEstimator(HistoryMetric):
    name = 'mode-count-estimator'
    input_type = 'classification_samples'

    def compute(self, input_data):
        prediction_x_hat, prediction_x = input_data

        modes = len(np.unique(np.argmax(prediction_x_hat, axis=1)))

        return modes


class StackedMNISTKL(HistoryMetric):
    name = 'clas-kl-divergence'
    input_type = 'classification_samples'

    def compute(self, input_data):
        prediction_x_hat, prediction_x = input_data

        p_false = np.argmax(prediction_x_hat, axis=1)
        # p_real = np.argmax(prediction_x, axis=1)
        total_modes = prediction_x_hat.shape[1]

        modes_count = np.zeros((total_modes, 1))
        for i in range(p_false.shape[0]):
            modeNum = int(p_false[i])
            modes_count[modeNum] += 1

        # calculate KL
        modes_count_normalized = modes_count / np.sum(modes_count)
        kl = 0
        Pdata = 1. / total_modes
        for i in range(total_modes):
            if int(modes_count[i]) == 0:
                continue
            kl += modes_count_normalized[i] * (np.log(modes_count_normalized[i]) - np.log(Pdata))

        return kl[0]

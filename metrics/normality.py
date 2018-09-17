import scipy as sp
import numpy as np

from core.metrics import HistoryMetric, HistogramMetric


class DagostinoNormalityTest(HistoryMetric):
    name = 'dagostino-normality'
    input_type = 'triplet_distance_vectors'

    def compute(self, input_data):
        k2, p = sp.stats.normaltest(np.squeeze(input_data))
        return p


class ShapiroNormalityTest(HistoryMetric):
    name = 'shapiro-normality'
    input_type = 'triplet_distance_vectors'

    def compute(self, input_data):
        W, p = sp.stats.shapiro(np.squeeze(input_data))
        return p


class NormalityCheckHistogram(HistogramMetric):
    name = 'histogram-normality'
    input_type = 'triplet_distance_vectors'

    def compute(self, input_data):
        return np.squeeze(input_data)

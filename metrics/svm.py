import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from core.metrics import Metric, HistoryMetric

class SVMEval(HistoryMetric):
    name = 'svm_eval'
    input_type = 'labelled_embedding'

    def compute(self, input_data):
        x_feats, y_labels = input_data
        x_train, y_train, x_test, y_test = x_feats[1000:], y_labels[1000:], x_feats[:1000], y_labels[:1000]

        # rescale data
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # grid search is performed on those C values
        param_grid = [{'C': [1e1]}]

        # get 10 stratified shuffle splits of 1000 samples (stratified meaning it
        # keeps the class distribution intact).
        dataset_splits = StratifiedShuffleSplit(
            n_splits=5, train_size=1000, test_size=0.2).split(x_train, y_train)

        # perform grid search on each different split and get best linear SVM
        scores = []
        for split in dataset_splits:
            grid = GridSearchCV(LinearSVC(),
                                param_grid=param_grid,
                                cv=[split])
            grid.fit(x_train, y_train)
            score_on_test = grid.score(x_test, y_test)
            scores.append(score_on_test)

        return np.mean(scores)

class SVMRBFEval(HistoryMetric):
    name = 'svm_rbf_eval'
    input_type = 'labelled_embedding'

    def compute(self, input_data):
        x_feats, y_labels = input_data
        x_train, y_train, x_test, y_test = x_feats[1000:], y_labels[1000:], x_feats[:1000], y_labels[:1000]

        # rescale data
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # grid search is performed on those C values
        param_grid = [{'C': [1e1], 'kernel': ['rbf']}]

        # get 10 stratified shuffle splits of 1000 samples (stratified meaning it
        # keeps the class distribution intact).
        dataset_splits = StratifiedShuffleSplit(
            n_splits=5, train_size=1000, test_size=0.2).split(x_train, y_train)

        # perform grid search on each different split and get best linear SVM
        scores = []
        for split in dataset_splits:
            grid = GridSearchCV(SVC(),
                                param_grid=param_grid,
                                cv=[split])
            grid.fit(x_train, y_train)
            score_on_test = grid.score(x_test, y_test)
            scores.append(score_on_test)

        return np.mean(scores)

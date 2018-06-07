import os
import random
from abc import ABCMeta, abstractmethod
from pprint import pprint

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE

import keras
from keras.engine.topology import Layer
from keras import Input, Model
from keras.layers import (Flatten, Dense, Activation, Reshape,
                          BatchNormalization, Concatenate, Dropout, LeakyReLU,
                          LocallyConnected2D, Add,
                          Lambda, AveragePooling1D, GlobalAveragePooling2D)
from keras.optimizers import Adam, Adadelta, RMSprop
from keras import initializers
from keras import backend as K
from keras.applications.mobilenet import MobileNet

from .base import BaseModel

from .utils import *
from .layers import *

def svm_eval(x_train, y_train, x_test, y_test):

        # rescale data
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # grid search is performed on those C values
        param_grid = [{'C': [1e1]}]

        # get 10 stratified shuffle splits of 1000 samples (stratified meaning it
        # keeps the class distribution intact).
        dataset_splits = StratifiedShuffleSplit(
            n_splits=2, train_size=1000, test_size=0.2).split(x_train, y_train)

        # perform grid search on each different split and get best linear SVM
        scores = []
        for split in dataset_splits:
            grid = GridSearchCV(LinearSVC(),
                                param_grid=param_grid,
                                cv=[split])
            grid.fit(x_train, y_train)
            score_on_test = grid.score(x_test, y_test)
            scores.append(score_on_test)

        mean = np.mean(scores)
        return mean

def svm_rbf_eval(x_train, y_train, x_test, y_test):

        # rescale data
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        # grid search is performed on those C values
        param_grid = [{'C': [1e1], 'kernel': ['rbf']}]

        # get 10 stratified shuffle splits of 1000 samples (stratified meaning it
        # keeps the class distribution intact).
        dataset_splits = StratifiedShuffleSplit(
            n_splits=2, train_size=1000, test_size=0.2).split(x_train, y_train)

        # perform grid search on each different split and get best linear SVM
        scores = []
        for split in dataset_splits:
            grid = GridSearchCV(SVC(),
                                param_grid=param_grid,
                                cv=[split])
            grid.fit(x_train, y_train)
            score_on_test = grid.score(x_test, y_test)
            scores.append(score_on_test)

        mean = np.mean(scores)
        return mean

def tsne(x_test, y_test):

    tsne = TSNE(n_components=2,
                verbose=1, perplexity=30,
                n_iter=1000)
    tsne_results = tsne.fit_transform(x_test)
    return np.concatenate((tsne_results, np.expand_dims(y_test, axis=1)), axis=1)

def lda(x_test, y_test):

    lda = LinearDiscriminantAnalysis(n_components=2)
    lda_result = lda.fit_transform(x_test, y_test)
    return np.concatenate((lda_result, np.expand_dims(y_test, axis=1)), axis=1)

def pca(x_test, y_test):

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(x_test)
    return np.concatenate((pca_result, np.expand_dims(y_test, axis=1)), axis=1)
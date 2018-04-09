#!/usr/bin/python
# based on
# https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b

import matplotlib
matplotlib.use('Agg')

import sys
import argparse
import logging
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import time
from sklearn.manifold import TSNE
from ggplot import *
from sklearn.decomposition import PCA
import scipy.io
from models.notifyier import *

logging.basicConfig()

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "img_features_file", help="feature HDF5 or matlab file with keys 'feats' and 'labels'")
    parser.add_argument(
        "skt_features_file", help="feature HDF5 or matlab file with keys 'feats' and 'labels'")
    parser.add_argument(
        '-o', "--output_image", help="image file path to save the resulting plot", default="tmp.jpg")
    parser.add_argument(
        "-p", "--perplexity", help="persplexity value to use on the t-sne",
        type=int, default=50)
    parser.add_argument(
        "-t", "--title", help="title for the plot",
        default="t-SNE Plot of CNN Feature Space")
    parser.add_argument(
        "-d", "--dimensions",
        help="initial dimension cut to perform using PCA before the t-sne",
        type=int, default=64)
    parser.add_argument(
        "-c", "--classes",
        help="number of classes to plot",
        type=int, default=10)
    parser.add_argument("--pca", action='store_true',
                        help="use only PCA to plot")
    parser.add_argument("-m", "--matlab", action='store_true',
                        help="feature file is a matlab file")
    args = parser.parse_args()

    # read file with features and labels
    if args.matlab:
        matfile = scipy.io.loadmat(args.img_features_file)
        img_labels = matfile['labels']
        img_features = matfile['feats']
        matfile = scipy.io.loadmat(args.skt_features_file)
        skt_labels = matfile['labels']
        skt_features = matfile['feats']

        img_labels = np.squeeze(img_labels)
        skt_labels = np.squeeze(skt_labels)
    else:
        with h5py.File(args.img_features_file, 'r') as hf:
            img_labels = hf['labels'][:]
            img_features = hf['feats'][:]
        with h5py.File(args.skt_features_file, 'r') as hf:
            skt_labels = hf['labels'][:]
            skt_features = hf['feats'][:]

    # cut throught matrices to select a specific number of classes to show
    # (eases visualization by avoiding color-cacophony)
    indices_to_select = np.isin(img_labels, [i for i in np.unique(img_labels)[:args.classes]])
    img_labels = np.array([y for (y, i) in zip(img_labels, indices_to_select) if i])
    img_features = np.array([x for (x, i) in zip(img_features, indices_to_select) if i])

    indices_to_select = np.isin(skt_labels, [i for i in np.unique(skt_labels)[:args.classes]])
    skt_labels = np.array([y for (y, i) in zip(skt_labels, indices_to_select) if i])
    skt_features = np.array([x for (x, i) in zip(skt_features, indices_to_select) if i])

    # mix images and sketches
    end_id_img_features = img_labels.shape[0]
    features = np.concatenate((img_features, skt_features))
    labels = np.concatenate((img_labels, skt_labels))

    # transform matrices into pandas data frame (easier to plot)
    feat_cols = ['feat'+str(i) for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feat_cols)
    df['label'] = labels
    df['label'] = df['label'].apply(lambda i: str(i))

    # apply t-sne algorithm
    time_start = time.time()
    tsne = TSNE(n_components=2,
                verbose=1, perplexity=args.perplexity,
                n_iter=10000)
    tsne_results = tsne.fit_transform(df.values)

    df_tsne = df.copy()
    df_tsne['x-tsne'] = tsne_results[:, 0]
    df_tsne['y-tsne'] = tsne_results[:, 1]
    df_tsne['shape'] = np.repeat(['.', 'v'], [end_id_img_features, tsne_results.shape[0] - end_id_img_features])

    # plot using semitransparent dots on the learned t-sne space
    chart = (ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label', group='shape'))
            + geom_point(aes(shape='shape'), size=70, alpha=0.7)
            + ggtitle(args.title))
    chart.save(args.output_image)
    notify_with_message("[{}] t-SNE Plot".format(args.img_features_file))
    notify_with_image(args.output_image)

if __name__ == "__main__":
    main(sys.argv[1:])

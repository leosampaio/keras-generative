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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from models.notifyier import *

logging.basicConfig()

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "features_file", help="feature HDF5 file with keys 'feats' and 'labels'")
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
    parser.add_argument("--lda", action='store_true',
                        help="use only LDA to plot")
    args = parser.parse_args()

    # read file with features and labels
    with h5py.File(args.features_file, 'r') as hf:
        labels = hf['labels'][:]
        features = hf['feats'][:]

    # cut throught matrices to select a specific number of classes to show
    # (eases visualization by avoiding color-cacophony)
    indices_to_select = np.isin(labels, [i for i in np.unique(labels)[:args.classes]])
    labels = np.array([y for (y, i) in zip(labels, indices_to_select) if i])
    features = np.array([x for (x, i) in zip(features, indices_to_select) if i])

    # transform matrices into pandas data frame (easier to plot)
    feat_cols = ['feat'+str(i) for i in range(features.shape[1])]
    df = pd.DataFrame(features, columns=feat_cols)
    df['label'] = labels
    df['label'] = df['label'].apply(lambda i: str(i))

    if args.pca:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df[feat_cols].values)

        df['pca-one'] = pca_result[:,0]
        df['pca-two'] = pca_result[:,1] 

        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

        chart = (ggplot(df, aes(x='pca-one', y='pca-two', color='label'))
        + geom_point(size=75,alpha=0.8)
        + ggtitle(args.title))
        chart.save(args.output_image)
        notify_with_message("[{}] PCA Plot".format(args.features_file))
        notify_with_image(args.output_image)
        return

    if args.lda:
        lda = LinearDiscriminantAnalysis(n_components=2)
        lda_result = lda.fit_transform(df[feat_cols].values, labels)

        df['lda-one'] = lda_result[:,0]
        df['lda-two'] = lda_result[:,1] 

        print('Explained variation per principal component: {}'.format(lda.explained_variance_ratio_))

        chart = (ggplot(df, aes(x='lda-one', y='lda-two', color='label'))
        + geom_point(size=75,alpha=0.8)
        + ggtitle(args.title))
        chart.save(args.output_image)
        notify_with_message("[{}] LDA Plot".format(args.features_file))
        notify_with_image(args.output_image)
        return

    # get lower resoluton embedding of the space to improve t-sne convergence
    pca = PCA(n_components=args.dimensions)
    pca_result = pca.fit_transform(df[feat_cols].values)
    print('Explained variation per principal component (PCA): {}'.format(np.sum(pca.explained_variance_ratio_)))

    # apply t-sne algorithm
    time_start = time.time()
    tsne = TSNE(n_components=2,
                verbose=1, perplexity=args.perplexity,
                n_iter=250)
    tsne_results = tsne.fit_transform(pca_result)

    df_tsne = df.copy()
    df_tsne['x-tsne'] = tsne_results[:, 0]
    df_tsne['y-tsne'] = tsne_results[:, 1]

    # plot using semitransparent dots on the learned t-sne space
    import pdb; pdb.set_trace()  # breakpoint 0904155f //
    chart = (ggplot(df_tsne, aes(x='x-tsne', y='y-tsne', color='label'))
            + geom_point(size=70,alpha=0.5)
            # + xaxt('n') + yaxt('n') 
            + scale_x_continuous("", breaks=[0], labels=[''])
            + scale_y_continuous("", breaks=[0], labels=[''])
            + theme(x_axis_text=element_text(text=""), y_axis_text=element_text(text=""))
            + xlab('') + ylab('')
            + ggtitle(args.title))
    chart.save(args.output_image)
    notify_with_message("[{}] t-SNE Plot".format(args.features_file))
    notify_with_image(args.output_image)

if __name__ == "__main__":
    main(sys.argv[1:])

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from core.metrics import ProjectionMetric


class TSNEProjection(ProjectionMetric):
    name = 'tsne'
    input_type = 'labelled_embedding'

    def compute(self, input_data):
        if len(input_data) == 2:
            x_feats, y_labels = input_data
            x_train, y_train, x_test, y_test = x_feats[1000:], y_labels[1000:], x_feats[:1000], y_labels[:1000]
        else:
            x_train, y_train, x_test, y_test = input_data

        tsne = TSNE(n_components=2,
                    verbose=1, perplexity=30,
                    n_iter=1000)
        tsne_results = tsne.fit_transform(x_test)
        return np.concatenate((tsne_results, np.expand_dims(y_test, axis=1)),
                              axis=1)


class LDAProjection(ProjectionMetric):
    name = 'lda'
    input_type = 'labelled_embedding'

    def compute(self, input_data):
        if len(input_data) == 2:
            x_feats, y_labels = input_data
            x_train, y_train, x_test, y_test = x_feats[1000:], y_labels[1000:], x_feats[:1000], y_labels[:1000]
        else:
            x_train, y_train, x_test, y_test = input_data

        lda = LinearDiscriminantAnalysis(n_components=2)
        lda.fit(x_train, y_train)
        lda_result = lda.transform(x_test)
        if lda_result.shape[1] == 1:
            raise ValueError("To use LDA projection, you need more than 2 dimensions")
        return np.concatenate((lda_result, np.expand_dims(y_test, axis=1)),
                              axis=1)


class PCAProjection(ProjectionMetric):
    name = 'pca'
    input_type = 'labelled_embedding'

    def compute(self, input_data):
        if len(input_data) == 2:
            x_feats, y_labels = input_data
            x_train, y_train, x_test, y_test = x_feats[1000:], y_labels[1000:], x_feats[:1000], y_labels[:1000]
        else:
            x_train, y_train, x_test, y_test = input_data

        pca = PCA(n_components=2)
        pca.fit(x_train)
        pca_result = pca.transform(x_test)
        return np.concatenate((pca_result, np.expand_dims(y_test, axis=1)),
                              axis=1)

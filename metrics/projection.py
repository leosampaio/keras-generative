import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from core.metrics import ProjectionMetric


class TSNEProjection(ProjectionMetric):
    name = 'tsne'
    input_type = 'labelled_embedding'

    def compute(self, input_data):
        x_feats, y_labels = input_data
        x_test, y_test = x_feats[:1000], y_labels[:1000]

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
        x_feats, y_labels = input_data
        x_test, y_test = x_feats[:1000], y_labels[:1000]

        lda = LinearDiscriminantAnalysis(n_components=2)
        lda_result = lda.fit_transform(x_test, y_test)
        return np.concatenate((lda_result, np.expand_dims(y_test, axis=1)),
                              axis=1)


class PCAProjection(ProjectionMetric):
    name = 'pca'
    input_type = 'labelled_embedding'

    def compute(self, input_data):
        x_feats, y_labels = input_data
        x_test, y_test = x_feats[:1000], y_labels[:1000]

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(x_test)
        return np.concatenate((pca_result, np.expand_dims(y_test, axis=1)),
                              axis=1)

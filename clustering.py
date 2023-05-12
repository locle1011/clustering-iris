import warnings
warnings.filterwarnings(action='ignore')

import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, n_clusters=3, init='random', seed=None):
        np.random.seed(seed)
        self.n_clusters = n_clusters
        if init == 'random':
            self.init_centroids = self.random_init
        elif init == 'al-daoud':
            self.init_centroids = self.al_daoud_init
        else:
            self.init_centroids = init

        self.centroids = 0

    @staticmethod
    def get_median_index_of_subset(n_samples, n_clusters):
        min_samples_by_cluster = n_samples//n_clusters
        n_max_samples_by_cluster = n_samples % n_clusters
        samples_by_cluster = np.array([min_samples_by_cluster+1]*n_max_samples_by_cluster + [
            min_samples_by_cluster]*(n_clusters-n_max_samples_by_cluster))
        index_median_by_cluster = samples_by_cluster//2
        index_median_by_cluster[1:] += (np.cumsum(samples_by_cluster[:-1]))
        return index_median_by_cluster

    def al_daoud_init(self, X: np.ndarray):
        max_var_index = np.argmax(np.var(X, axis=0))
        median_indexes = self.get_median_index_of_subset(X.shape[0], self.n_clusters)
        return X[np.argpartition(X[:, max_var_index], median_indexes)][median_indexes, :]

    def random_init(self, X: np.ndarray):
        return X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

    def predict(self, X: np.ndarray):
        distances = cdist(X, self.centroids, metric='euclidean')
        return np.argmin(distances, axis=1)

    def is_converged(self, current_centroids, new_centroids):
        return np.array_equal(current_centroids, new_centroids)

    def update_centroids(self, X: np.ndarray, labels):
        centroids = np.zeros(shape=(self.n_clusters, X.shape[1]))
        for c in range(self.n_clusters):
            Xc = X[labels == c]
            centroids[c] = np.mean(Xc, axis=0)
        return centroids

    def fit(self, X: np.ndarray, history=False):
        if history:
            return self.fit_with_history(X)
        else:
            return self.fit_without_history(X)

    def fit_without_history(self, X: np.ndarray):
        self.centroids = self.init_centroids(X)
        while True:
            new_centroids = self.update_centroids(X, self.predict(X))
            if self.is_converged(self.centroids, new_centroids):
                break
            else:
                self.centroids = new_centroids
        return self

    def fit_with_history(self, X: np.ndarray):
        self.centroids = self.init_centroids(X)
        centroids = [self.centroids]
        labels = []
        while True:
            labels.append(self.predict(X))
            new_centroids = self.update_centroids(X, labels[-1])
            if self.is_converged(self.centroids, new_centroids):
                break
            else:
                self.centroids = new_centroids
                centroids.append(new_centroids)

        return self, centroids, labels

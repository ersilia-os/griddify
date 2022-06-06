import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

from tqdm import tqdm


class FeatureDistances(object):
    def __init__(self, metric="cosine", max_n=10000, max_subsampling_rounds=5):
        self.metric = metric
        self.max_n = max_n
        self.max_subsampling_rounds = max_subsampling_rounds

    def _transpose(self, data):
        return np.array(data).T

    def _subsample(self, X):
        N = X.shape[1]
        idxs = np.array([i for i in range(N)])
        visited_idxs = set()
        round_idxs = []
        seed = 42
        for _ in range(self.max_subsampling_rounds):
            n = min(N, self.max_n)
            np.random.seed(seed=seed)
            idxs_ = np.random.choice(idxs, size=n, replace=False)
            round_idxs += [idxs_]
            visited_idxs.update(list(idxs_))
            if len(visited_idxs) == len(idxs):
                break
            seed += 1
        return round_idxs

    def calculate(self, data):
        data = pd.DataFrame(data)
        X = self._transpose(data)
        round_idxs = self._subsample(X)
        D = np.zeros((X.shape[0], X.shape[0], len(round_idxs)))
        for k, idxs in tqdm(enumerate(round_idxs)):
            X_ = X[:, idxs]
            D_ = squareform(pdist(X_, metric=self.metric))
            D[:, :, k] = D_
        D = np.nanmean(D, axis=2)
        D[np.isnan(D)] = np.nanmax(D)
        return pd.DataFrame(D, columns=list(data.columns))

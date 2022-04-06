import warnings

warnings.filterwarnings("ignore")

import numpy as np
from umap import UMAP
from sklearn.preprocessing import StandardScaler, MinMaxScaler


ZSCORE_CLIP = (-5, 5)

RANDOM_SEED = 42


class Tabular2Cloud(object):
    def __init__(self):
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.zscore_clip = ZSCORE_CLIP

    def _clip(self, X):
        return np.clip(X, self.zscore_clip[0], self.zscore_clip[1])

    def _is_distance_matrix(self, X):
        tol = 1e-8
        if X.shape[0] != X.shape[1]:
            return False
        for i in range(X.shape[0]):
            if X[i, i] != 0:
                return False
        return np.all(np.abs(X - X.T) < tol)

    def fit(self, X):
        X = np.array(X)
        if self._is_distance_matrix(X):
            self.reducer = UMAP(metric="precomputed", random_state=RANDOM_SEED)
        else:
            self.reducer = UMAP(random_state=RANDOM_SEED)
        self.reducer.fit(X)
        Xt = self.reducer.transform(X)
        self.standard_scaler.fit(Xt)
        Xt = self.standard_scaler.transform(Xt)
        Xt = self._clip(Xt)
        self.minmax_scaler.fit(Xt)

    def transform(self, X):
        X = np.array(X)
        Xt = self.reducer.transform(X)
        Xt = self.standard_scaler.transform(Xt)
        Xt = self._clip(Xt)
        Xt = self.minmax_scaler.transform(Xt)
        return Xt

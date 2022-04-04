import numpy as np


class Flat2Grid(object):
    def __init__(self, mappings, side):
        self._mappings = mappings
        self._side = side

    def transform(self, X):
        X = np.array(X)
        Xt = np.zeros((X.shape[0], self._side, self._side))
        for i in range(X.shape[0]):
            x = X[i, :]
            for j, v in enumerate(x):
                idx_i, idx_j = self._mappings[j]
                Xt[i, idx_i, idx_j] = v
        return Xt

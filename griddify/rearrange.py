import numpy as np


class Flat2Grid(object):
    def __init__(self, mappings, side):
        self._mappings = mappings
        self._side = side

    def transform(self, X):
        X = np.array(X)
        Xt_sum = np.zeros((X.shape[0], self._side, self._side))
        Xt_cnt = np.zeros(Xt_sum.shape, dtype=int)
        for i in range(X.shape[0]):
            x = X[i, :]
            for j, v in enumerate(x):
                idx_i, idx_j = self._mappings[j]
                Xt_sum[i, idx_i, idx_j] += v
                Xt_cnt[i, idx_i, idx_j] += 1
        Xt = Xt_sum / Xt_cnt
        return Xt

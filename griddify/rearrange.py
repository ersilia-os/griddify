import numpy as np
import collections


class Flat2Grid(object):
    def __init__(self, mappings, side):
        self._mappings = mappings
        self._side = side
        self._method = np.mean # TODO: include median, most-extreme, ...

    def transform(self, X):
        X = np.array(X)
        d = collections.defaultdict(list)
        for i in range(X.shape[0]):
            x = X[i, :]
            for j, v in enumerate(x):
                idx_i, idx_j = self._mappings[j]
                d[(i, idx_i, idx_j)] += [v]
        Xt = np.zeros((X.shape[0], self._side, self._side))
        for k,v in d.items():
            Xt[k[0], k[1], k[2]] = self._method(v)
        return Xt

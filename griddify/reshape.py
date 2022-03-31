import numpy as np


class Tabular2Grid(object):

    def __init__(self, cloud2grid):
        self._size = cloud2grid._size
        self._empty_grid = cloud2grid.get_empty_grid()
        
    def fit(self, X_grid):
        assert X_grid.astype(int) == X_grid
        self._mapper = X_grid
        
    def transform(self, X):
        Xt = []
        for i in X.shape[0]:
            g = self._empty_grid.copy()
            x = X[i,:]
            for j, v in enumerate(x):
                idx_i, idx_j = self._mapper[j]
                g[idx_i, idx_j] = v
            Xt += [g]
        return np.array(Xt)
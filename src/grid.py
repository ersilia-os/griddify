import numpy as np
from scipy.spatial.distance import cdist
import lap


class Cloud2Grid(object):

    def __init__(self, max_side=128):
        self._max_side=max_side

    def fit(self, X_2d):
        self.side = np.min([self._max_side, np.ceil(np.sqrt(X_2d.shape[0]))])
        xv, yv = np.meshgrid(np.linspace(0, 1, self.side), np.linspace(0, 1, self.side))
        self.grid = np.dstack((xv, yv)).reshape(-1, 2)
        cost = cdist(self.grid, X_2d, 'sqeuclidean')
        cost = cost * (1000000 / cost.max())
        cost = cost.astype(int)
        min_cost, row_assigns, col_assigns = lap.lapjv(cost)
        grid_jv = self.grid[col_assigns]
        self._min_cost = min_cost
        self._row_assigns = row_assigns
        self._col_assigns = col_assigns
        self.grid_jv = grid_jv

    def transform(self, X_2d):
        pass

    def save(self):
        pass

    def load(self):
        pass



class Tabular2Grid(object):

    def __init__(self, shape, metric="cosine", reducer="umap"):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass
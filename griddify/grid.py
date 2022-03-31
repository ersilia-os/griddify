import numpy as np
from scipy.spatial.distance import cdist
import lap
from sklearn.cluster import KMeans


class Cloud2Grid(object):

    def __init__(self, max_side=128):
        self._max_side=max_side
        self._clusters=None

    def _is_cloud(self, X):
        if len(X.shape) != 2:
            return False
        if X.shape[1] != 2:
            return False
        return True

    def _find_side(self, X):
        side = int(np.sqrt(X.shape[0]))
        if side > self._max_side:
            side = self._max_side
        return side

    def _needs_downsampling(self, X):
        avail = self._side**2
        assert avail <= X.shape[0]
        if avail == X.shape[0]:
            return False
        else:
            return True

    def _downsample_with_clustering(self, X):
        self._clusters = KMeans(n_clusters=self._side**2)
        self._clusters.fit(X)
        

    def fit(self, X):
        assert self._is_cloud(X)
        self._side = self._find_side(X)
        self._do_cluster = self._needs_downsampling(X)
        



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

    def transform(self, X):
        assert self._is_cloud(X)

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
import numpy as np
import lap
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


GRIDDIFY_METRIC = "sqeuclidean"


class Cloud2Grid(object):
    def __init__(self, max_side=128):
        self._max_side = max_side
        self._clusters = None

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
        centroids = self._clusters.cluster_centers_
        return centroids

    def _griddify(self, X):
        xv, yv = np.meshgrid(
            np.linspace(0, 1, self._side), np.linspace(0, 1, self._side)
        )
        self.grid = np.dstack((xv, yv)).reshape(-1, 2)
        cost = cdist(self.grid, X, GRIDDIFY_METRIC)
        cost = cost * (1000000 / cost.max())
        cost = cost.astype(int)
        min_cost, row_assigns, col_assigns = lap.lapjv(cost)
        self._grid_jv = self.grid[col_assigns]
        self._min_cost = min_cost
        self._row_assigns = row_assigns
        self._col_assigns = col_assigns

    def _grid_coordinates_as_integers(self, X):
        d = dict((v, i) for i, v in enumerate(np.linspace(0, 1, self._side)))
        Xt = np.zeros(X.shape, dtype=int)
        for i in range(X.shape[0]):
            Xt[i] = [d[X[i, 0]], d[X[i, 1]]]
        return Xt

    def fit(self, X):
        assert self._is_cloud(X)
        self._side = self._find_side(X)
        self._do_cluster = self._needs_downsampling(X)
        if self._do_cluster:
            X = self._downsample_with_clustering(X)
        self._griddify(X)
        self.nearest_neighbors = NearestNeighbors(n_neighbors=1)
        self.nearest_neighbors.fit(X)

    def transform(self, X, as_integers=False):
        assert self._is_cloud(X)
        idxs = self.nearest_neighbors.kneighbors(X, return_distance=False)[:, 0]
        X_grid = np.zeros((X.shape[0], 2))
        for i, idx in enumerate(idxs):
            X_grid[i] = self._grid_jv[idx]
        if as_integers:
            return self._grid_coordinates_as_integers(X_grid)
        else:
            return X_grid

    def get_mappings(self, X):
        X = self.transform(X, as_integers=True)
        return (X, self._side)


class Cloud2CircleGrid(object):
    def __init__(self, max_side=128):
        self._max_side = max_side
        self._clusters = None

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
        centroids = self._clusters.cluster_centers_
        return centroids

    def _griddify(self, X):
        xv, yv = np.meshgrid(
            np.linspace(0, 1, self._side), np.linspace(0, 1, self._side)
        )
        self.grid = np.dstack((xv, yv)).reshape(-1, 2)
        cost = cdist(self.grid, X, GRIDDIFY_METRIC)
        cost = cost * (1000000 / cost.max())
        cost = cost.astype(int)
        min_cost, row_assigns, col_assigns = lap.lapjv(cost)
        self._grid_jv = self.grid[col_assigns]
        self._min_cost = min_cost
        self._row_assigns = row_assigns
        self._col_assigns = col_assigns

    def _grid_coordinates_as_integers(self, X):
        d = dict((v, i) for i, v in enumerate(np.linspace(0, 1, self._side)))
        Xt = np.zeros(X.shape, dtype=int)
        for i in range(X.shape[0]):
            Xt[i] = [d[X[i, 0]], d[X[i, 1]]]
        return Xt

    def fit(self, X):
        assert self._is_cloud(X)
        self._side = self._find_side(X)
        self._do_cluster = self._needs_downsampling(X)
        if self._do_cluster:
            X = self._downsample_with_clustering(X)
        self._griddify(X)
        self.nearest_neighbors = NearestNeighbors(n_neighbors=1)
        self.nearest_neighbors.fit(X)

    def transform(self, X, as_integers=False):
        assert self._is_cloud(X)
        idxs = self.nearest_neighbors.kneighbors(X, return_distance=False)[:, 0]
        X_grid = np.zeros((X.shape[0], 2))
        for i, idx in enumerate(idxs):
            X_grid[i] = self._grid_jv[idx]
        if as_integers:
            return self._grid_coordinates_as_integers(X_grid)
        else:
            return X_grid

    def get_mappings(self, X):
        X = self.transform(X, as_integers=True)
        return (X, self._side)
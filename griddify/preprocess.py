import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold


MAX_NA = 0.2
SPARSE_ZEROS = 0.25

MAX_DATATYPER_N = 10000


class ColumnTyper(object):

    def __init__(self, data):
        self.data = list(data)


class DataTyper(object):

    def __init__(self, data):
        self.X = np.array(data)[:MAX_DATATYPER_N]

    def _is_two_column_sparse(self):
        pass

    def _assess_sparseness_homogeneous(self, X):
        X = X.ravel()
        n_zeros = np.sum(X == 0)
        if n_zeros / len(X) > SPARSE_ZEROS:
            return True
        else:
            return False

    def _assess_sparseness_heterogeneous(self, X):
        pass

    def is_homogeneous(self):
        pass

    def is_sparse(self):
        pass

    def is_binary(self):
        pass

    def is_counts(self):
        pass



class NanFilter(object):
    def __init__(self):
        pass

    def fit(self, X):
        max_na = int((1 - MAX_NA) * X.shape[0])
        idxs = []
        for j in range(X.shape[1]):
            c = np.sum(np.isnan(X[:, j]))
            if c > max_na:
                continue
            else:
                idxs += [j]
        self.col_idxs = idxs

    def transform(self, X):
        return X[:, self.col_idxs]


class Imputer(object):
    def __init__(self):
        self._fallback = 0

    def fit(self, X):
        ms = []
        for j in range(X.shape[1]):
            vals = X[:, j]
            mask = ~np.isnan(vals)
            vals = vals[mask]
            if len(vals) == 0:
                m = self._fallback
            else:
                m = np.median(vals)
            ms += [m]
        self.impute_values = np.array(ms)

    def transform(self, X):
        for j in range(X.shape[1]):
            mask = np.isnan(X[:, j])
            X[mask, j] = self.impute_values[j]
        return X


class VarianceFilter(object):
    def __init__(self):
        pass

    def fit(self, X):
        self.sel = VarianceThreshold()
        self.sel.fit(X)
        self.col_idxs = list(self.sel.transform([[i for i in range(X.shape[1])]])[0])

    def transform(self, X):
        return self.sel.transform(X)


class ColwiseDenseScaler(object):
    def __init__(self):
        self.abs_limit = 10
        self.skip = False

    def set_skip(self):
        self.skip = True

    def fit(self, X):
        if self.skip:
            return
        self.scaler = RobustScaler()
        self.scaler.fit(X)

    def transform(self, X):
        if self.skip:
            return X
        X = self.scaler.transform(X)
        X = np.clip(X, -self.abs_limit, self.abs_limit)
        return X


# TODO
class ColwiseSparseScaler(object):
    def __init__(self):
        pass

    def set_skip(self):
        self.skip = True


# TODO
class ColwiseNormalizer(object):
    def __init__(self):
        pass




class Preprocessing(object):
    def __init__(self, scale=True):
        self.scale = scale
        self.nan_filter = NanFilter()
        self.imputer = Imputer()
        self.variance_filter = VarianceFilter()

    def fit(self, data):
        self._columns = list(data.columns)
        X = np.array(data)
        self.nan_filter.fit(X)
        X = self.nan_filter.transform(X)
        self._columns = [self._columns[i] for i in self.nan_filter.col_idxs]
        self.imputer.fit(X)
        X = self.imputer.transform(X)
        if self.scale:
            self.scaler = RobustScaler()
            self.scaler.fit(X)
            X = self.scaler.transform(X)
        else:
            self.scaler = None
        self.variance_filter.fit(X)
        X = self.variance_filter.transform(X)
        self._columns = [self._columns[i] for i in self.variance_filter.col_idxs]

    def transform(self, data):
        X = np.array(data)
        X = self.nan_filter.transform(X)
        X = self.imputer.transform(X)
        if self.scale:
            X = self.scaler.transform(X)
        X = self.variance_filter.transform(X)
        return pd.DataFrame(X, columns=self._columns)

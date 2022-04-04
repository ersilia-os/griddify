import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold


MAX_NA = 0.2


class NanFilter(object):
    def __init__(self):
        self._name = "nan_filter"

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


class Scaler(object):
    def __init__(self):
        self._name = "scaler"
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


class Imputer(object):
    def __init__(self):
        self._name = "imputer"
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
        self._name = "variance_filter"

    def fit(self, X):
        self.sel = VarianceThreshold()
        self.sel.fit(X)
        self.col_idxs = list(self.sel.transform([[i for i in range(X.shape[1])]])[0])

    def transform(self, X):
        return self.sel.transform(X)


class Preprocessing(object):
    def __init__(self):
        self.nan_filter = NanFilter()
        self.scaler = Scaler()
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
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.variance_filter.fit(X)
        X = self.variance_filter.transform(X)
        self._columns = [self._columns[i] for i in self.variance_filter.col_idxs]

    def transform(self, data):
        X = np.array(data)
        X = self.nan_filter.transform(X)
        X = self.imputer.transform(X)
        X = self.scaler.transform(X)
        X = self.variance_filter.transform(X)
        return pd.DataFrame(X, columns=self._columns)

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist


class FeatureDistances(object):

    def __init__(self, metric="cosine", max_n=10000):
        self.metric = metric
        self.max_n = max_n

    def calculate(self, data):
        D = squareform(pdist(np.array(data).T[:self.max_n], metric=self.metric))
        D[np.isnan(D)] = np.nanmax(D)
        return pd.DataFrame(D, columns=list(data.columns))
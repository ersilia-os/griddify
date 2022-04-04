import numpy as np
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
# from tqdm import tqdm
# from .utils import Mixed_KSG


class FeatureDistances(object):
    def __init__(self, metric="cosine", max_n=10000):
        self.metric = metric
        self.max_n = max_n

    def calculate(self, data):
        data = pd.DataFrame(data)
        D = squareform(pdist(np.array(data).T[:,:self.max_n], metric=self.metric))
        D[np.isnan(D)] = np.nanmax(D)
        return pd.DataFrame(D, columns=list(data.columns))

"""
class FeatureMutualInformation(object):
    def __init__(self, max_n=100000):
        self.max_n = max_n

    def calculate(self, data):
        data = pd.DataFrame(data)
        X = np.array(data)[:self.max_n]
        M = np.zeros((X.shape[1], X.shape[1]))
        for i in tqdm(range(X.shape[1])):
            for j in range(i, X.shape[1]):
                x = X[:,i]
                y = X[:,j]
                mi = Mixed_KSG(x,y,k=5)
                M[i,j]=mi
                M[j,i]=mi
        return pd.DataFrame(M, columns=list(data.columns))
"""
from .preprocess import Preprocessing
from .features import FeatureDistances
from .cloud import Tabular2Cloud
from .grid import Cloud2Grid
from .rearrange import Flat2Grid


class Griddify(object):
    def __init__(self, preprocess=True, metric="cosine", max_side=128):
        self._preprocess = preprocess
        self._metric = metric
        self._max_side = max_side

    def fit(self, data):
        if self._preprocess:
            self.pp = Preprocessing()
            self.pp.fit(data)
            data = self.pp.transform(data)
        else:
            self.pp = None
        fd = FeatureDistances(metric=self._metric).calculate(data)
        self.tc = Tabular2Cloud()
        self.tc.fit(fd)
        Xc = self.tc.transform(fd)
        self.cg = Cloud2Grid(max_side=self._max_side)
        self.cg.fit(Xc)
        self.mappings, self.side = self.cg.get_mappings(Xc)

    def transform(self, data):
        if self._preprocess:
            data = self.pp.transform(data)
        fg = Flat2Grid(self.mappings, self.side)
        Xi = fg.transform(data)
        return Xi

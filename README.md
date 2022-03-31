# Griddify
Griddify high-dimensional tabular data for easy visualization and deep learning

## Installation

```
git clone https://github.com/ersilia-os/griddify.git
cd griddify
pip install -e .
```

## Step by step

### Create a 2D cloud of your data features

First, process your data so that it contains no empty values, it is scaled, etc.
```python
from griddify import Preprocessing

pp = Preprocessing()
pp.fit(data)
data = pp.transform(data)
```

You can now obtain a 2D cloud of your data features. At the moment, UMAP is used.
```python
from griddify import Tabular2Cloud

tc = Tabular2Cloud()
tc.fit(data)
Xc = tc.transform(data)
```

If you want, you can directly pass a distance matrix as tabular data.
```python
distances = pdist(data)
pp.fit(distances)
Xc = tc.transform(distances)
```


### Rearrange the 2D cloud on a grid

```python
from griddify import Cloud2Grid

g = Cloud2Grid()
g.fit(Xc)
Xg = g.transform(Xc)
```

### Rearrange your samples into grids

```python
from griddify import 

```

## Pipeline

```python
from griddify import Griddify

gf = Griddify()
gf.fit(data)
grids = gf.transform(data)

```
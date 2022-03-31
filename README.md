# Griddify
Griddify high-dimensional tabular data for easy visualization and image-based deep learning

## Installation

```
git clone https://github.com/ersilia-os/griddify.git
cd griddify
pip install -e .
```

## Step by step

### Get your multidimensional dataset and preprocess it

We will use a dataset of 200 physicochemical descriptors for about 10k compounds. You can get this data with the following command.

```python
from griddify import datasets

data = datasets.get_compound_descriptors()
```

It is important that you preprocess your data (impute missing values, normalize, etc.). We provide functionality to do so.

```python
from griddify import Preprocessing

pp = Preprocessing()
pp.fit(data)
data = pp.transform(data)
```

### Create a 2D cloud of your data features

Now you must project features on a 2D spaces. There are two options to doing so:

#### Option 1: Transpose data



#### Option 2: Distance matrix

```python
from griddify import 
```

You can now obtain a 2D cloud of your data features. At the moment, [UMAP](https://umap-learn.readthedocs.io/en/latest/) is used.
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
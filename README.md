# Griddify
Griddify high-dimensional tabular data for easy visualization and image-based deep learning

## Installation

```
git clone https://github.com/ersilia-os/griddify.git
cd griddify
pip install -e .
```

## One-step usage

```python
from griddify import Griddify

gf = Griddify()
gf.fit(data)
X = gf.transform(data)
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

Start by calculating correlations between features.

```python
from griddify import FeatureDistances

fd = FeatureDistances(metric="cosine").calculate(data)
```

You can now obtain a 2D cloud of your data features. By default, [UMAP](https://umap-learn.readthedocs.io/en/latest/) is used.

```python
from griddify import Tabular2Cloud

tc = Tabular2Cloud()
tc.fit(fd)
Xc = tc.transform(fd)
```

It is always good to inspect the resulting projection. The cloud contains as many points as features in your dataset.

```python
from griddify.plots import cloud_plot

cloud_plot(Xc)
```

### Rearrange the 2D cloud on a grid

Distribute points in the cloud to a grid using a linear assignment algorithm.

```python
from griddify import Cloud2Grid

cg = Cloud2Grid()
cg.fit(Xc)
Xg = cg.transform(Xc)
```

You can check the rearrangement with an arrows plot.
```python
from griddify.plots import arrows_plot

arrows_plot(Xc, Xg)
```

### Rearrange your flat data points into grids

Let's go back to the original tabular data. We want to transform the input data, where each data sample is represented by an one-dimensional array, into an output data where each sample is represented by an image (i.e. a two-dimensional grid). Please be sure to use normalized data.

```python
from griddify import Flat2Grid

fg = Flat2Grid(cloud2grid=cg)
fg.fit(data)
X = fg.transform(data)
```

Explore one sample.

```python
from griddify.plots import grid_plot

grid_plot(X[0])
```

## Learn more

The [Ersilia Open Source Initiative](https://ersilia.io) is on a mission to strenghten research capacity in low income countries. Please reach out to us if you want to contribute: [hello@ersilia.io]()
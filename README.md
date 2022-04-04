# Griddify
Redistribute tabular data into a grid for easy visualization and image-based deep learning

## Installation

```
git clone https://github.com/ersilia-os/griddify.git
cd griddify
pip install -e .
```

## Step by step

### Get a multidimensional dataset and preprocess it

In this example, we will use a dataset of 200 physicochemical [descriptors](https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html) calculated about 10k compounds. You can get this data with the following command.

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

### Create a 2D cloud of data features

Start by calculating distances between features.

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

It is always good to inspect the resulting projection. The cloud contains as many points as features exist in your dataset.

```python
from griddify.plots import cloud_plot

cloud_plot(Xc)
```

### Rearrange the 2D cloud onto a grid

Distribute points in the cloud to a grid using a [linear assignment](https://github.com/gatagat/lap) algorithm.

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

To continue with the next steps, it is actually more convenient to get mappings as integers. The following method gives you the size of the grid as well.

```python
mappings, side = cg.get_mappings(Xc)
```

### Rearrange your flat data points into grids

Let's go back to the original tabular data. We want to transform the input data, where each data sample is represented with a one-dimensional array, into an output data where each sample is represented with an image (i.e. a two-dimensional grid). Please ensure that data are normalize or scaled.

```python
from griddify import Flat2Grid

fg = Flat2Grid(mappings, side)
Xi = fg.transform(data)
```

Explore one sample.

```python
from griddify.plots import grid_plot

grid_plot(Xi[0])
```

## Full pipeline

You can run the full pipeline described above with only a few lines of code.

```python
from griddify import datasets
from griddify import Griddify

data = datasets.get_compound_descriptors()

gf = Griddify(preprocess=True)
gf.fit(data)
Xi = gf.transform(data)
```

## Learn more

The [Ersilia Open Source Initiative](https://ersilia.io) is on a mission to strenghten research capacity in low income countries. Please reach out to us if you want to contribute: [hello@ersilia.io]()
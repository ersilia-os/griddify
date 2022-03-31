# Tab2Grid
Griddify high-dimensional tabular data for easy visualization and deep learning

## Installation

```
git clone https://github.com/ersilia-os/tab2grid.git
cd tab2grid
pip install -e .
```

### Usage

```python
from tab2grid import Cloud2Grid

g = Cloud2Grid()
g.fit(Xc)
Xg = g.transform(Xc)
```
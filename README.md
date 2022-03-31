# Griddify
Griddify high-dimensional tabular data for easy visualization and deep learning

## Installation

```
git clone https://github.com/ersilia-os/griddify.git
cd griddify
pip install -e .
```

### Usage

```python
from griddify import Cloud2Grid

g = Cloud2Grid()
g.fit(Xc)
Xg = g.transform(Xc)
```
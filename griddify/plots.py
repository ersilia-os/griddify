from scipy.spatial.distance import euclidean
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

def arrows_plot(X_cloud, X_grid, ax=None, capping_distance=0.5):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    cmap = cm.get_cmap("viridis")
    dists = []
    for start, end in zip(X_cloud, X_grid):
        dists += [euclidean(start, end)]
    norm = mpl.colors.Normalize(vmin=0, vmax=capping_distance)
    values = [norm(x) for x in dists]
    colors = [cmap(x) for x in values]
    i = 0
    for start, end in zip(X_cloud, X_grid):
        color = colors[i]
        ax.arrow(
            start[0],
            start[1],
            end[0] - start[0],
            end[1] - start[1],
            head_length=0.01,
            head_width=0.01,
            color=color,
        )
        i += 1
    return ax


def cloud_plot(X, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.scatter(X[:, 0], X[:, 1])  # TODO
    return ax


def grid_plot(X, ax=None, s=300, cmap="Spectral", vlim=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    cmap = cm.get_cmap(cmap)
    if vlim is None:
        vmin = np.min(X)
        vmax = np.max(X)
    else:
        vmin = vlim[0]
        vmax = vlim[1]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    x = []
    y = []
    z = []
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x += [i]
            y += [j]
            z += [X[i, j]]
    colors = [cmap(norm(z_)) for z_ in z]
    for x_, y_, c_ in zip(x, y, colors):
        rect = Rectangle(xy=(x_, y_), width=1, height=1, facecolor=c_)
        ax.add_patch(rect)
    ax.set_xlim(0, X.shape[0])
    ax.set_ylim(0, X.shape[1])
    return ax

from scipy.spatial.distance import euclidean
import matplotlib as mpl
from matplotlib import cm


def arrows_plot(X_cloud, X_grid, ax, capping_distance=0.5):
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
        ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_length=0.01, head_width=0.01, color=color)
        i += 1
    return ax


def cloud_plot(X, ax):
    ax.scatter(X[:,0], X[:,1])
    return ax


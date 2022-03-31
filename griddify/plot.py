from scipy.spatial.distance import euclidean


def arrows_plot(X_cloud, X_grid, ax):
    dists = []
    for start, end in zip(X_cloud, X_grid):
        dists += [euclidean(start, end)]
    for start, end in zip(X_cloud, X_grid):
        ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_length=0.01, head_width=0.01)
    return ax


def cloud_plot(X, ax):
    ax.scatter(X[:,0], X[:,1])
    return ax


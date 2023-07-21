import numpy as np
import pandas as pd
import matplotlib as mpl


def matshow45(ax, series, radius=None, cmap=None, norm=None):

    """
    fig, ax = plt.subplots()
    plotdata = pd.DataFrame(
        np.random.rand(10, 10),
        index=np.arange(10),
        columns=np.arange(10),
    ).stack()
    ax.set_aspect(1)
    matshow45(ax, plotdata)
    """
    offsets = []
    colors = []

    assert len(series.index.names) == 2
    x = series.index.get_level_values(0)
    y = series.index.get_level_values(1)

    if radius is None:
        radius = np.diff(y)[0] / 2

    centerxs = x + (y - x) / 2
    centerys = (y - x) / 2

    xlim = [
        centerxs.min() - radius,
        centerxs.max() + radius,
    ]
    ax.set_xlim(xlim)

    ylim = [
        centerys.unique().min(),
        centerys.unique().max(),
    ]
    ax.set_ylim(ylim)

    if norm is None:
        norm = mpl.colors.Normalize(vmin=series.min(), vmax=series.max())

    if cmap is None:
        cmap = mpl.cm.get_cmap()

    vertices = []

    for centerx, centery, value in zip(centerxs, centerys, series.values):
        center = np.array([centerx, centery])
        offsets.append(center)
        colors.append(cmap(norm(value)))

        vertices.append(
            [
                center + np.array([radius * 1.1, 0]),
                center + np.array([0, radius * 1.1]),
                center + np.array([-radius * 1.1, 0]),
                center + np.array([0, -radius * 1.1]),
            ]
        )
    vertices = np.array(vertices)
    collection = mpl.collections.PolyCollection(
        vertices,
        ec=None,
        lw=0,
        fc=colors,
    )

    ax.add_collection(collection)

    for x in [xlim[1]]:
        x2 = x
        x1 = x2 + (xlim[0] - x2) / 2
        y2 = 0
        y1 = x2 - x1

        if True:
            color = "black"
            lw = 0.8
            zorder = 10
        elif False:
            color = "#eee"
            lw = 0.5
            zorder = -1
        ax.plot(
            [x1 + radius, x2 + radius],
            [y1, y2],
            zorder=zorder,
            color=color,
            lw=lw,
        )

        x1, x2 = (xlim[1] - xlim[0]) / 2 - x2, (xlim[1] - xlim[0]) / 2 - x1
        ax.plot(
            [-x1 - radius, -x2 - radius],
            [y1, y2],
            zorder=zorder,
            color=color,
            lw=lw,
        )
    ax.axis("off")

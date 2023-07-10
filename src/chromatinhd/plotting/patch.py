import pandas as pd
import numpy as np


phases = pd.DataFrame(
    {"phase": ["train", "validation"], "color": ["#888888", "tomato"]}
).set_index("phase")


colors = ["#0074D9", "#FF4136", "#FF851B", "#2ECC40", "#39CCCC", "#85144b"]


def replace_patch(ax, patch, points=10, ha="center", va="center"):
    """
    Replaces a patch, often an axis label, with a new rectangular axis according to figure size "points". Eample usage:


    fig, ax = plt.subplots()
    ax.set_title("100%, (0.5,1-0.3,.3,.3)")
    x.plot([0, 2], [0, 2])

    for l in ax.get_xticklabels():
        ax1 = replace_patch(ax, l)
        ax1.plot([0, 1], [0, 1])
    """
    fig = ax.get_figure()

    fig.draw_without_rendering()

    w, h = fig.transFigure.inverted().transform([[1, 1]]).ravel() * points
    bbox = fig.transFigure.inverted().transform(patch.get_window_extent())
    dw = bbox[1, 0] - bbox[0, 0]
    dh = bbox[1, 1] - bbox[0, 1]
    x = bbox[0, 0] + dw / 2
    y = bbox[0, 1] + dh / 2

    if ha == "left":
        x += 0
    elif ha == "right":
        x -= w - dw / 2
    elif ha == "center":
        x -= w / 2

    if va == "bottom":
        y += 0
    elif va == "top":
        y -= h - dh / 2
    elif va == "center":
        y -= h / 2

    ax1 = fig.add_axes([x, y, w, h])
    # ax1.axis("off")

    return ax1

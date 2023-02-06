import pandas as pd
import seaborn as sns
import scipy.stats
import numpy as np

types_info = pd.DataFrame(
    [
        ["peak", "Peak"],
        ["flank", "Flank"],
        ["ridge", "Ridge"],
        ["chain", "Chain"],
        ["canyon", "Canyon"],
        ["hill", "Hill"],
    ],
    columns=["type", "label"],
).set_index("type")
types_info["color"] = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
][: len(types_info)]


def plot_peak(ax):
    color = types_info.loc["peak", "color"]
    plotdata_mean = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    plotdata_mean["y"] = scipy.stats.norm(0.5, 0.15).pdf(plotdata_mean.x)
    plotdata_oi = pd.DataFrame({"x": plotdata_mean["x"]})
    plotdata_oi["y"] = plotdata_mean["y"] * 3.0

    # rescale in 0-1
    scale = plotdata_oi["y"].max()
    plotdata_oi["y"] = plotdata_oi["y"] / scale
    plotdata_mean["y"] = plotdata_mean["y"] / scale

    ax.plot(plotdata_mean["x"], plotdata_mean["y"], color="#333", dashes=(1, 1), lw=1)
    ax.plot(plotdata_oi["x"], plotdata_oi["y"], color=color, lw=1)
    ax.fill_between(
        plotdata_oi["x"],
        plotdata_mean["y"],
        plotdata_oi["y"],
        fc=color,
        alpha=0.5,
        lw=1,
    )


def plot_canyon(ax):
    color = types_info.loc["canyon", "color"]

    plotdata_mean = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    plotdata_mean["y"] = scipy.stats.norm(0.25, 0.15).pdf(
        plotdata_mean.x
    ) * 2 + scipy.stats.norm(0.75, 0.15).pdf(plotdata_mean.x)

    plotdata_oi = pd.DataFrame({"x": plotdata_mean["x"]})
    plotdata_oi["fc"] = np.exp(
        (np.log(4) / (1 + np.exp(-20 * (plotdata_mean["x"] - 0.5))))
        - (np.log(2) / (1 + np.exp(-20 * (plotdata_mean["x"] + 0.5))))
    )
    plotdata_oi["y"] = plotdata_mean["y"] * plotdata_oi["fc"]
    plotdata_oi["positive"] = plotdata_mean["positive"] = (
        plotdata_oi["y"] > plotdata_mean["y"]
    )

    # rescale in 0-1
    scale = plotdata_oi["y"].max()
    plotdata_oi["y"] = plotdata_oi["y"] / scale
    plotdata_mean["y"] = plotdata_mean["y"] / scale

    ax.plot(plotdata_mean["x"], plotdata_mean["y"], color="#333", dashes=(1, 1), lw=1)
    ax.plot(plotdata_oi["x"], plotdata_oi["y"], color="#333", lw=1)
    ax.fill_between(
        plotdata_oi.query("~positive")["x"],
        plotdata_mean.query("~positive")["y"],
        plotdata_oi.query("~positive")["y"],
        # fc=color,
        fc="#333",
        alpha=0.2,
        lw=1,
    )
    ax.fill_between(
        plotdata_oi.query("positive")["x"],
        plotdata_mean.query("positive")["y"],
        plotdata_oi.query("positive")["y"],
        fc=color,
        alpha=0.5,
        lw=1,
    )
    # plotdata_mean_max = plotdata_mean.loc[plotdata_mean["y"].idxmax()]
    # plotdata_oi_min = plotdata_oi.loc[plotdata_oi["x"] == plotdata_mean_max["x"]].iloc[
    #     0
    # ]
    # ax.arrow(
    #     plotdata_mean_max["x"],
    #     plotdata_mean_max["y"],
    #     0,
    #     plotdata_oi_min["y"] - plotdata_mean_max["y"],
    #     color="black",
    #     lw=0.5,
    # )
    ax.plot(plotdata_oi["x"], plotdata_oi["y"], color=color, lw=1)


def plot_flank(ax):
    color = types_info.loc["flank", "color"]

    plotdata_mean = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    plotdata_mean["y"] = scipy.stats.norm(0.35, 0.2).pdf(
        plotdata_mean.x
    ) + 0.3 * scipy.stats.uniform(0.0, 1.0).pdf(
        plotdata_mean.x
    )  # scipy.stats.norm(0.5, 2.0).pdf(plotdata_mean.x)

    plotdata_oi = pd.DataFrame({"x": plotdata_mean["x"]})
    plotdata_oi["fc"] = 1 + (2 / (1 + np.exp(-20 * (plotdata_mean["x"] - 0.65))))
    plotdata_oi["y"] = plotdata_mean["y"] * plotdata_oi["fc"]

    # rescale in 0-1
    scale = plotdata_oi["y"].max()
    plotdata_oi["y"] = plotdata_oi["y"] / scale
    plotdata_mean["y"] = plotdata_mean["y"] / scale

    ax.plot(plotdata_mean["x"], plotdata_mean["y"], color="#333", dashes=(1, 1), lw=1)
    ax.plot(plotdata_oi["x"], plotdata_oi["y"], color=color, lw=1)
    ax.fill_between(
        plotdata_oi["x"],
        plotdata_mean["y"],
        plotdata_oi["y"],
        fc=color,
        alpha=0.5,
        lw=1,
    )


def plot_ridge(ax):
    color = types_info.loc["ridge", "color"]

    plotdata_mean = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    rg = np.random.RandomState(6)
    plotdata_mean["y"] = np.sum(
        [
            rg.rand() * scipy.stats.norm(i / 10, 0.1).pdf(plotdata_mean.x)
            for i in range(10)
        ],
        0,
    )
    plotdata_oi = pd.DataFrame({"x": plotdata_mean["x"]})
    plotdata_oi["fc"] = np.exp(
        (np.log(4) / (1 + np.exp(-20 * (plotdata_mean["x"] - 0.1))))
        - (np.log(4) / (1 + np.exp(-20 * (plotdata_mean["x"] - 0.9))))
    )
    plotdata_oi["y"] = plotdata_mean["y"] * plotdata_oi["fc"]

    # rescale in 0-1
    scale = plotdata_oi["y"].max()
    plotdata_oi["y"] = plotdata_oi["y"] / scale
    plotdata_mean["y"] = plotdata_mean["y"] / scale

    ax.plot(plotdata_mean["x"], plotdata_mean["y"], color="#333", dashes=(1, 1), lw=1)
    ax.plot(plotdata_oi["x"], plotdata_oi["y"], color=color, lw=1)
    ax.fill_between(
        plotdata_oi["x"],
        plotdata_mean["y"],
        plotdata_oi["y"],
        fc=color,
        alpha=0.5,
        lw=1,
    )


def plot_chain(ax):
    color = types_info.loc["chain", "color"]

    plotdata_mean = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    plotdata_mean["y"] = scipy.stats.norm(0.25, 0.15).pdf(
        plotdata_mean.x
    ) + scipy.stats.norm(0.75, 0.15).pdf(plotdata_mean.x)
    plotdata_oi = pd.DataFrame({"x": plotdata_mean["x"]})
    plotdata_oi["fc"] = 2.0
    plotdata_oi["y"] = plotdata_mean["y"] * plotdata_oi["fc"]

    # rescale in 0-1
    scale = plotdata_oi["y"].max()
    plotdata_oi["y"] = plotdata_oi["y"] / scale
    plotdata_mean["y"] = plotdata_mean["y"] / scale

    ax.plot(plotdata_mean["x"], plotdata_mean["y"], color="#333", dashes=(1, 1), lw=1)
    ax.plot(plotdata_oi["x"], plotdata_oi["y"], color=color, lw=1)
    ax.fill_between(
        plotdata_oi["x"],
        plotdata_mean["y"],
        plotdata_oi["y"],
        fc=color,
        alpha=0.5,
        lw=1,
    )


def plot_hill(ax):
    color = types_info.loc["hill", "color"]

    plotdata_mean = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    plotdata_mean["y"] = scipy.stats.norm(0.25, 0.15).pdf(
        plotdata_mean.x
    ) * 2 + scipy.stats.norm(0.75, 0.15).pdf(plotdata_mean.x)

    plotdata_oi = pd.DataFrame({"x": plotdata_mean["x"]})
    plotdata_oi["fc"] = 1 + (1 / (1 + np.exp(-10 * (plotdata_mean["x"] - 0.65))))

    plotdata_oi["y"] = plotdata_mean["y"] * plotdata_oi["fc"]

    # rescale in 0-1
    scale = plotdata_oi["y"].max()
    plotdata_oi["y"] = plotdata_oi["y"] / scale
    plotdata_mean["y"] = plotdata_mean["y"] / scale

    ax.plot(plotdata_mean["x"], plotdata_mean["y"], color="#333", dashes=(1, 1), lw=1)
    ax.plot(plotdata_oi["x"], plotdata_oi["y"], color="#333", lw=1)
    ax.fill_between(
        plotdata_oi["x"],
        plotdata_mean["y"],
        plotdata_oi["y"],
        fc=color,
        alpha=0.2,
        lw=1,
    )
    ax.plot(plotdata_oi["x"], plotdata_oi["y"], color=color, lw=1)


def plot_type(ax, type):
    type_funcs = {
        "peak": plot_peak,
        "canyon": plot_canyon,
        "flank": plot_flank,
        "chain": plot_chain,
        "ridge": plot_ridge,
        "hill": plot_hill,
    }
    if type in type_funcs:
        type_funcs[type](ax)
        ax.set_xticks([])
        ax.set_yticks([])


import chromatinhd.plotting


def label_axis(ax, axis):
    for l in axis.get_ticklabels():
        ax_tick = chromatinhd.plotting.replace_patch(
            ax,
            l,
            points=20,
            va="top" if axis.axis_name == "x" else "center",
            ha="right" if axis.axis_name == "y" else "center",
        )
        ax_tick.axis("off")
        plot_type(ax_tick, l.get_text())
    axis.set_ticklabels(["  "] * len(axis.get_ticklabels()), fontsize=10, rotation=0)

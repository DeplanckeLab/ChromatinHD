import chromatinhd.grid
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd


class Predictive(chromatinhd.grid.Panel):
    def __init__(self, plotdata, window, width):
        super().__init__((width, 0.5))

        ax = self.ax
        ax.set_xlim(*window)

        ax.plot(
            plotdata["position"],
            plotdata["deltacor"],
            color="#333",
            lw=1,
        )

        # ax.set_ylim(0, 1)
        ax.set_ylabel("$\Delta$ cor", rotation=0, ha="right", va="center")
        ax.set_xticks([])
        ax.invert_yaxis()
        ax.set_ylim(0)

        ax2 = self.add_twinx()
        ax2.plot(
            plotdata["position"],
            plotdata["lost"],
            color="tomato",
            # color="#333",
            lw=1,
        )
        ax2.set_xlim(ax.get_xlim())
        ax2.set_ylabel(
            "# fragments\nper 1kb\nper 1k cells",
            rotation=0,
            ha="left",
            va="center",
            color="tomato",
        )
        ax2.tick_params(axis="y", colors="tomato")
        ax2.set_ylim(0)

        # vline at tss
        ax.axvline(0, color="#888888", lw=0.5, zorder=-1, dashes=(2, 2))

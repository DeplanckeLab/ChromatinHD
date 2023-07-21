import chromatinhd.grid
import matplotlib as mpl
from chromatinhd.grid.broken import Broken, Panel
import numpy as np


class Effect(chromatinhd.grid.Panel):
    def __init__(self, plotdata, window, width, show_accessibility=False):
        super().__init__((width, 0.5))

        ax = self.ax
        ax.set_xlim(*window)

        ax.plot(
            plotdata["position"],
            plotdata["effect"],
            color="#333",
            lw=1,
        )
        ax.fill_between(
            plotdata["position"],
            plotdata["effect"],
            0,
            color="#333",
            alpha=0.2,
            lw=0,
        )

        ax.set_ylabel(
            "Effect",
            rotation=0,
            ha="right",
            va="center",
        )

        ax.set_xticks([])
        ax.invert_yaxis()
        ymax = plotdata["effect"].abs().max()
        ax.set_ylim(-ymax, ymax)

        # change vertical alignment of last y tick to bottom
        ax.set_yticks([0, ax.get_ylim()[1]])
        ax.get_yticklabels()[-1].set_verticalalignment("top")
        ax.get_yticklabels()[0].set_verticalalignment("bottom")

        # vline at tss
        ax.axvline(0, color="#888888", lw=0.5, zorder=-1, dashes=(2, 2))

    @classmethod
    def from_genemultiwindow(
        cls, genemultiwindow, gene, width, show_accessibility=False
    ):
        plotdata = genemultiwindow.get_plotdata(gene).reset_index()
        window = np.array([plotdata["position"].min(), plotdata["position"].max()])
        return cls(plotdata, window, width, show_accessibility=show_accessibility)

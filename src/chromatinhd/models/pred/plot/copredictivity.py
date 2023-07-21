import chromatinhd.grid
import chromatinhd.plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


class Copredictivity(chromatinhd.grid.Panel):
    def __init__(self, plotdata, width):
        super().__init__((width, width / 2))

        norm = mpl.colors.CenteredNorm(0, np.abs(plotdata["cor"]).max())
        cmap = mpl.cm.RdBu_r

        chromatinhd.plot.matshow45(
            self.ax,
            plotdata.set_index(["window_mid1", "window_mid2"])["cor"],
            cmap=cmap,
            norm=norm,
            radius=50,
        )
        self.ax.invert_yaxis()

        panel_copredictivity_legend = self.add_inset(
            chromatinhd.grid.Panel((0.05, 0.8)), pos=(0.0, 0.0), offset=(0.0, 0.2)
        )
        plt.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=panel_copredictivity_legend.ax,
            orientation="vertical",
        )
        panel_copredictivity_legend.ax.set_ylabel(
            "Co-predictivity\n(cor $\\Delta$cor)",
            rotation=0,
            ha="right",
            va="center",
        )
        panel_copredictivity_legend.ax.yaxis.set_ticks_position("left")
        panel_copredictivity_legend.ax.yaxis.set_label_position("left")

    @classmethod
    def from_genepairwindow(cls, genepairwindow, gene, width):
        plotdata = genepairwindow.get_plotdata(gene).reset_index()
        return cls(plotdata, width)

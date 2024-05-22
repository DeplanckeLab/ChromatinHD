import chromatinhd.grid
import chromatinhd.plot
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools


class Copredictivity(chromatinhd.grid.Panel):
    """
    Plot co-predictivity of a gene.
    """

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
    def from_regionpairwindow(cls, regionpairwindow, gene, width):
        """
        Plot co-predictivity of a gene using a RegionPairWindow object.
        """
        plotdata = regionpairwindow.get_plotdata(gene).reset_index()
        return cls(plotdata, width)


class CopredictivityBroken(chromatinhd.grid.Panel):
    """
    Plot co-predictivity for different regions
    """

    def __init__(self, plotdata, breaking):
        pass

    @classmethod
    def from_regionpairwindow(cls, regionpairwindow, gene, breaking):
        plotdata_windows = regionpairwindow.scores[gene].mean("fold").to_dataframe()
        plotdata_interaction = regionpairwindow.interaction[gene].mean("fold").to_pandas().unstack().to_frame("cor")

        plotdata = plotdata_interaction.copy()

        # make plotdata, making sure we have all window combinations, otherwise nan
        plotdata = (
            pd.DataFrame(itertools.combinations(windows.index, 2), columns=["window1", "window2"])
            .set_index(["window1", "window2"])
            .join(plotdata_interaction)
        )
        plotdata.loc[np.isnan(plotdata["cor"]), "cor"] = 0.0
        plotdata["dist"] = (
            windows.loc[plotdata.index.get_level_values("window2"), "window_mid"].values
            - windows.loc[plotdata.index.get_level_values("window1"), "window_mid"].values
        )

        transform = chd.grid.broken.TransformBroken(breaking)
        plotdata["window1_broken"] = transform(
            windows.loc[plotdata.index.get_level_values("window1"), "window_mid"].values
        )
        plotdata["window2_broken"] = transform(
            windows.loc[plotdata.index.get_level_values("window2"), "window_mid"].values
        )

        plotdata = plotdata.loc[~pd.isnull(plotdata["window1_broken"]) & ~pd.isnull(plotdata["window2_broken"])]

        plotdata.loc[plotdata["dist"] < 1000, "cor"] = 0.0

import polyptich.grid
import matplotlib as mpl
from polyptich.grid.broken import Broken
import numpy as np


class Predictivity(polyptich.grid.Panel):
    """
    Plot predictivity of a gene.
    """

    def __init__(
        self,
        plotdata,
        window,
        width,
        show_accessibility=False,
        color_by_effect=True,
        limit=-0.05,
        label_y=True,
        height=0.5,
    ):
        super().__init__((width, height))

        if "position" not in plotdata.columns:
            plotdata = plotdata.reset_index()

        plotdata["effect_sign"] = np.sign(plotdata["effect"])
        plotdata["segment"] = plotdata["effect_sign"].diff().ne(0).cumsum()

        ax = self.ax
        ax.set_xlim(*window)

        for segment, segment_data in plotdata.groupby("segment"):
            if color_by_effect:
                color = "tomato" if segment_data["effect"].iloc[0] > 0 else "#0074D9"
            else:
                color = "#333"
            ax.plot(
                segment_data["position"],
                segment_data["deltacor"],
                lw=1,
                color=color,
            )
            ax.fill_between(
                segment_data["position"],
                segment_data["deltacor"],
                0,
                alpha=0.2,
                lw=0,
                color=color,
            )

        # ax.plot(
        #     plotdata["position"],
        #     plotdata["deltacor"],
        #     color="#333",
        #     lw=1,
        # )
        # ax.fill_between(
        #     plotdata["position"],
        #     plotdata["deltacor"],
        #     0,
        #     color="#333",
        #     alpha=0.2,
        #     lw=0,
        # )

        if label_y is True:
            label_y = "Predictivity\n($\\Delta$ cor)"
        ax.set_ylabel(
            label_y,
            rotation=0,
            ha="right",
            va="center",
        )

        ax.set_xticks([])
        ax.invert_yaxis()
        ax.set_ylim(0, max(limit, ax.get_ylim()[1]))

        if show_accessibility:
            ax2 = self.add_twinx()
            ax2.plot(
                plotdata["position"],
                plotdata["lost"],
                color="tomato",
                # color="#333",
                lw=1,
            )
            ax2.fill_between(
                plotdata["position"],
                plotdata["lost"],
                0,
                color="tomato",
                alpha=0.2,
                lw=0,
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
            ax2.set_ylim(
                0,
                plotdata["lost"].max() / (plotdata["deltacor"].min() / ax.get_ylim()[1]),
            )

        # change vertical alignment of last y tick to bottom
        ax.set_yticks([0, ax.get_ylim()[1]])
        ax.get_yticklabels()[-1].set_verticalalignment("top")
        ax.get_yticklabels()[0].set_verticalalignment("bottom")

        # vline at tss
        ax.axvline(0, color="#888888", lw=0.5, zorder=-1, dashes=(2, 2))

    @classmethod
    def from_regionmultiwindow(cls, regionmultiwindow, gene, width, show_accessibility=False, window=None):
        """
        Plot predictivity of a specific gene using a RegionMultiWindow object
        """
        plotdata = regionmultiwindow.get_plotdata(gene).reset_index()
        if window is None:
            window = np.array([plotdata["position"].min(), plotdata["position"].max()])
        return cls(plotdata, width=width, show_accessibility=show_accessibility, window=window)

    def add_arrow(self, position, y=0.5, orientation="left"):
        ax = self.ax
        trans = mpl.transforms.blended_transform_factory(x_transform=ax.transData, y_transform=ax.transAxes)
        if orientation == "left":
            xytext = (15, 15)
        elif orientation == "right":
            xytext = (-15, 15)
        ax.annotate(
            text="",
            xy=(position, y),
            xytext=xytext,
            textcoords="offset points",
            xycoords=trans,
            arrowprops=dict(arrowstyle="->", color="black", lw=1, connectionstyle="arc3"),
        )


class Pileup(polyptich.grid.Panel):
    def __init__(
        self,
        plotdata,
        window,
        width,
        ymax=2.0,
    ):
        super().__init__((width, 0.5))
        if "position" not in plotdata.columns:
            plotdata = plotdata.reset_index()

        ax = self.ax
        ax.set_xlim(*window)
        ax.plot(
            plotdata["position"],
            plotdata["lost"],
            color="#333",
            lw=1,
        )
        ax.fill_between(
            plotdata["position"],
            plotdata["lost"],
            0,
            color="#333",
            alpha=0.2,
            lw=0,
        )
        ax.set_xlim(ax.get_xlim())
        ax.set_ylabel(
            "# fragments\nper 1kb\nper 1k cells",
            rotation=0,
            ha="right",
            va="center",
        )

        # change vertical alignment of last y tick to bottom
        ax.set_yticks([0, ymax])
        ax.get_yticklabels()[-1].set_verticalalignment("top")
        ax.get_yticklabels()[0].set_verticalalignment("bottom")

        # vline at tss
        ax.axvline(0, color="#888888", lw=0.5, zorder=-1, dashes=(2, 2))

        ax.set_xticks([])
        ax.set_ylim(0, ymax)

    @classmethod
    def from_regionmultiwindow(cls, regionmultiwindow, gene, width, window=None):
        """
        Plot pileup of a specific gene using a regionmultiwindow object
        """
        plotdata = regionmultiwindow.get_plotdata(gene).reset_index()
        if window is None:
            window = np.array([plotdata["position"].min(), plotdata["position"].max()])
        return cls(plotdata, width=width, window=window)


class PredictivityBroken(Broken):
    def __init__(self, plotdata, breaking, height=0.5, ymax=None, reverse=False, **kwargs):
        super().__init__(
            breaking=breaking,
            height=height,
            **kwargs,
        )

        if ymax is None:
            ymax = plotdata["deltacor"].min() * 1.05

        for (region, region_info), (panel, ax) in zip(breaking.regions.iterrows(), self.elements[0]):
            plotdata_region = plotdata[
                (plotdata["position"] >= region_info["start"]) & (plotdata["position"] <= region_info["end"])
            ].copy()

            plotdata_region["effect_sign"] = np.sign(plotdata_region["effect"])
            plotdata_region["segment"] = plotdata_region["effect_sign"].diff().ne(0).cumsum()

            for segment, segment_data in plotdata_region.groupby("segment"):
                color = "tomato" if segment_data["effect"].iloc[0] > 0 else "#0074D9"
                ax.plot(
                    segment_data["position"],
                    segment_data["deltacor"],
                    lw=1,
                    color=color,
                )
                ax.fill_between(
                    segment_data["position"],
                    segment_data["deltacor"],
                    0,
                    alpha=0.2,
                    lw=0,
                    color=color,
                )
            if reverse:
                ax.set_ylim(ymax, 0)
            else:
                ax.set_ylim(0, ymax)
            ax.axvline(0, color="#888888", lw=0.5, zorder=-1, dashes=(2, 2))

        ax = self[0, 0].ax
        ax.set_ylabel(
            "$\\Delta$ cor",
            rotation=0,
            ha="right",
            va="center",
        )
        ax.set_yticks([0, ymax])
        ax.get_yticklabels()[-1].set_verticalalignment("top")
        ax.get_yticklabels()[0].set_verticalalignment("bottom")

    @classmethod
    def from_regionmultiwindow(cls, regionmultiwindow, gene, breaking, *args, **kwargs):
        """
        Plot pileup of a specific gene using a regionmultiwindow object
        """
        plotdata = regionmultiwindow.get_plotdata(gene).reset_index()
        return cls(plotdata, breaking=breaking, *args, **kwargs)


class PileupBroken(Broken):
    def __init__(self, plotdata, breaking, height=0.5, ymax=2.0, reverse=False, **kwargs):
        super().__init__(
            breaking=breaking,
            height=height,
            **kwargs,
        )

        if ymax is None:
            ymax = plotdata["lost"].max() * 1.05

        for (region, region_info), (panel, ax) in zip(breaking.regions.iterrows(), self.elements[0]):
            plotdata_region = plotdata[
                (plotdata["position"] >= region_info["start"]) & (plotdata["position"] <= region_info["end"])
            ].copy()

            ax.plot(
                plotdata_region["position"],
                plotdata_region["lost"],
                color="#333",
                lw=1,
            )
            ax.fill_between(
                plotdata_region["position"],
                plotdata_region["lost"],
                0,
                color="#333",
                alpha=0.2,
                lw=0,
            )
            if reverse:
                ax.set_ylim(ymax, 0)
            else:
                ax.set_ylim(0, ymax)
            ax.axvline(0, color="#888888", lw=0.5, zorder=-1, dashes=(2, 2))

        ax = self[0, 0].ax
        ax.set_ylabel(
            "# fragments\nper 1kb\nper 1k cells",
            rotation=0,
            ha="right",
            va="center",
        )

    @classmethod
    def from_regionmultiwindow(cls, regionmultiwindow, gene, breaking, **kwargs):
        """
        Plot pileup of a specific gene using a regionmultiwindow object
        """
        plotdata = regionmultiwindow.get_plotdata(gene).reset_index()
        return cls(plotdata, breaking=breaking, **kwargs)


# class LabelBroken(Broken):
#     def __init__(self, regions, *args, **kwargs):
#         super().__init__(regions=regions, height=0.00001, *args, **kwargs)

#         assert len(self.elements[0]) == len(regions)

#         for (region, region_info), (panel, ax) in zip(regions.iterrows(), self.elements[0]):
#             ax.set_xlim(region_info["start"], region_info["end"])
#             ax.set_xticks([])
#             ax.set_ylim(0, 0.1)
#             ax.set_yticks([])
#             ax.spines.left.set_visible(False)
#             ax.spines.top.set_visible(False)
#             ax.spines.right.set_visible(False)

#             # plot mean position
#             if panel.dim[0] > 0.1:
#                 ax.set_xticks([region_info["start"] + (region_info["end"] - region_info["start"]) / 2])
#                 ax.tick_params(
#                     axis="x",
#                     rotation=90,
#                     pad=5,
#                     length=0,
#                     which="major",
#                     top=True,
#                     labeltop=True,
#                     bottom=False,
#                     labelbottom=False,
#                     labelsize=8,
#                 )
#                 ax.xaxis.set_major_formatter(
#                     mpl.ticker.FuncFormatter(
#                         lambda x, pos: f"{x / 1000:+.0f}kb" if x % 1000 == 0 else f"{x / 1000:+.1f}kb"
#                     )
#                 )
#             # set region boundaries
#             ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator([region_info["start"], region_info["end"]]))
#             ax.tick_params(axis="x", which="minor", top=True, bottom=False)

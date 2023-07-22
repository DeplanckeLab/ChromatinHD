import chromatinhd.grid
import matplotlib as mpl
from chromatinhd.grid.broken import Broken, Panel
import numpy as np


class Predictivity(chromatinhd.grid.Panel):
    def __init__(self, plotdata, window, width, show_accessibility=False):
        super().__init__((width, 0.5))

        plotdata["effect_sign"] = np.sign(plotdata["effect"])
        plotdata["segment"] = plotdata["effect_sign"].diff().ne(0).cumsum()

        ax = self.ax
        ax.set_xlim(*window)

        for segment, segment_data in plotdata.groupby("segment"):
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

        ax.set_ylabel(
            "Predictivity\n($\\Delta$ cor)",
            rotation=0,
            ha="right",
            va="center",
        )

        ax.set_xticks([])
        ax.invert_yaxis()
        ax.set_ylim(0, max(-0.05, ax.get_ylim()[1]))

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
                plotdata["lost"].max()
                / (plotdata["deltacor"].min() / ax.get_ylim()[1]),
            )

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


class Pileup(chromatinhd.grid.Panel):
    def __init__(self, plotdata, window, width):
        super().__init__((width, 0.5))

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
        ax.set_yticks([0, ax.get_ylim()[1]])
        ax.get_yticklabels()[-1].set_verticalalignment("top")
        ax.get_yticklabels()[0].set_verticalalignment("bottom")

        # vline at tss
        ax.axvline(0, color="#888888", lw=0.5, zorder=-1, dashes=(2, 2))

        ax.set_xticks([])
        ax.set_ylim(0)

    @classmethod
    def from_genemultiwindow(cls, genemultiwindow, gene, width):
        plotdata = genemultiwindow.get_plotdata(gene).reset_index()
        window = np.array([plotdata["position"].min(), plotdata["position"].max()])
        return cls(plotdata, window, width)


class PredictivityBroken(Broken):
    def __init__(
        self, plotdata, regions, width, *args, gap=1, break_size=4, height=0.5, **kwargs
    ):
        super().__init__(
            regions=regions,
            height=height,
            width=width,
            gap=gap,
            *args,
            **kwargs,
        )

        ylim = plotdata["deltacor"].min() * 1.05

        for ((region, region_info), (panel, ax)) in zip(
            regions.iterrows(), self.elements[0]
        ):
            plotdata_region = plotdata[
                (plotdata["position"] >= region_info["start"])
                & (plotdata["position"] <= region_info["end"])
            ]

            ax.plot(
                plotdata_region["position"],
                plotdata_region["deltacor"],
                color="#333",
                lw=1,
            )

            c = [
                "tomato" if effect > 0 else "#333"
                for effect in plotdata_region["effect"]
            ]
            ax.scatter(
                plotdata_region["position"],
                plotdata_region["deltacor"],
                c=c,
                s=3,
                zorder=5,
            )
            ax.set_ylim(0, ylim)
            ax.axvline(0, color="#888888", lw=0.5, zorder=-1, dashes=(2, 2))

        ax = self[0, 0].ax
        ax.set_ylabel(
            "$\\Delta$ cor",
            rotation=0,
            ha="right",
            va="center",
        )


class LabelBroken(Broken):
    def __init__(self, regions, *args, **kwargs):
        super().__init__(regions=regions, height=0.00001, *args, **kwargs)

        assert len(self.elements[0]) == len(regions)

        for ((region, region_info), (panel, ax)) in zip(
            regions.iterrows(), self.elements[0]
        ):
            ax.set_xlim(region_info["start"], region_info["end"])
            ax.set_xticks([])
            ax.set_ylim(0, 0.1)
            ax.set_yticks([])
            ax.spines.left.set_visible(False)
            ax.spines.top.set_visible(False)
            ax.spines.right.set_visible(False)

            # plot mean position
            if panel.dim[0] > 0.1:
                ax.set_xticks(
                    [
                        region_info["start"]
                        + (region_info["end"] - region_info["start"]) / 2
                    ]
                )
                ax.tick_params(
                    axis="x",
                    rotation=90,
                    pad=5,
                    length=0,
                    which="major",
                    top=True,
                    labeltop=True,
                    bottom=False,
                    labelbottom=False,
                    labelsize=8,
                )
                ax.xaxis.set_major_formatter(
                    mpl.ticker.FuncFormatter(
                        lambda x, pos: f"{x / 1000:+.0f}kb"
                        if x % 1000 == 0
                        else f"{x / 1000:+.1f}kb"
                    )
                )
            # set region boundaries
            ax.xaxis.set_minor_locator(
                mpl.ticker.FixedLocator([region_info["start"], region_info["end"]])
            )
            ax.tick_params(axis="x", which="minor", top=True, bottom=False)

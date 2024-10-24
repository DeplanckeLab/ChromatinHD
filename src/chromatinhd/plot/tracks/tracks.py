import polyptich as pp
import pandas as pd
import numpy as np
import matplotlib as mpl

class TracksBroken(pp.grid.broken.Broken):
    """
    Visualize multiple tracks in a broken axis plot.

    """

    def __init__(
        self, design, region, breaking, window=None, height=0.6, **kwargs
    ):
        super().__init__(
            breaking=breaking,
            height=height,
            **kwargs,
        )

        if "tss" in region:
            pad = region["start"] - region["tss"]
        else:
            pad = 0

        plotdata = pd.DataFrame(
            {
                "pos": np.arange(region["end"] - region["start"]) + pad,
            }
        )
        for k, bw in design["bw"].items():
            plotdata[k] = bw.values(region["chrom"], region["start"], region["end"])[
                :: region["strand"]
            ]

        for i, (region, region_info), (panel, ax) in zip(
            range(len(breaking.regions)), breaking.regions.iterrows(), self.elements[0]
        ):
            plotdata_region = plotdata.loc[
                (plotdata["pos"] >= region_info["start"])
                & (plotdata["pos"] <= region_info["end"])
            ]
            if len(plotdata_region) > 0:
                for k in design.index:
                    ax.fill_between(
                        plotdata_region["pos"],
                        0,
                        plotdata_region[k],
                        label=k,
                        lw=0.,
                        color="#33333366",
                    )
            ax.set_yscale("symlog", linthresh=10)
            ax.set_ylim(0, 50)
            ax.set_yticks([], minor = True)
            ax.set_yticks([])
            ax.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
            # top ytick vertical alignment top
            if i == 0:
                ax.set_yticks([0, 10, 100])
                ax.set_yticks([0, 2.5, 5, 7.5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], minor = True)
                ax.get_yticklabels()[0].set_verticalalignment("bottom")
                ax.get_yticklabels()[-1].set_verticalalignment("top")
                ax.tick_params(axis="y", length=4, labelsize=6, pad = 1)

    @classmethod
    def from_bigwigs(
        cls,
        design: pd.DataFrame,
        region: pd.Series,
        breaking: pp.grid.Breaking,
        window=None,
        height=0.6,
        **kwargs,
    ):
        """
        Visualize multiple tracks in a broken axis plot.

        Parameters
        ----------
        design:
            Design of the tracks to plot. Has to have the following columns:
            - `bw`: bigwig file path
            - `name`: name of the track
            optionally:
            - `color`: color of the track
        region:
            Dictionary/pd.Series with the following
            - `chrom`: chromosome
            - `start`: start position
            - `end`: end position
            - `strand`: strand
        breaking : pp.grid.Breaking
            Breaking object
        """
        try:
            import pyBigWig
        except ImportError:
            raise ImportError(
                "pyBigWig is required for this method. Please install it using `pip install pyBigWig`."
            )
        design["bw"] = [pyBigWig.open(f"{sample}.bw") for sample in design["sample"]]
        return TracksBroken(
            design=design,
            region=region,
            breaking=breaking,
            window=window,
            height=height,
            **kwargs,
        )

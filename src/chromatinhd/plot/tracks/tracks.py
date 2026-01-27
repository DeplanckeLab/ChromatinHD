import polyptich as pp
import pandas as pd
import numpy as np
import matplotlib as mpl

class TracksBroken(pp.grid.broken.Broken):
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
    window:
        Window size to plot. If None, the whole region is plotted.
    height:
        Height of the plot. Default is 0.6.
    scale_column:
        Column in the design dataframe to use for scaling. If None, will be ugly
    """

    def __init__(
        self, design, region, breaking, window=None, height=0.6, scale_column = None, **kwargs
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

        plotdata_input = plotdata.copy()

        for k, bw in design["bw"].items():
            values = np.array(bw.values(region["chrom"], region["start"], region["end"])[
                :: region["strand"]
            ])

            if scale_column is not None:
                scale = design.loc[k, scale_column]
            else:
                scale = values.max()
            values = values/scale
        
            plotdata[k] = values

            if "bw_input" in design.columns:
                bw_input = design.loc[k, "bw_input"]
                values_input = np.array(bw_input.values(region["chrom"], region["start"], region["end"])[
                    :: region["strand"]
                ])
                values_input = values_input/ scale
                plotdata_input[k] = values_input
        

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
                        color="#7c9c9099",
                    )

                    if "bw_input" in design.columns:
                        plotdata_region_input = plotdata_input.loc[
                            (plotdata_input["pos"] >= region_info["start"])
                            & (plotdata_input["pos"] <= region_info["end"])
                        ]
                        ax.fill_between(
                            plotdata_region_input["pos"],
                            0,
                            plotdata_region_input[k],
                            label=f"{k} input",
                            lw=0.,
                            color="#d1c1cd99",
                        )
            ax.set_ylim(0, 1)

            if i == 0:
                ax.set_yticks([0, 1.0])
                ax.set_yticklabels(["0", f"{scale:.1f}"], fontsize=6)
                ax.get_yticklabels()[0].set_verticalalignment("bottom")
                ax.get_yticklabels()[-1].set_verticalalignment("top")
            else:
                ax.set_yticks([])

            

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
        if "sample_input" in design.columns.tolist():
            design["bw_input"] = [pyBigWig.open(f"{sample}.bw") for sample in design["sample_input"]]
        return TracksBroken(
            design=design,
            region=region,
            breaking=breaking,
            window=window,
            height=height,
            **kwargs,
        )

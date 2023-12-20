from chromatinhd.grid.broken import Broken, Panel
import pandas as pd
import adjustText
import seaborn as sns


def center_position(peaks, region):
    peaks = peaks.copy()
    if peaks.shape[0] == 0:
        peaks = pd.DataFrame(columns=["pos"])
    else:
        peaks[["pos"]] = [
            [
                (peak["pos"] - region["tss"]) * int(region["strand"]),
            ][:: int(region["strand"])]
            for _, peak in peaks.iterrows()
        ]
    return peaks


class Associations(Panel):
    def __init__(self, associations, region_id, width, window=None, *args, height=0.1, show_ld=True, **kwargs):
        super().__init__(
            (width, height),
            *args,
            **kwargs,
        )
        ax = self.ax

        region_oi = associations.regions.coordinates.loc[region_id]

        if window is None:
            window = associations.regions.window

        plotdata = associations.association.query("chr == @region_oi.chrom").copy()
        plotdata = center_position(plotdata, region_oi)

        if not show_ld:
            plotdata = plotdata.loc[plotdata["snp_main"] == plotdata["snp"]]

        plotdata = plotdata.loc[(plotdata["pos"] >= window[0]) & (plotdata["pos"] <= window[1])]
        plotdata = plotdata.groupby("snp").first().reset_index()

        snpmain_colors = {
            snpmain: color for snpmain, color in zip(plotdata["snp_main"].unique(), sns.color_palette("tab10"))
        }

        ax.scatter(
            plotdata["pos"],
            [0.3] * len(plotdata),
            marker="v",
            s=5,
            c=plotdata["snp_main"].map(snpmain_colors),
        )

        texts = []
        for i, (index, row) in enumerate(plotdata.iterrows()):
            texts.append(
                ax.text(
                    row["pos"],
                    0.8,
                    row["rsid"],
                    fontsize=5,
                    ha="center",
                    va="bottom",
                    color=snpmain_colors[row["snp_main"]],
                )
            )
        self.texts = texts
        ax.set_ylim(0, 1)
        ax.annotate(
            "Immune GWAS",
            (0, 0.5),
            xytext=(-2, 0),
            xycoords="axes fraction",
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=10,
        )
        ax.set_xlim(*window)
        ax.axis("off")
        ax.axis("off")


class SNPsBroken(Panel):
    def __init__(self, plotdata, regions, width, transform, *args, gap=1, height=0.1, **kwargs):
        super().__init__(
            (width, height),
            *args,
            **kwargs,
        )
        ax = self.ax

        plotdata["position_broken"] = transform(plotdata["pos"].values)
        plotdata = plotdata.loc[~pd.isnull(plotdata["position_broken"])]

        snpmain_colors = {
            snpmain: color for snpmain, color in zip(plotdata["snp_main"].unique(), sns.color_palette("tab10"))
        }

        ax.scatter(
            plotdata["position_broken"],
            [0.3] * len(plotdata),
            marker="v",
            s=3,
            c=plotdata["snp_main"].map(snpmain_colors),
        )

        texts = []
        for i, (index, row) in enumerate(plotdata.iterrows()):
            texts.append(
                ax.text(
                    row["position_broken"],
                    0.8,
                    row["rsid"],
                    fontsize=5,
                    ha="center",
                    va="bottom",
                    color=snpmain_colors[row["snp_main"]],
                )
            )
        self.texts = texts
        ax.set_ylim(0, 1)
        ax.set_xlim(regions["cumend"].min(), regions["cumend"].max())
        ax.axis("off")
        ax.axis("off")

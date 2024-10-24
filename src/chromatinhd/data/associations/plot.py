from polyptich.grid.broken import Broken, Panel
import pandas as pd
import adjustText
import seaborn as sns
import matplotlib as mpl

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
    def __init__(
        self, associations, region_id, width, window=None, *args, height=0.13, show_ld=True, label_y="GWAS", **kwargs
    ):
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
            snpmain: color for snpmain, color in zip(plotdata["snp_main"].unique(), sns.color_palette("tab20"))
        }

        ax.scatter(
            plotdata["pos"],
            [0.05] * len(plotdata),
            marker="|",
            s=10,
            c=plotdata["snp_main"].map(snpmain_colors),
        )

        texts = []
        for i, (index, row) in enumerate(plotdata.iterrows()):
            texts.append(
                ax.text(
                    row["pos"],
                    0.3,
                    row["rsid"],
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    color=snpmain_colors[row["snp_main"]],
                )
            )
        self.texts = texts
        ax.set_ylim(0, 1)

        if label_y is not False:
            ax.annotate(
                label_y,
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


class AssociationsBroken(Broken):
    def __init__(self, associations, region_id, breaking, window=None, height=0.1, show_ld=True, **kwargs):
        super().__init__(
            breaking=breaking,
            height=height,
            **kwargs,
        )

        region_oi = associations.regions.coordinates.loc[region_id]

        if window is None:
            window = associations.regions.window

        plotdata = associations.association.query("chr == @region_oi.chrom").copy()
        plotdata = plotdata.loc[(plotdata["pos"] >= region_oi["start"]) & (plotdata["pos"] <= region_oi["end"])]
        plotdata = center_position(plotdata, region_oi)
        plotdata = plotdata.groupby("pos").first().reset_index()

        snpmain_colors = {
            # snpmain: color for snpmain, color in zip(plotdata["snp_main"].unique(), sns.color_palette("tab10"))
        }

        def get_snp_color(snp_main):
            if snp_main not in snpmain_colors:
                if len(snpmain_colors) > 9:
                    snpmain_colors[snp_main] = "grey"
                else:
                    snpmain_colors[snp_main] = sns.color_palette("tab10")[len(snpmain_colors)]
            return snpmain_colors[snp_main]

        if not show_ld:
            plotdata = plotdata.loc[plotdata["snp_main"] == plotdata["snp"]]

        for (region, region_info), (panel, ax) in zip(breaking.regions.iterrows(), self.elements[0]):
            ax.axis("off")
            plotdata_region = plotdata.loc[
                (plotdata["pos"] >= region_info["start"]) & (plotdata["pos"] <= region_info["end"])
            ]
            if len(plotdata_region) > 0:
                colors = [get_snp_color(snp_main) for snp_main in plotdata_region["snp_main"]]
                ax.scatter(
                    plotdata_region["pos"],
                    [0.3] * len(plotdata_region),
                    marker="v",
                    s=5,
                    c=colors,
                )

                for i, (index, row) in enumerate(plotdata_region.iterrows()):
                    text = ax.text(
                        row["pos"],
                        0.8,
                        row["rsid"],
                        fontsize=6,
                        ha="center",
                        va="bottom",
                        color=colors[i],
                    )
                    text.set_path_effects(
                        [
                            mpl.patheffects.withStroke(linewidth=2, foreground="white"),
                        ]
                    )
                # ax.scatter(
                #     plotdata_region["position"],
                #     [1] * len(plotdata_region),
                #     transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                #     marker="v",
                #     color=motif.color,
                #     alpha=1,
                #     s=100,
                #     zorder=20,
                # )

        # plotdata = plotdata.loc[(plotdata["pos"] >= window[0]) & (plotdata["pos"] <= window[1])]
        # plotdata = plotdata.groupby("snp").first().reset_index()

        # snpmain_colors = {
        #     snpmain: color for snpmain, color in zip(plotdata["snp_main"].unique(), sns.color_palette("tab20"))
        # }

        # ax.scatter(
        #     plotdata["pos"],
        #     [0.3] * len(plotdata),
        #     marker="v",
        #     s=5,
        #     c=plotdata["snp_main"].map(snpmain_colors),
        # )

        # texts = []
        # for i, (index, row) in enumerate(plotdata.iterrows()):
        #     texts.append(
        #         ax.text(
        #             row["pos"],
        #             0.8,
        #             row["rsid"],
        #             fontsize=5,
        #             ha="center",
        #             va="bottom",
        #             color=snpmain_colors[row["snp_main"]],
        #         )
        #     )
        # self.texts = texts
        # ax.set_ylim(0, 1)
        # ax.annotate(
        #     "Immune GWAS",
        #     (0, 0.5),
        #     xytext=(-2, 0),
        #     xycoords="axes fraction",
        #     textcoords="offset points",
        #     ha="right",
        #     va="center",
        #     fontsize=10,
        # )
        # ax.set_xlim(*window)
        # ax.axis("off")
        # ax.axis("off")


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

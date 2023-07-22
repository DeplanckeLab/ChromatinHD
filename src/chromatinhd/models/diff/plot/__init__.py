import chromatinhd
import chromatinhd.grid
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd

from .differential import Differential
from .differential_expression import DifferentialExpression


class MotifsLegend(chromatinhd.grid.Wrap):
    def __init__(self, motifs_oi, cluster_info, width, panel_height, **kwargs):
        super().__init__(ncol=1, **kwargs)

        motifs_shown = set()

        for cluster in cluster_info.index:
            ax = chromatinhd.grid.Ax((width, panel_height))
            self.add(ax)
            ax = ax.ax

            motifs_oi_cluster = motifs_oi.loc[
                [cluster in clusters for clusters in motifs_oi["clusters"]]
            ]

            n = len(motifs_oi_cluster)
            ax.axis("off")
            ax.set_xlim(-0.5, 2)
            ax.set_ylim((n + 4) / 2 - 4, (n + 4) / 2)

            for i, (motif, motif_info) in enumerate(
                motifs_oi_cluster.loc[
                    motifs_oi_cluster.index.difference(motifs_shown)
                ].iterrows()
            ):
                ax.scatter(
                    [0], [i], color=motif_info["color"], marker="|", clip_on=False
                )
                ax.text(
                    0.1,
                    i,
                    s=motif_info["label"],
                    ha="left",
                    va="center",
                    color=motif_info["color"],
                )
            motifs_shown.update(set(motifs_oi_cluster.index))

        self.set_title("Motifs")


class MotifsHighlighting:
    def __init__(self, wrap_differential, motifdata, motifs_oi, cluster_info):
        motifs_oi["indicator"] = range(len(motifs_oi))
        for cluster, ax in zip(cluster_info.index, wrap_differential):
            ax = ax.ax

            # motifs
            motifs_oi_cluster = motifs_oi.loc[
                [cluster in clusters for clusters in motifs_oi["clusters"]]
            ]
            motifdata_cluster = motifdata.loc[
                motifdata["motif"].isin(motifs_oi_cluster.index)
            ]
            texts = []
            for _, z in motifdata_cluster.iterrows():
                ax.axvline(
                    z["position"],
                    # color="grey",
                    color=motifs_oi.loc[z["motif"], "color"],
                    zorder=5,
                )
                ax.scatter(
                    z["position"],
                    ax.get_ylim()[-1],
                    color=motifs_oi.loc[z["motif"], "color"],
                    zorder=100,
                    marker="v",
                    s=100,
                )
                # text = ax.text(
                #     z["position"],
                #     ax.get_ylim()[-1],
                #     s=str(motifs_oi.loc[z["motif"], "indicator"]),
                #     va="top",
                #     ha="center",
                #     zorder=10,
                #     fontsize=6,
                # )
                # text.set_path_effects(
                #     [
                #         mpl.patheffects.Stroke(linewidth=2, foreground="white"),
                #         mpl.patheffects.Normal(),
                #     ]
                # )
                # texts.append(text)
            # import adjustText

            # adjustText.adjust_text(texts, ax=ax, autoalign="x", va="top", ha="center")


class Peaks(chromatinhd.grid.Ax):
    def __init__(
        self,
        peaks,
        peakcallers,
        window,
        width,
        label_methods=True,
        label_rows=True,
        label_methods_side="right",
        row_height=1,
    ):
        super().__init__((width, row_height * len(peakcallers) / 5))

        ax = self.ax
        ax.set_xlim(*window)
        for peakcaller, peaks_peakcaller in peaks.groupby("peakcaller"):
            y = peakcallers.loc[peakcaller, "ix"]

            if len(peaks_peakcaller) == 0:
                continue
            if ("cluster" not in peaks_peakcaller.columns) or pd.isnull(
                peaks_peakcaller["cluster"]
            ).all():
                for _, peak in peaks_peakcaller.iterrows():
                    rect = mpl.patches.Rectangle(
                        (peak["start"], y),
                        peak["end"] - peak["start"],
                        1,
                        fc="#333",
                        lw=0,
                    )
                    ax.add_patch(rect)
                    ax.plot([peak["start"]] * 2, [y, y + 1], color="grey", lw=0.5)
                    ax.plot([peak["end"]] * 2, [y, y + 1], color="grey", lw=0.5)
            else:
                n_clusters = peaks_peakcaller["cluster"].max() + 1
                h = 1 / n_clusters
                for _, peak in peaks_peakcaller.iterrows():
                    rect = mpl.patches.Rectangle(
                        (peak["start"], y + peak["cluster"] / n_clusters),
                        peak["end"] - peak["start"],
                        h,
                        fc="#333",
                        lw=0,
                    )
                    ax.add_patch(rect)
            if y > 0:
                ax.axhline(y, color="#DDD", zorder=10, lw=0.5)

        ax.set_ylim(peakcallers["ix"].max() + 1, 0)
        if label_methods:
            ax.set_yticks(peakcallers["ix"] + 0.5)
            ax.set_yticks(peakcallers["ix"].tolist() + [len(peakcallers)], minor=True)
            ax.set_yticklabels(
                peakcallers["label"],
                fontsize=min(16 * row_height, 10),
                va="center",
            )
            if label_rows:
                ax.set_ylabel("CREs", rotation=0, ha="right", va="center")
            else:
                ax.set_ylabel("")
        else:
            ax.set_yticks([])
            ax.set_ylabel("")
        ax.tick_params(
            axis="y",
            which="major",
            length=0,
            pad=1,
            right=label_methods_side == "right",
            left=not label_methods_side == "left",
        )
        ax.tick_params(
            axis="y",
            which="minor",
            length=1,
            pad=1,
            right=label_methods_side == "right",
            left=not label_methods_side == "left",
        )
        ax.yaxis.tick_right()

        ax.set_xticks([])


class Conservation(chromatinhd.grid.Ax):
    def __init__(self, plotdata_conservation, window, width):
        super().__init__((width, 0.3))

        ax = self.ax
        ax.set_xlim(*window)

        ax.plot(
            plotdata_conservation["position"],
            plotdata_conservation["conservation"],
            color="#333",
            lw=1,
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("Conservation", rotation=0, ha="right", va="center")
        ax.set_xticks([])


class GC(chromatinhd.grid.Ax):
    def __init__(self, plotdata_gc, window, width):
        super().__init__((width, 0.3))

        ax = self.ax
        ax.set_xlim(*window)

        ax.plot(
            plotdata_gc["position"],
            plotdata_gc["gc"],
            color="#333",
            lw=1,
        )

        ax.set_ylim(0, 1)
        ax.set_ylabel("%GC", rotation=0, ha="right", va="center")
        ax.set_xticks([])


def find_runs(x):
    return np.where((np.diff(np.concatenate([[False], x, [False]])) != 0))[0].reshape(
        (-1, 2)
    )


class CommonUnique:
    def __init__(
        self,
        ax,
        peak_position_chosen_oi,
        region_position_chosen_oi,
        expanded_slice_oi,
        window,
        method_info,
    ):
        # add region and peak unique spans
        trans = mpl.transforms.blended_transform_factory(
            y_transform=ax.transAxes, x_transform=ax.transData
        )
        for start, end in find_runs(
            peak_position_chosen_oi & ~region_position_chosen_oi
        ):
            start = start + expanded_slice_oi["start"] + window[0]
            end = end + expanded_slice_oi["start"] + window[0]

            color = method_info.loc["peak", "color"]

            ax.axvspan(
                start,
                end,
                fc=color,
                alpha=0.3,
                lw=0,
            )
            rect = mpl.patches.Rectangle(
                (start, 1),
                end - start,
                0.2,
                transform=trans,
                fc=color,
                clip_on=False,
                lw=0,
            )
            ax.add_patch(rect)
        for start, end in find_runs(
            peak_position_chosen_oi & region_position_chosen_oi
        ):
            color = method_info.loc["common", "color"]

            start = start + expanded_slice_oi["start"] + window[0]
            end = end + expanded_slice_oi["start"] + window[0]
            ax.axvspan(
                start,
                end,
                fc=color,
                alpha=0.3,
                lw=0,
            )
            rect = mpl.patches.Rectangle(
                (start, 1),
                end - start,
                0.2,
                transform=trans,
                fc=color,
                clip_on=False,
                lw=0,
            )
            ax.add_patch(rect)
        for start, end in find_runs(
            ~peak_position_chosen_oi & region_position_chosen_oi
        ):
            color = method_info.loc["region", "color"]

            start = start + expanded_slice_oi["start"] + window[0]
            end = end + expanded_slice_oi["start"] + window[0]
            ax.axvspan(
                start,
                end,
                fc=color,
                alpha=0.3,
                lw=0,
            )

            rect = mpl.patches.Rectangle(
                (start, 1),
                end - start,
                0.2,
                transform=trans,
                fc=color,
                lw=0,
                clip_on=False,
            )
            ax.add_patch(rect)


class LabelSlice:
    def __init__(self, ax, gene_label, cluster_label, slice_oi, window):
        start = slice_oi["start"] + window[0]
        end = slice_oi["end"] + window[0]

        text = ax.annotate(
            f"$\\it{{{gene_label}}}$ $\\bf{{{cluster_label}}}$",
            (0, 1),
            (2, 2),
            va="bottom",
            ha="left",
            xycoords="axes fraction",
            textcoords="offset points",
            fontsize=6,
            color="#999",
            zorder=200,
        )
        text.set_path_effects(
            [
                mpl.patheffects.Stroke(foreground="#FFFFFFFF", linewidth=2),
                mpl.patheffects.Normal(),
            ]
        )

        trans = mpl.transforms.blended_transform_factory(
            y_transform=ax.transAxes, x_transform=ax.transData
        )
        text = ax.annotate(
            f"{start:+}",
            (start, 1),
            (-2, -2),
            va="top",
            ha="right",
            xycoords=trans,
            textcoords="offset points",
            fontsize=6,
            color="#999",
            zorder=200,
        )
        text.set_path_effects(
            [
                mpl.patheffects.Stroke(foreground="white", linewidth=2),
                mpl.patheffects.Normal(),
            ]
        )
        text = ax.annotate(
            f"{end:+}",
            (end, 1),
            (2, -2),
            va="top",
            ha="left",
            xycoords=trans,
            textcoords="offset points",
            fontsize=6,
            color="#999",
            zorder=200,
        )
        text.set_path_effects(
            [
                mpl.patheffects.Stroke(foreground="white", linewidth=2),
                mpl.patheffects.Normal(),
            ]
        )


class LegendResolution:
    def __init__(self, ax, resolution):
        pass


import io

chromstate_info = pd.read_table(
    io.StringIO(
        """ix	mnemomic	description	color_name	color_code
1	TssA	Active TSS	Red	255,0,0
2	TssAFlnk	Flanking Active TSS	Orange Red	255,69,0
3	TxFlnk	Transcr. at gene 5' and 3'	LimeGreen	50,205,50
4	Tx	Strong transcription	Green	0,128,0
5	TxWk	Weak transcription	DarkGreen	0,100,0
6	EnhG	Genic enhancers	GreenYellow	194,225,5
7	Enh	Enhancers	Yellow	255,255,0
8	ZNF/Rpts	ZNF genes & repeats	Medium Aquamarine	102,205,170
9	Het	Heterochromatin	PaleTurquoise	138,145,208
10	TssBiv	Bivalent/Poised TSS	IndianRed	205,92,92
11	BivFlnk	Flanking Bivalent TSS/Enh	DarkSalmon	233,150,122
12	EnhBiv	Bivalent Enhancer	DarkKhaki	189,183,107
13	ReprPC	Repressed PolyComb	Silver	128,128,128
14	ReprPCWk	Weak Repressed PolyComb	Gainsboro	192,192,192
15	Quies	Quiescent/Low	White	255,255,255
"""
    )
).set_index("mnemomic")
chromstate_info["color"] = [
    np.array(c.split(","), dtype=float) / 255 for c in chromstate_info["color_code"]
]


class Annot(chromatinhd.grid.Ax):
    def __init__(self, plotdata, window, width, cluster_info):
        super().__init__((width, len(cluster_info) * 0.15))

        ax = self.ax
        ax.set_xlim(*window)

        cluster_info["ix"] = np.arange(len(cluster_info))

        for cluster in cluster_info.index:
            plotdata_cluster = plotdata.query("cluster == @cluster")
            y = cluster_info.loc[cluster, "ix"]
            for _, annot in plotdata_cluster.iterrows():
                color = chromstate_info.loc[annot["name"], "color"]
                patch = mpl.patches.Rectangle(
                    (annot["start"], y), annot["end"] - annot["start"], 1, fc=color
                )
                ax.add_patch(patch)

        ax.set_ylim(0, len(cluster_info))
        ax.set_yticks(cluster_info["ix"] + 0.5)
        ax.set_yticklabels(cluster_info.index)
        ax.invert_yaxis()

        # ax.plot(
        #     plotdata["position"],
        #     plotdata["gc"],
        #     color="#333",
        #     lw=1,
        # )

        # ax.set_ylim(0, 1)
        # ax.set_ylabel("%GC", rotation=0, ha="right", va="center")
        # ax.set_xticks([])


class AnnotLegend(chromatinhd.grid.Ax):
    def __init__(self, ax_annot, width=3):
        super().__init__((width, ax_annot.dim[1]))

        ax = self.ax

        chromstate_info["ix"] = np.arange(len(chromstate_info))

        ax.set_ylim(0, self.dim[1])

        for chromstate, chromstate_info_ in chromstate_info.iterrows():
            ax.text(
                0,
                chromstate_info_["ix"] * 0.1,
                fontsize=8,
                s=chromstate_info_["description"],
                color=chromstate_info_["color"],
                va="top",
            )
        ax.invert_yaxis()
        ax.axis("off")

import chromatinhd.grid
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd


class Genes(chromatinhd.grid.Ax):
    def __init__(
        self,
        plotdata_genes,
        plotdata_exons,
        plotdata_coding,
        gene_id,
        promoter,
        window,
        width,
    ):
        super().__init__((width, len(plotdata_genes) * 0.08))

        ax = self.ax

        ax.xaxis.tick_top()
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("Distance to TSS")
        ax.xaxis.set_label_position("top")

        sns.despine(ax=ax, right=True, left=True, bottom=True)

        ax.set_xlim(*window)
        ax.set_ylim(-0.5, plotdata_genes["ix"].max() + 0.5)
        for gene, gene_info in plotdata_genes.iterrows():
            y = gene_info["ix"]
            is_oi = gene == gene_id
            ax.plot(
                [gene_info["start"], gene_info["end"]],
                [y, y],
                color="black" if is_oi else "grey",
            )

            if (gene_info["start"] > window[0]) & (gene_info["start"] < window[1]):
                strand = gene_info["strand"] * promoter["strand"]

                label = (
                    gene_info["symbol"] + " > "
                    if strand == 1
                    else " < " + gene_info["symbol"]
                )
                ha = "right"
                # ha = "left" if (strand == -1) else "right"

                ax.text(
                    gene_info["start"],
                    y,
                    label,
                    style="italic",
                    ha=ha,
                    va="center",
                    fontsize=6,
                    weight="bold" if is_oi else "regular",
                )
            else:
                label = gene_info["symbol"]
                ax.text(
                    0,
                    y,
                    "(" + label + ")",
                    style="italic",
                    ha="center",
                    va="center",
                    fontsize=6,
                    bbox=dict(facecolor="#FFFFFF88", boxstyle="square,pad=0", lw=0),
                )

            plotdata_exons_gene = plotdata_exons.query("gene == @gene")
            h = 1
            for exon, exon_info in plotdata_exons_gene.iterrows():
                rect = mpl.patches.Rectangle(
                    (exon_info["start"], y - h / 2),
                    exon_info["end"] - exon_info["start"],
                    h,
                    fc="white",
                    ec="#333333",
                    lw=1.0,
                    zorder=9,
                )
                ax.add_patch(rect)

            plotdata_coding_gene = plotdata_coding.query("gene == @gene")
            for coding, coding_info in plotdata_coding_gene.iterrows():
                rect = mpl.patches.Rectangle(
                    (coding_info["start"], y - h / 2),
                    coding_info["end"] - coding_info["start"],
                    h,
                    fc="#333333",
                    ec="#333333",
                    lw=1.0,
                    zorder=10,
                )
                ax.add_patch(rect)


class Differential(chromatinhd.grid.Wrap):
    def __init__(
        self,
        plotdata_genome,
        plotdata_genome_mean,
        cluster_info,
        window,
        width,
        panel_height,
        plotdata_empirical=None,
        show_atac_diff=True,
        cmap_atac_diff=mpl.cm.RdBu_r,
        norm_atac_diff=mpl.colors.Normalize(np.log(0.5), np.log(2.0), clip=True),
        title=True,
        **kwargs,
    ):
        super().__init__(ncol=1, **kwargs)

        for cluster_ix in cluster_info["dimension"]:
            ax_genome = chromatinhd.grid.Ax((width, panel_height))
            self.add(ax_genome)

            # genome
            ax = ax_genome.ax
            ax.set_ylim(0, 20)
            ax.set_xlim(*window)
            ax.axvline(0, dashes=(1, 1), color="#AAA", zorder=-1)

            if plotdata_empirical is not None:
                # empirical distribution of atac-seq cuts
                plotdata_empirical_cluster = plotdata_empirical.query(
                    "cluster == @cluster_ix"
                )
                ax.fill_between(
                    plotdata_empirical_cluster["coord"],
                    np.exp(plotdata_empirical_cluster["prob"]),
                    alpha=0.2,
                    color="#333",
                )

            if show_atac_diff:
                # posterior distribution of atac-seq cuts
                plotdata_genome_cluster = plotdata_genome.xs(
                    cluster_ix, level="cluster"
                )
                ax.plot(
                    plotdata_genome_mean.index,
                    np.exp(plotdata_genome_mean["prob"]),
                    color="black",
                    lw=1,
                    zorder=1,
                    linestyle="dashed",
                )
                ax.plot(
                    plotdata_genome_cluster.index,
                    np.exp(plotdata_genome_cluster["prob"]),
                    color="black",
                    lw=1,
                    zorder=1,
                )
                polygon = ax.fill_between(
                    plotdata_genome_mean.index,
                    np.exp(plotdata_genome_mean["prob"]),
                    np.exp(plotdata_genome_cluster["prob"]),
                    color="black",
                    zorder=0,
                )

                # up/down gradient
                verts = np.vstack([p.vertices for p in polygon.get_paths()])
                c = plotdata_genome_cluster["prob_diff"].values
                c[c == np.inf] = 0.0
                gradient = ax.imshow(
                    c.reshape(1, -1),
                    cmap=cmap_atac_diff,
                    aspect="auto",
                    extent=[
                        verts[:, 0].min(),
                        verts[:, 0].max(),
                        verts[:, 1].min(),
                        verts[:, 1].max(),
                    ],
                    zorder=25,
                    norm=norm_atac_diff,
                )
                gradient.set_clip_path(polygon.get_paths()[0], transform=ax.transData)
                polygon.set_alpha(0)

                ax.set_yticks([])
                ax.set_xticks([])

        if title is not False:
            self.set_title("ATAC-seq insertion")


class DifferentialExpression(chromatinhd.grid.Wrap):
    def __init__(
        self,
        plotdata_expression,
        plotdata_expression_clusters,
        cluster_info,
        width,
        panel_height,
        cmap_expression=mpl.cm.Reds,
        norm_expression=None,
        **kwargs,
    ):
        super().__init__(ncol=1, **kwargs)

        if norm_expression is None:
            norm_expression = mpl.colors.Normalize(
                0.0, plotdata_expression_clusters.max(), clip=True
            )

        for cluster_ix in cluster_info["dimension"]:
            ax = chromatinhd.grid.Ax((width, panel_height))
            self.add(ax)
            ax = ax.ax

            sns.despine(ax=ax, left=True, right=True, top=True, bottom=True)
            ax.set_yticks([])
            ax.set_xticks([])

            circle = mpl.patches.Circle(
                (0, 0),
                norm_expression(plotdata_expression_clusters.iloc[cluster_ix]) * 0.9
                + 0.1,
                fc=cmap_expression(
                    norm_expression(plotdata_expression_clusters.iloc[cluster_ix])
                ),
                lw=1,
                ec="#333333",
            )
            ax.add_patch(circle)
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_aspect(1)

            ax.set_ylabel(
                f"{cluster_info.iloc[cluster_ix]['label']}",
                rotation=0,
                ha="right",
                va="center",
            )

        self.set_title("RNA-seq")


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
    def __init__(self, peaks, peak_methods, window, width):
        super().__init__((width, len(peak_methods) / 5))

        ax = self.ax
        ax.set_xlim(*window)
        for peakname, peaks_method in peaks.groupby("method"):
            if ("cluster" not in peaks_method.columns) or pd.isnull(
                peaks_method["cluster"]
            ).all():
                y = peak_methods.loc[peakname, "ix"]
                for _, peak in peaks_method.iterrows():
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
                y = peak_methods.loc[peakname, "ix"]
                n_clusters = len(peaks_method["cluster"].unique())
                h = 1 / n_clusters
                for _, peak in peaks_method.iterrows():
                    rect = mpl.patches.Rectangle(
                        (peak["start"], y + peak["cluster"] / n_clusters),
                        peak["end"] - peak["start"],
                        h,
                        fc="#333",
                        lw=0,
                    )
                    ax.add_patch(rect)
            y = peak_methods.loc[peakname, "ix"]
            if y > 0:
                ax.axhline(y, color="#DDD", zorder=10, lw=1)

        ax.set_ylim(peak_methods["ix"].max() + 1, 0)
        ax.set_yticks(peak_methods["ix"] + 0.5)
        ax.set_yticks(peak_methods["ix"].tolist() + [len(peak_methods)], minor=True)
        ax.set_yticklabels(peak_methods.index)
        ax.set_ylabel("Peaks", rotation=0, ha="right", va="center")
        ax.tick_params(axis="y", which="major", length=0)
        ax.tick_params(axis="y", which="minor", length=10)
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

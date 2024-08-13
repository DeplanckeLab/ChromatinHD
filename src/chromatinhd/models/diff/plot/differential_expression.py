import matplotlib as mpl
import seaborn as sns
import chromatinhd


def get_cmap_rna_diff():
    return mpl.cm.BuGn


class DifferentialExpression(chromatinhd.grid.Wrap):
    def __init__(
        self,
        plotdata_expression_clusters,
        cluster_info,
        width=0.5,
        panel_height=0.5,
        norm_expression=None,
        symbol=None,
        show_n_cells=True,
        order=False,
        **kwargs,
    ):
        super().__init__(ncol=1, **{"padding_height": 0, **kwargs})

        plotdata_expression_clusters = plotdata_expression_clusters.loc[cluster_info.index]

        if order is True:
            self.order = plotdata_expression_clusters.sort_values(ascending=False).index
        elif order is not False:
            self.order = order
        else:
            self.order = plotdata_expression_clusters.index

        if norm_expression is None:
            norm_expression = mpl.colors.Normalize(
                min(0.0, plotdata_expression_clusters.min()), plotdata_expression_clusters.max(), clip=True
            )

        cmap_expression = get_cmap_rna_diff()

        for cluster_id in self.order:
            panel, ax = chromatinhd.grid.Ax((width, panel_height))
            self.add(panel)

            sns.despine(ax=ax, left=True, right=True, top=True, bottom=True)
            ax.set_yticks([])
            ax.set_xticks([])

            circle = mpl.patches.Circle(
                (0, 0),
                norm_expression(plotdata_expression_clusters[cluster_id]) * 0.9 + 0.1,
                fc=cmap_expression(norm_expression(plotdata_expression_clusters[cluster_id])),
                lw=1,
                ec="#333333",
            )
            ax.add_patch(circle)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_aspect(1)

            label = cluster_info.loc[cluster_id, "label"]

            if show_n_cells and "n_cells" in cluster_info.columns:
                label += f" ({cluster_info.loc[cluster_id, 'n_cells']})"
            ax.text(
                1.05,
                0,
                label,
                ha="left",
                va="center",
                color="#333",
            )
            # text.set_path_effects(
            #     [
            #         mpl.patheffects.Stroke(linewidth=2, foreground="#333333"),
            #         mpl.patheffects.Normal(),
            #     ]
            # )

        if symbol is not None:
            ax.annotate(
                text="$\\mathit{" + symbol + "}$\nexpression",
                xy=(1, len(cluster_info) / 2),
                xytext=(5, 0),
                textcoords="offset points",
                xycoords="axes fraction",
                ha="left",
                va="center",
            )

    @classmethod
    def from_transcriptome(
        cls, transcriptome, clustering, gene, width=0.5, panel_height=0.5, cluster_info=None, layer=None, **kwargs
    ):
        import pandas as pd

        plotdata_expression_clusters = (
            pd.Series(transcriptome.get_X(gene, layer=layer), index=transcriptome.obs.index)
            .groupby(clustering.labels.values, observed=True)
            .mean()
        )

        if cluster_info is None:
            cluster_info = clustering.cluster_info

        return cls(
            plotdata_expression_clusters=plotdata_expression_clusters,
            cluster_info=cluster_info,
            width=width,
            panel_height=panel_height,
            **kwargs,
        )

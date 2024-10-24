import matplotlib as mpl
import seaborn as sns
import polyptich
import numpy as np

def get_cmap_rna():
    return mpl.cm.BuGn


def get_cmap_rna_diff():
    return mpl.cm.RdBu_r

class DifferentialExpression(polyptich.grid.Wrap):
    def __init__(
        self,
        plotdata_expression_clusters,
        cluster_info,
        width=0.5,
        panel_height=0.5,
        norm_expression=None,
        symbol=None,
        show_cluster=True,
        show_n_cells=True,
        order=False,
        relative_to = None,
        annotate_expression=False,
        **kwargs,
    ):
        super().__init__(ncol=1, **{"padding_height": 0, **kwargs})

        if relative_to is None:
            cmap_expression = get_cmap_rna()
            if norm_expression is None:
                norm_expression = mpl.colors.Normalize(
                    min(0.0, plotdata_expression_clusters.min()), plotdata_expression_clusters.max(), clip=True
                )
        else:
            cmap_expression = get_cmap_rna_diff()
            plotdata_expression_clusters = plotdata_expression_clusters - plotdata_expression_clusters.loc[relative_to]     
            norm_expression = mpl.colors.Normalize(np.log(0.25), np.log(4), clip=True)   

        plotdata_expression_clusters = plotdata_expression_clusters.loc[cluster_info.index]

        if order is True:
            self.order = plotdata_expression_clusters.sort_values(ascending=False).index
        elif order is not False:
            self.order = order
        else:
            self.order = plotdata_expression_clusters.index

        for cluster_id in self.order:
            panel, ax = polyptich.grid.Panel((width, panel_height))
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

            if show_cluster:
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

            if annotate_expression:
                text = ax.text(
                    0,
                    0,
                    f"{np.exp(plotdata_expression_clusters[cluster_id]):.2f}",
                    ha="center",
                    va="center",
                    color="#FFF",
                )
                text.set_path_effects(
                    [
                        mpl.patheffects.Stroke(linewidth=2, foreground="#33333388"),
                        mpl.patheffects.Normal(),
                    ]
                )

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
        cls, transcriptome, clustering, gene, width=0.5, panel_height=0.5, cluster_info=None, layer="counts", **kwargs
    ):
        import pandas as pd

        plotdata_expression_clusters = np.log(
            (np.exp(pd.Series(transcriptome.get_X(gene, layer=layer), index=transcriptome.obs.index))-1)
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

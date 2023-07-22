import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import chromatinhd
import scanpy as sc


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
        **kwargs,
    ):
        super().__init__(ncol=1, **{"padding_height": 0, **kwargs})

        plotdata_expression_clusters = plotdata_expression_clusters.loc[
            ~plotdata_expression_clusters.index.isin(["Plasma"])
        ]

        if norm_expression is None:
            norm_expression = mpl.colors.Normalize(
                0.0, plotdata_expression_clusters.max(), clip=True
            )

        cmap_expression = get_cmap_rna_diff()

        for cluster_id in cluster_info.index:
            panel, ax = chromatinhd.grid.Ax((width, panel_height))
            self.add(panel)

            sns.despine(ax=ax, left=True, right=True, top=True, bottom=True)
            ax.set_yticks([])
            ax.set_xticks([])

            circle = mpl.patches.Circle(
                (0, 0),
                norm_expression(plotdata_expression_clusters[cluster_id]) * 0.9 + 0.1,
                fc=cmap_expression(
                    norm_expression(plotdata_expression_clusters[cluster_id])
                ),
                lw=1,
                ec="#333333",
            )
            ax.add_patch(circle)
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_aspect(1)
            text = ax.text(
                1.05,
                0,
                cluster_info.loc[cluster_id, "label"],
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
        cls, transcriptome, clustering, gene, width=0.5, panel_height=0.5, **kwargs
    ):
        transcriptome.adata.obs["cluster"] = clustering.labels
        plotdata_expression = sc.get.obs_df(
            transcriptome.adata, [gene, "cluster"]
        ).rename(columns={gene: "expression"})
        plotdata_expression_clusters = plotdata_expression.groupby("cluster")[
            "expression"
        ].mean()
        cluster_info = clustering.cluster_info

        return cls(
            plotdata_expression_clusters=plotdata_expression_clusters,
            cluster_info=cluster_info,
            width=width,
            panel_height=panel_height,
            **kwargs,
        )

import chromatinhd
import chromatinhd.grid
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd


def get_cmap_atac_diff():
    return mpl.cm.RdBu_r


def get_norm_atac_diff():
    return mpl.colors.Normalize(np.log(1 / 4), np.log(4.0), clip=True)

class Differential(chromatinhd.grid.Wrap):
    def __init__(
        self,
        plotdata,
        plotdata_mean,
        cluster_info,
        width,
        window=None,
        panel_height=0.5,
        plotdata_empirical=None,
        show_atac_diff=True,
        cmap_atac_diff=mpl.cm.RdBu_r,
        norm_atac_diff=mpl.colors.Normalize(np.log(1 / 4), np.log(4.0), clip=True),
        ymax=20,
        **kwargs,
    ):
        super().__init__(ncol=1, **{"padding_height": 0, **kwargs})
        self.show_atac_diff = show_atac_diff
        self.cmap_atac_diff = cmap_atac_diff
        self.norm_atac_diff = norm_atac_diff
        self.window = window
        self.cluster_info = cluster_info

        # check plotdata
        plotdata = (
            plotdata.reset_index()
            .assign(coord=lambda x: x.coord.astype(int))
            .set_index(["cluster", "coord"])
        )
        plotdata_mean = (
            plotdata_mean.reset_index()
            .assign(coord=lambda x: x.coord.astype(int))
            .set_index(["coord"])
        )

        if "prob_diff" not in plotdata.columns:
            plotdata["prob_diff"] = plotdata["prob"] - plotdata_mean["prob"]

        if window is None:
            window = (
                plotdata.index.get_level_values("coord").min(),
                plotdata.index.get_level_values("coord").max(),
            )

        for cluster, cluster_info_oi in self.cluster_info.iterrows():
            panel, ax = chromatinhd.grid.Ax((width, panel_height))
            self.add(panel)

            ax.set_ylim(0, ymax)
            ax.set_xlim(*window)
            ax.axvline(0, dashes=(1, 1), color="#AAA", zorder=-1)

            text = ax.annotate(
                text=f"{cluster_info_oi['label']}",
                xy=(0, 1),
                xytext=(2, -2),
                textcoords="offset points",
                xycoords="axes fraction",
                ha="left",
                va="top",
                fontsize=10,
                color="#333",
            )
            text.set_path_effects(
                [
                    mpl.patheffects.Stroke(linewidth=2, foreground="white"),
                    mpl.patheffects.Normal(),
                ]
            )

            if plotdata_empirical is not None:
                # empirical distribution of atac-seq cuts
                plotdata_empirical_cluster = plotdata_empirical.query(
                    "cluster == @cluster"
                )
                ax.fill_between(
                    plotdata_empirical_cluster["coord"],
                    np.exp(plotdata_empirical_cluster["prob"]),
                    alpha=0.2,
                    color="#333",
                )

            ax.set_xticks([])

            if cluster != self.cluster_info.index[0]:
                ax.set_yticks([])
        ax.annotate(
            text="Mean &\ndifferential\naccessibility",
            xy=(0, len(cluster_info) / 2),
            xycoords="axes fraction",
            xytext=(-20, 0),
            textcoords="offset points",
            rotation=0,
            ha="right",
            va="center",
        )

        self.draw(plotdata, plotdata_mean)

    def draw(self, plotdata, plotdata_mean):
        self.artists = []

        for ax, cluster in zip(self.elements, self.cluster_info.index):
            ax = ax.ax
            if self.show_atac_diff:
                # posterior distribution of atac-seq cuts
                plotdata_cluster = plotdata.xs(cluster, level="cluster")
                (background,) = ax.plot(
                    plotdata_mean.index,
                    np.exp(plotdata_mean["prob"]),
                    color="black",
                    lw=1,
                    zorder=1,
                    linestyle="dashed",
                )
                (differential,) = ax.plot(
                    plotdata_cluster.index,
                    np.exp(plotdata_cluster["prob"]),
                    color="black",
                    lw=1,
                    zorder=1,
                )
                polygon = ax.fill_between(
                    plotdata_mean.index,
                    np.exp(plotdata_mean["prob"]),
                    np.exp(plotdata_cluster["prob"]),
                    color="black",
                    zorder=0,
                )

                # up/down gradient
                verts = np.vstack([p.vertices for p in polygon.get_paths()])
                c = plotdata_cluster["prob_diff"].values
                c[c == np.inf] = 0.0
                c[c == -np.inf] = -10.0
                gradient = ax.imshow(
                    c.reshape(1, -1),
                    cmap=self.cmap_atac_diff,
                    aspect="auto",
                    extent=[
                        verts[:, 0].min(),
                        verts[:, 0].max(),
                        verts[:, 1].min(),
                        verts[:, 1].max(),
                    ],
                    zorder=25,
                    norm=self.norm_atac_diff,
                )
                gradient.set_clip_path(polygon.get_paths()[0], transform=ax.transData)
                polygon.set_alpha(0)

                self.artists.extend([gradient, polygon, background, differential])

    def get_artists(self):
        return self.artists

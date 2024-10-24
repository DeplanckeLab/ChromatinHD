import chromatinhd
import polyptich.grid
import matplotlib as mpl
import matplotlib.patheffects
import numpy as np
import pandas as pd


def get_cmap_atac_diff():
    return mpl.cm.RdBu_r


def get_norm_atac_diff():
    return mpl.colors.Normalize(np.log(1 / 4), np.log(4.0), clip=True)


class Differential(polyptich.grid.Wrap):
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
        ymax=100,
        ylintresh=25,
        order=False,
        relative_to=None,
        label_accessibility=True,
        label_cluster=True,
        show_tss=True,
        **kwargs,
    ):
        """
        Parameters

        plotdata:
            dataframe with columns "coord", "prob", "cluster"
        plotdata_mean:
            dataframe with columns "coord", "prob"
        cluster_info:
            dataframe with columns "label"
        width:
            width of the plot
        window:
            window to show
        panel_height:
            height of each panel
        plotdata_empirical:
            dataframe with columns "coord", "prob", "cluster"
        show_atac_diff:
            show the differential accessibility
        cmap_atac_diff:
            colormap for the differential accessibility
        norm_atac_diff:
            normalization for the differential accessibility
        ymax:
            maximum y value
        ylintresh:
            linear threshold for the symlog scale
        order:
            order of the clusters
        relative_to:
            cluster or clusters to show the differential accessibility relative to
        """

        super().__init__(ncol=1, **{"padding_height": 0, **kwargs})
        self.show_atac_diff = show_atac_diff
        self.cmap_atac_diff = cmap_atac_diff
        self.norm_atac_diff = norm_atac_diff
        self.window = window
        self.cluster_info = cluster_info

        plotdata, plotdata_mean, order, window = _process_plotdata(
            plotdata,
            plotdata_mean,
            cluster_info,
            order,
            relative_to,
            window=window,
        )

        self.order = order

        for cluster, cluster_info_oi in self.cluster_info.loc[self.order].iterrows():
            panel, ax = polyptich.grid.Panel((width, panel_height))
            self.add(panel)

            _scale_differential(ax, ymax)
            _setup_differential(
                ax,
                ymax,
                cluster_info_oi,
                label=label_accessibility and (cluster == self.order[0]),
                label_cluster=label_cluster,
                show_tss=show_tss,
            )
            ax.set_xlim(*window)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if plotdata_empirical is not None:
                # empirical distribution of atac-seq cuts
                plotdata_empirical_cluster = plotdata_empirical.query("cluster == @cluster")
                ax.fill_between(
                    plotdata_empirical_cluster["coord"],
                    np.exp(plotdata_empirical_cluster["prob"]),
                    alpha=0.2,
                    color="#333",
                )

            ax.set_xticks([])

            # if cluster != self.order[0]:
            #     ax.set_yticklabels([])
            # elif label_accessibility is not False:
            #     ax.set_ylabel("Accessibility\nper 100 cells\nper 100bp", rotation=0, ha="right", va="center")

        self.draw(plotdata, plotdata_mean)

    def draw(self, plotdata, plotdata_mean):
        self.artists = []

        for ax, cluster in zip(self.elements, self.order):
            ax = ax.ax
            if self.show_atac_diff:
                # posterior distribution of atac-seq cuts
                plotdata_cluster = plotdata.xs(cluster, level="cluster")
                (background,) = ax.plot(
                    plotdata_mean.index,
                    np.exp(plotdata_mean["prob"]),
                    color="grey",
                    lw=0.5,
                    zorder=1,
                    # linestyle="dotted",
                )
                (differential,) = ax.plot(
                    plotdata_cluster.index,
                    np.exp(plotdata_cluster["prob"]),
                    color="black",
                    lw=0.5,
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

                # extra bar that shows the gradient in the top of the plot
                # ?

                self.artists.extend([gradient, polygon, background, differential])

    @classmethod
    def from_regionpositional(cls, region_id, regionpositional, width, relative_to=None, **kwargs):
        plotdata, plotdata_mean = regionpositional.get_plotdata(region_id, relative_to=relative_to)
        self = cls(
            plotdata=plotdata,
            plotdata_mean=plotdata_mean,
            width=width,
            **kwargs,
        )
        self.region_id = region_id
        self.region_ix = regionpositional.regions.coordinates.index.get_loc(region_id)
        return self

    def add_differential_slices(self, differential_slices):
        slicescores = differential_slices.get_slice_scores()
        slicescores = slicescores.loc[slicescores["region_ix"] == self.region_ix]
        for start, end, cluster_ix in zip(
            slicescores["start"],
            slicescores["end"],
            slicescores["cluster_ix"],
        ):
            ax = self.elements[self.order.get_loc(self.cluster_info.index[cluster_ix])].ax
            ax.axvspan(start, end, color="#33333333", zorder=0, lw=0)

    def get_artists(self):
        return self.artists


class DifferentialBroken(polyptich.grid.Wrap):
    def __init__(
        self,
        plotdata,
        plotdata_mean,
        cluster_info,
        breaking,
        window=None,
        panel_height=0.5,
        show_atac_diff=True,
        cmap_atac_diff=mpl.cm.RdBu_r,
        norm_atac_diff=mpl.colors.Normalize(np.log(1 / 4), np.log(4.0), clip=True),
        ymax=100,
        ylintresh=25,
        order=False,
        relative_to=None,
        label_accessibility=True,
        label_cluster=True,
        **kwargs,
    ):
        """
        Parameters

        plotdata:
            dataframe with columns "coord", "prob", "cluster"
        plotdata_mean:
            dataframe with columns "coord", "prob"
        cluster_info:
            dataframe with columns "label"
        breaking:
            breaking of the genome
        width:
            width of the plot
        window:
            window to show
        panel_height:
            height of each panel
        plotdata_empirical:
            dataframe with columns "coord", "prob", "cluster"
        show_atac_diff:
            show the differential accessibility
        cmap_atac_diff:
            colormap for the differential accessibility
        norm_atac_diff:
            normalization for the differential accessibility
        ymax:
            maximum y value
        order:
            order of the clusters
        relative_to:
            cluster or clusters to show the differential accessibility relative to
        """

        super().__init__(ncol=1, **{"padding_height": 0, **kwargs})
        self.show_atac_diff = show_atac_diff
        self.cmap_atac_diff = cmap_atac_diff
        self.norm_atac_diff = norm_atac_diff
        self.window = window
        self.cluster_info = cluster_info
        self.breaking = breaking

        if cluster_info is None:
            raise ValueError("cluster_info should not be None")
        plotdata = plotdata.query("cluster in @cluster_info.index")

        if order is True:
            self.order = plotdata.groupby(level=0).mean().sort_values(ascending=False).index
        elif order is False:
            self.order = cluster_info.index
        else:
            self.order = order

        plotdata, plotdata_mean, order, _ = _process_plotdata(plotdata, plotdata_mean, cluster_info, order, relative_to)

        self.order = order

        for cluster, cluster_info_oi in self.cluster_info.loc[self.order].iterrows():
            broken = self.add(
                polyptich.grid.Broken(breaking, height=panel_height, margin_top=0.0, padding_height=0.0)
            )

            panel, ax = broken[0, 0]
            _setup_differential(
                ax,
                ymax,
                cluster_info_oi,
                label=label_accessibility and (cluster == self.order[0]),
                label_cluster=label_cluster,
            )

            for panel, ax in broken:
                _scale_differential(ax, ymax, lintresh = ylintresh)
                ax.set_yticks([])
                ax.set_yticks([], minor=True)
                ax.axvline(0, dashes=(1, 1), color="#AAA", zorder=-1, lw=1)

        if self.show_atac_diff:
            self.draw(plotdata, plotdata_mean)

    def draw(self, plotdata, plotdata_mean):
        artists = []
        for (cluster, cluster_info_oi), broken in zip(self.cluster_info.loc[self.order].iterrows(), self):
            for (region, region_info), (panel, ax) in zip(self.breaking.regions.iterrows(), broken):
                plotdata_cluster = plotdata.xs(cluster, level="cluster")
                plotdata_cluster_break = plotdata_cluster.loc[
                    (plotdata_cluster.index.get_level_values("coord") >= region_info["start"])
                    & (plotdata_cluster.index.get_level_values("coord") <= region_info["end"])
                ]
                plotdata_mean_break = plotdata_mean.loc[
                    (plotdata_mean.index >= region_info["start"]) & (plotdata_mean.index <= region_info["end"])
                ]
                artists_cluster_region = _draw_differential(
                    ax, plotdata_cluster_break, plotdata_mean_break, self.cmap_atac_diff, self.norm_atac_diff
                )
                artists.extend(artists_cluster_region)
                ax.axvline(region_info["start"], color="#AAA", zorder=-1, lw=1)
                ax.axvline(region_info["end"], color="#AAA", zorder=-1, lw=1)
        return artists

    @classmethod
    def from_regionpositional(cls, region_id, regionpositional, breaking, cluster_info, relative_to=None, **kwargs):
        plotdata, plotdata_mean = regionpositional.get_plotdata(region_id, relative_to=relative_to)
        self = cls(
            plotdata=plotdata,
            plotdata_mean=plotdata_mean,
            cluster_info=cluster_info,
            breaking=breaking,
            **kwargs,
        )
        self.region_id = region_id
        self.region_ix = regionpositional.regions.coordinates.index.get_loc(region_id)
        return self

    def add_differential_slices(self, differential_slices):
        slicescores = differential_slices.get_slice_scores()
        slicescores = slicescores.loc[slicescores["region_ix"] == self.region_ix]

        for (cluster, cluster_info_oi), broken in zip(self.cluster_info.loc[self.order].iterrows(), self):
            for (region, region_info), (panel, ax) in zip(self.breaking.regions.iterrows(), broken):
                # find slicescores that (partially) overlap
                slicescores_oi = slicescores.loc[
                    ~((slicescores["start"] >= region_info["end"]) & (slicescores["end"] <= region_info["start"]))
                ]
                # slicescores_oi = slicescores.loc[(slicescores["start"] >= region_info["start"]) & (slicescores["end"] <= region_info["end"])]
                for start, end in zip(slicescores_oi["start"], slicescores_oi["end"]):
                    ax.axvspan(start, end, color="#33333333", zorder=0, lw=0)


def _draw_differential(ax, plotdata_cluster, plotdata_mean, cmap_atac_diff, norm_atac_diff):
    (background,) = ax.plot(
        plotdata_mean.index,
        np.exp(plotdata_mean["prob"]),
        color="grey",
        lw=0.5,
        zorder=1,
        # linestyle="dotted",
    )
    (differential,) = ax.plot(
        plotdata_cluster.index,
        np.exp(plotdata_cluster["prob"]),
        color="black",
        lw=0.5,
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

    return gradient, polygon, background, differential


def _scale_differential(ax, ymax, lintresh = 25):
    ax.set_yscale("symlog", linthresh=lintresh)
    ax.set_ylim(0, ymax)


def _setup_differential(ax, ymax, cluster_info_oi, label=False, label_cluster=True, show_tss=True):
    minor_ticks = np.array([2.5, 5, 7.5, 25, 50, 75, 250, 500])
    minor_ticks = minor_ticks[minor_ticks <= ymax]

    ax.set_yticks(minor_ticks, minor=True)
    ax.tick_params(axis="y", which="minor", length=2, color="#33333333")

    major_ticks = np.array([0, 10, 100])
    major_ticks = major_ticks[major_ticks <= ymax]
    ax.set_yticks(major_ticks, minor=False)

    if show_tss:
        ax.axvline(0, dashes=(1, 1), color="#AAA", zorder=-1, lw=1)

    if label_cluster:
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
            zorder=30,
        )
        text.set_path_effects(
            [
                mpl.patheffects.Stroke(linewidth=2, foreground="white"),
                mpl.patheffects.Normal(),
            ]
        )

    ax.set_xticks([])

    if label:
        ax.set_ylabel("Accessibility\nper 100 cells\nper 100bp", rotation=0, ha="right", va="center")
    else:
        ax.set_yticklabels([])
        ax.set_yticklabels([], minor=True)


def _process_plotdata(plotdata, plotdata_mean, cluster_info, order, relative_to, window=None):
    # check plotdata
    plotdata = plotdata.reset_index().assign(coord=lambda x: x.coord.astype(int)).set_index(["cluster", "coord"])
    plotdata_mean = plotdata_mean.reset_index().assign(coord=lambda x: x.coord.astype(int)).set_index(["coord"])

    # determine relative to
    # if "prob_diff" not in plotdata.loc[self.order].columns:
    if relative_to is None:
        plotdata["prob_diff"] = plotdata["prob"] - plotdata_mean["prob"]
    elif isinstance(relative_to, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        plotdata["prob_diff"] = (
            plotdata["prob"]
            - plotdata.loc[relative_to]
            .groupby(level="coord")
            .mean()["prob"][plotdata.index.get_level_values("coord")]
            .values
        )
        plotdata_mean = plotdata.loc[relative_to].groupby(level="coord").mean()
    else:
        plotdata["prob_diff"] = plotdata["prob"] - plotdata.loc[relative_to]["prob"]
        plotdata_mean = plotdata.loc[relative_to]

    # subset on requested clusters
    plotdata = plotdata.query("cluster in @cluster_info.index")

    # determine order
    if order is True:
        order = plotdata.groupby(level=0).mean().sort_values(ascending=False).index
    elif order is False:
        order = cluster_info.index
    else:
        order = order

    if window is None:
        window = (
            plotdata.index.get_level_values("coord").min(),
            plotdata.index.get_level_values("coord").max(),
        )
    else:
        if isinstance(window, pd.Series):
            window = window.values.tolist()
        plotdata = plotdata.loc[
            (plotdata.index.get_level_values("coord") >= window[0])
            & (plotdata.index.get_level_values("coord") <= window[1])
        ]
        plotdata_mean = plotdata_mean.loc[(plotdata_mean.index >= window[0]) & (plotdata_mean.index <= window[1])]

    return plotdata, plotdata_mean, order, window

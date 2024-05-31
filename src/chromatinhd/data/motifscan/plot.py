import matplotlib as mpl
import numpy as np
import pandas as pd

from chromatinhd.grid import Grid, Panel, Broken
from chromatinhd.utils import indptr_to_indices


class Motifs(Grid):
    def __init__(self, motifscan, gene, motifs_oi=None, window=None, width=1.0, panel_height=0.1):
        """
        Plot the location of motifs in a region
        """

        super().__init__()

        region_ix = motifscan.regions.coordinates.index.get_loc(gene)

        # get motif data
        positions, indices = motifscan.get_slice(
            region_ix=region_ix, start=window[0], end=window[1], return_scores=False, return_strands=False
        )

        # get motifs oi
        if motifs_oi is None:
            motifs_oi = motifscan.motifs
        else:
            if not isinstance(motifs_oi, pd.DataFrame):
                raise ValueError("motifs_oi should be a dataframe")
            elif not motifs_oi.index.isin(motifscan.motifs.index).all():
                raise ValueError("motifs_oi should be a dataframe with indices in motifscan.motifs")

        if "color" not in motifs_oi.columns:
            motifs_oi["color"] = [mpl.colors.rgb2hex(x) for x in mpl.cm.tab20(np.arange(motifs_oi.shape[0]) % 20)]

        motifscan.motifs["ix"] = np.arange(motifscan.motifs.shape[0])
        motifdata = []
        for motif in motifs_oi.index:
            motif_ix = motifscan.motifs.index.get_loc(motif)
            positions_oi = positions[indices == motif_ix]
            motifdata.extend([{"position": pos, "motif": motif} for pos in positions_oi])
        motifdata = self.motifdata = pd.DataFrame(motifdata, columns=["position", "motif"])

        for motif in motifs_oi.itertuples():
            panel, ax = self.add_under(Panel((width, panel_height)), padding=0)
            ax.set_xlim(window)
            ax.set_xticks([])
            ax.set_yticks([])

            if "label" in motif._fields:
                label = motif.label
            else:
                label = motif.Index

            ax.annotate(
                label,
                (1.0, 0.5),
                xycoords="axes fraction",
                xytext=(2, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                ha="left",
                color=motif.color,
            )

            # add motifs
            motif_id = motif.Index
            plotdata = motifdata.loc[motifdata["motif"] == motif_id].copy()

            if len(plotdata) > 0:
                ax.scatter(
                    plotdata["position"],
                    [1] * len(plotdata),
                    transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                    marker="v",
                    color=motif.color,
                    alpha=1,
                    s=100,
                    zorder=20,
                )


class GroupedMotifs(Grid):
    def __init__(
        self,
        motifscan,
        gene,
        motifs_oi,
        window=None,
        width=1.0,
        panel_height=0.1,
        label_motifs=True,
        label_motifs_side="right",
        group_info=None,
    ):
        """
        Plot the location of motifs in a region.
        """

        super().__init__()

        motifs_oi, group_info, motifdata = _process_grouped_motifs(
            gene, motifs_oi, motifscan, group_info=group_info, window=window
        )

        for group, group_info_oi in group_info.iterrows():
            group_motifs = motifs_oi.query("group == @group")
            panel, ax = self.add_under(Panel((width, panel_height)), padding=0)
            ax.set_xlim(window)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.axis("off")
            color = group_motifs["color"].iloc[0] if group_motifs.shape[0] == 1 else "black"
            ax.axhspan(0, 1, color=color, zorder=0, alpha=0.1, transform=ax.transAxes, lw=0)

            if label_motifs:
                if label_motifs_side == "right":
                    xy = (1.0, 0.5)
                    ha = "left"
                    xytext = (2, 0)
                else:
                    xy = (0.0, 0.5)
                    ha = "right"
                    xytext = (-2, 0)
                ax.annotate(
                    group_info_oi["label"],
                    xy,
                    xycoords="axes fraction",
                    xytext=xytext,
                    textcoords="offset points",
                    color=group_motifs["color"].iloc[0] if group_motifs.shape[0] == 1 else "black",
                    va="center",
                    fontsize=9,
                    ha=ha,
                )

                # ax.set_yticks(range(len(group_motifs)))
                # ax.set_yticklabels(group_motifs["label"].tolist(), fontsize=9)
                # ax.tick_params(axis="y", which="major", pad=0.5)

                # rainbow_text(
                #     ax=ax,
                #     x=1.0,
                #     y=0.5,
                #     strings=group_motifs["label"].tolist(),
                #     colors=group_motifs["color"].tolist(),
                # )

            # plot the motifs
            for motif in group_motifs.itertuples():
                # add motifs
                motif_id = motif.Index
                plotdata = motifdata.loc[motifdata["motif"] == motif_id].copy()

                if len(plotdata) > 0:
                    ax.scatter(
                        plotdata["position"],
                        [1] * len(plotdata),
                        transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                        marker="v",
                        color=motif.color,
                        alpha=1,
                        s=100,
                        zorder=20,
                    )


def _process_grouped_motifs(gene, motifs_oi, motifscan, window=None, group_info=None):
    region_ix = motifscan.regions.coordinates.index.get_loc(gene)

    # get motif data
    if window is not None:
        positions, indices = motifscan.get_slice(
            region_ix=region_ix, start=window[0], end=window[1], return_scores=False, return_strands=False
        )
    else:
        positions, indices = motifscan.get_slice(region_ix=region_ix, return_scores=False, return_strands=False)

    # check motifs oi
    if not isinstance(motifs_oi, pd.DataFrame):
        raise ValueError("motifs_oi should be a dataframe")
    elif not motifs_oi.index.isin(motifscan.motifs.index).all():
        raise ValueError("motifs_oi should be a dataframe with indices in motifscan.motifs")
    elif "group" not in motifs_oi.columns:
        motifs_oi["group"] = motifs_oi.index
        # raise ValueError("motifs_oi should have a 'group' column")

    # get group info
    if group_info is None:
        group_info = motifs_oi.groupby("group").first()[[]]
        group_info = group_info.loc[motifs_oi["group"].unique()]
        group_info["label"] = group_info.index

    if "label" not in motifs_oi.columns:
        motifs_oi["label"] = motifs_oi["group"]

    if "color" not in motifs_oi.columns:
        group_info["color"] = [mpl.colors.rgb2hex(x) for x in mpl.cm.tab20(np.arange(group_info.shape[0]) % 20)]
        motifs_oi["color"] = motifs_oi["group"].map(group_info["color"].to_dict())

    motifscan.motifs["ix"] = np.arange(motifscan.motifs.shape[0])
    motifdata = []
    for motif in motifs_oi.index:
        motif_ix = motifscan.motifs.index.get_loc(motif)
        positions_oi = positions[indices == motif_ix]
        motifdata.extend([{"position": pos, "motif": motif} for pos in positions_oi])
    motifdata = pd.DataFrame(motifdata, columns=["position", "motif"])
    return motifs_oi, group_info, motifdata


class GroupedMotifsBroken(Grid):
    def __init__(self, motifscan, gene, motifs_oi, breaking, group_info=None, panel_height=0.1):
        """
        Plot the location of motifs in a region.
        """

        super().__init__()

        motifs_oi, group_info, motifdata = _process_grouped_motifs(gene, motifs_oi, motifscan, group_info=group_info)

        for group, group_info_oi in group_info.iterrows():
            broken = self.add_under(
                Broken(breaking, height=panel_height, margin_height=0.0, padding_height=0.01), padding=0
            )
            group_motifs = motifs_oi.query("group == @group")

            panel, ax = broken[0, -1]
            _setup_group(ax, group_info_oi, group_motifs)

            color = group_motifs["color"].iloc[0] if group_motifs.shape[0] == 1 else "black"

            for panel, ax in broken:
                ax.axis("off")
                ax.axhspan(0, 1, color=color, zorder=0, alpha=0.1, transform=ax.transAxes, lw=0)

            # plot the motifs
            for motif in group_motifs.itertuples():
                # add motifs
                motif_id = motif.Index

                plotdata = motifdata.loc[motifdata["motif"] == motif_id].copy()
                for (region, region_info), (panel, ax) in zip(breaking.regions.iterrows(), broken):
                    plotdata_region = plotdata.loc[
                        (plotdata["position"] >= region_info["start"]) & (plotdata["position"] <= region_info["end"])
                    ]
                    if len(plotdata_region) > 0:
                        ax.scatter(
                            plotdata_region["position"],
                            [1] * len(plotdata_region),
                            transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                            marker="v",
                            color=motif.color,
                            alpha=1,
                            s=100,
                            zorder=20,
                        )


def _setup_group(ax, group_info_oi, group_motifs):
    ax.set_xticks([])
    ax.set_yticks([])

    if "label" in group_info_oi.keys():
        color = group_motifs["color"].tolist()[0] if group_motifs.shape[0] == 1 else "black"
        ax.text(s=group_info_oi["label"], color=color, x=1.0, y=0.0, transform=ax.transAxes)
    else:
        rainbow_text(
            ax=ax,
            x=1.0,
            y=0.0,
            strings=group_motifs["label"].tolist(),
            colors=group_motifs["color"].tolist(),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )


def rainbow_text(x, y, strings, colors, ax, transform=None, **kw):
    """
    Take a list of ``strings`` and ``colors`` and place them next to each
    other, with text strings[i] being shown in colors[i].

    This example shows how to do both vertical and horizontal text, and will
    pass all keyword arguments to plt.text, so you can set the font size,
    family, etc.

    The text will get added to the ``ax`` axes, if provided, otherwise the
    currently active axes will be used.
    """
    if transform is None:
        transform = ax.transData
    # canvas = ax.figure.canvas

    t = transform

    prev = None

    for s, c in zip(strings, colors):
        if prev is None:
            prev = ax.text(x, y, s + " ", color=c, transform=t, **kw)
        else:
            prev = ax.annotate(s + " ", xycoords=prev, xy=(1, 0), color=c, **kw)
        # text = ax.text(x, y, s + " ", color=c, transform=t, **kw)
        # text.draw(canvas.get_renderer())
        # ex = text.get_window_extent()
        # print(ex.width)
        # t = mpl.transforms.offset_copy(text._transform, x=ex.width, units="dots")

import matplotlib as mpl
import numpy as np
import pandas as pd

from polyptich.grid import Grid, Panel, Broken
from chromatinhd.utils import indptr_to_indices


class Motifs(Grid):
    def __init__(
        self, motifscan, gene, motifs_oi=None, window=None, width=1.0, panel_height=0.1
    ):
        """
        Plot the location of motifs in a region
        """

        super().__init__()

        region_ix = motifscan.regions.coordinates.index.get_loc(gene)

        # get motif data
        positions, indices = motifscan.get_slice(
            region_ix=region_ix,
            start=window[0],
            end=window[1],
            return_scores=False,
            return_strands=False,
        )

        # get motifs oi
        if motifs_oi is None:
            motifs_oi = motifscan.motifs
        else:
            if not isinstance(motifs_oi, pd.DataFrame):
                raise ValueError("motifs_oi should be a dataframe")
            elif not motifs_oi.index.isin(motifscan.motifs.index).all():
                raise ValueError(
                    "motifs_oi should be a dataframe with indices in motifscan.motifs"
                )

        if "color" not in motifs_oi.columns:
            motifs_oi["color"] = [
                mpl.colors.rgb2hex(x)
                for x in mpl.cm.tab20(np.arange(motifs_oi.shape[0]) % 20)
            ]

        motifscan.motifs["ix"] = np.arange(motifscan.motifs.shape[0])
        motifdata = []
        for motif in motifs_oi.index:
            motif_ix = motifscan.motifs.index.get_loc(motif)
            positions_oi = positions[indices == motif_ix]
            motifdata.extend([{"position": pos, "motif": motif} for pos in positions_oi])
        motifdata = self.motifdata = pd.DataFrame(
            motifdata, columns=["position", "motif"]
        )

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
                    transform=mpl.transforms.blended_transform_factory(
                        ax.transData, ax.transAxes
                    ),
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

        motifs_oi, group_info = _process_grouped_motifs_oi(
            motifs_oi, motifscan, group_info=group_info
        )

        motifdata = _process_grouped_motifs(
            gene, motifs_oi, motifscan, group_info=group_info, window=window
        )

        for group, group_info_oi in group_info.iterrows():
            group_motifs = motifs_oi.query("group == @group")
            panel, ax = self.add_under(Panel((width, panel_height)), padding=0)
            ax.set_xlim(window)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.axis("off")
            color = (
                group_motifs["color"].iloc[0]
                if "color" in group_motifs.columns
                else "black"
            )
            ax.axhspan(
                0, 1, color=color, zorder=0, alpha=0.1, transform=ax.transAxes, lw=0
            )
            ax.set_ylim(0.1, 1.1)

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
                    color=group_motifs["color"].iloc[0]
                    if group_motifs.shape[0] == 1
                    else "black",
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
                        transform=mpl.transforms.blended_transform_factory(
                            ax.transData, ax.transAxes
                        ),
                        marker="v",
                        color=motif.color,
                        alpha=1,
                        s=100,
                        zorder=20,
                    )


def blend_with_white(color, alpha):
    # Convert color to RGB if it's in HEX format
    if isinstance(color, str):
        color = mpl.colors.to_rgb(color)

    # Define the white background color
    white = (1, 1, 1)

    # Blend the color with white using the alpha value
    blended_color = tuple(alpha * c + (1 - alpha) * w for c, w in zip(color, white))

    return blended_color

def _process_grouped_motifs_oi(
    motifs_oi, motifscan, group_info=None
):
    # check motifs oi
    if not isinstance(motifs_oi, pd.DataFrame):
        raise ValueError("motifs_oi should be a dataframe")
    elif not motifs_oi.index.isin(motifscan.motifs.index).all():
        raise ValueError(
            "motifs_oi should be a dataframe with indices in motifscan.motifs"
        )
    elif "group" not in motifs_oi.columns:
        motifs_oi["group"] = motifs_oi.index

    # get group info
    if group_info is None:
        group_info = motifs_oi.groupby("group").first()[[]]
        group_info = group_info.loc[motifs_oi["group"].unique()]
        group_info["label"] = group_info.index

    if "label" not in motifs_oi.columns:
        motifs_oi["label"] = motifs_oi["group"]

    if "color" not in motifs_oi.columns.tolist():
        if "color" not in group_info.columns:
            group_info["color"] = [
                mpl.colors.rgb2hex(x)
                for x in mpl.cm.tab20(np.arange(group_info.shape[0]) % 20)
            ]
        motifs_oi["color"] = motifs_oi["group"].map(group_info["color"].to_dict())
    return motifs_oi, group_info

def _process_grouped_motifs(
    gene, motifs_oi, motifscan, window=None, group_info=None, slices_oi=None, return_strands = False, prune = 20,
):
    region_ix = motifscan.regions.coordinates.index.get_loc(gene)

    if prune is not False:
        return_strands = True

    # get motif data
    if window is not None:
        positions, indices, scores, strands = motifscan.get_slice(
            region_ix=region_ix,
            start=window[0],
            end=window[1],
            return_scores=True,
            return_strands=True,
        )
    else:
        positions, indices, scores, strands = motifscan.get_slice(
            region_ix=region_ix, return_scores=True, return_strands=True,
        )

    motifscan.motifs["ix"] = np.arange(motifscan.motifs.shape[0])
    motifdata = []
    for motif, group in zip(motifs_oi.index, motifs_oi["group"]):
        motif_ix = motifscan.motifs.index.get_loc(motif)
        positions_oi = positions[indices == motif_ix]
        if return_strands:
            strands_oi = strands[indices == motif_ix]
            scores_oi = scores[indices == motif_ix]
            motifdata.extend(
                [
                    {"position": pos, "motif": motif, "group": group, "strand": strand, "score":score}
                    for pos, strand, score in zip(positions_oi, strands_oi, scores_oi)
                ]
            )
        else:
            motifdata.extend(
                [{"position": pos, "motif": motif, "group": group,} for pos in positions_oi]
            )
        
    motifdata = pd.DataFrame(motifdata, columns=["position", "motif", "group", "strand", "score"]) if return_strands else pd.DataFrame(motifdata, columns=["position", "motif", "group"])
    motifdata = motifdata.sort_values("position", ascending = True)

    # check slices oi
    if slices_oi is not None:
        assert "cluster" in slices_oi.columns
        assert "start" in slices_oi.columns
        assert "end" in slices_oi.columns
        assert "region" in slices_oi.columns

        slices_oi = slices_oi.loc[slices_oi["region"] == gene]

        padding = 20

        motifdata["oi"] = (
            (
                motifdata["position"].values[None, :]
                >= slices_oi["start"].values[:, None] - padding
            )
            & (
                motifdata["position"].values[None, :]
                <= slices_oi["end"].values[:, None] + padding
            )
        ).any(axis=0)

    # prune
    if (prune is not False) and len(motifdata):
        if "score" not in motifdata.columns:
            raise ValueError("motifdata should have a 'score' column to prune, set return_strands = True")

        # go over each group and delete motifs that are too close
        motifdata_pruned = []
        for _, group in motifdata.groupby("group"):
            group["distance"] = group["position"].diff().fillna(0)

            group["section"] = (group["distance"] > prune).cumsum()
            group = group.sort_values("score", ascending = False).groupby("section", as_index = False).first()
            motifdata_pruned.append(group)
        motifdata = pd.concat(motifdata_pruned)

    return motifdata


def intersect_positions_slices(positions, slices_start, slices_end):
    """
    Given a list of positions and a list of slices, return a list of positions that intersect with the slices.
    """
    positions = np.array(positions)
    slices_start = np.array(slices_start)
    slices_end = np.array(slices_end)

    intersected = (positions[:, None] >= slices_start[None, :]) & (
        positions[:, None] <= slices_end[None, :]
    )
    return intersected


class GroupedMotifsBroken(Grid):
    def __init__(
        self,
        motifscan,
        gene,
        motifs_oi,
        breaking,
        group_info=None,
        panel_height=0.1,
        slices_oi=None,
        show_triangle:bool=True,
        show_bar:bool=True,
    ):
        """
        Plot the location of motifs in a region.

        Parameters
        ----------
        motifscan : MotifScan
            MotifScan object
        gene : str
            Gene name
        motifs_oi : pd.DataFrame
            Dataframe with motifs to plot. Should have a 'group' column.
        breaking : Breaking
            Breaking object
        group_info : pd.DataFrame
            Dataframe with group information. Should have a 'label' column.
        panel_height : float
            Height in inches of each motif group line
        slices_oi : pd.DataFrame
            Dataframe with slices information. Should have 'start', 'end', 'cluster', and 'region' columns.
        show_triangle : bool
            Show triangle markers
        show_bar : bool
            Show bar markers
        
        """

        super().__init__()

        motifs_oi, group_info = _process_grouped_motifs_oi(
            motifs_oi, motifscan, group_info=group_info, 
        )

        motifdatas = [_process_grouped_motifs(
            gene, motifs_oi, motifscan, group_info=group_info, window=[region["start"], region["end"]], slices_oi = slices_oi
        ) for _, region in breaking.regions.iterrows()]

        for group, group_info_oi in group_info.iterrows():
            broken = self.add_under(
                Broken(
                    breaking, height=panel_height, margin_top=0.0, padding_height=0.01
                ),
                padding=0,
            )
            group_motifs = motifs_oi.query("group == @group")

            panel, ax = broken[0, 0]
            _setup_group(ax, group_info_oi, group_motifs)

            if "color" in group_info_oi.index:
                color = group_info_oi["color"]
            elif "color" in group_motifs.columns:
                color = group_motifs["color"].iloc[0]
            else:
                color = "black"
            for panel, ax in broken:
                ax.axis("off")
                ax.axhspan(
                    0, 1, color=color, zorder=0, alpha=0.1, transform=ax.transAxes, lw=0
                )

            # create marker
            # create a very narrow triangle
            marker = mpl.path.Path(
                [
                    [-0.5, 1.0],
                    [0.5, 1.],
                    [0.0, 0],
                    [-0.5, 1.],
                ]
            )
            marker = mpl.path.Path(
                [
                    [-0.5, 0],
                    [0.5, 0],
                    [0.0, 1.],
                    [-0.5, 0],
                ]
            )

            # plot the motifs
            for (region, region_info), (panel, ax), motifdata in zip(
                breaking.regions.iterrows(), broken, motifdatas
            ):
                motifdata_region = motifdata.loc[
                    (motifdata["motif"].isin(group_motifs.index))
                ]

                # remove duplicates from the same group
                motifdata_region = motifdata_region.drop_duplicates(
                    keep="first", subset=["position"]
                )

                plotdata = motifdata_region

                if len(plotdata) > 0:
                    if "oi" not in plotdata.columns:
                        plotdata["oi"] = True
                    if "oi" in plotdata.columns:
                        plotdata_oi = plotdata.loc[plotdata["oi"]]
                        plotdata_not_oi = plotdata.loc[~plotdata["oi"]]

                        # join very close
                        # plotdata_oi = plotdata_oi.sort_values("position")
                        # plotdata_oi["distance_to_next"] = plotdata_oi["position"].diff().fillna(0.)
                        # plotdata_oi["group"] = np.cumsum(plotdata_oi["distance_to_next"] > 50)
                        # plotdata_oi["n"] = 1
                        # plotdata_oi = plotdata_oi.groupby("group").agg(
                        #     {"position": "mean", "n": "sum", "oi": "first"}
                        # )

                        # oi
                        if show_triangle:
                            # front
                            ax.scatter(
                                plotdata_oi["position"],
                                [0.] * len(plotdata_oi),
                                transform=mpl.transforms.blended_transform_factory(
                                    ax.transData, ax.transAxes
                                ),
                                marker = marker,
                                color=color,
                                s=100,
                                zorder=20,
                                lw = 0.
                            )
                            # background white
                            ax.scatter(
                                plotdata_oi["position"],
                                [1.] * len(plotdata_oi),
                                transform=mpl.transforms.blended_transform_factory(
                                    ax.transData, ax.transAxes
                                ),
                                marker = marker,
                                color="white",
                                s=400,
                                zorder=19,
                                lw = 1.
                            )
                            # other not oi
                            ax.scatter(
                                plotdata_not_oi["position"],
                                [0.] * len(plotdata_not_oi),
                                transform=mpl.transforms.blended_transform_factory(
                                    ax.transData, ax.transAxes
                                ),
                                marker = marker,
                                color=blend_with_white(color, 0.3),
                                s=100,
                                zorder=18,
                                lw = 0.
                            )
                        if show_bar:
                            # front
                            ax.scatter(
                                plotdata_oi["position"],
                                [1.] * len(plotdata_oi),
                                transform=mpl.transforms.blended_transform_factory(
                                    ax.transData, ax.transAxes
                                ),
                                # marker="v",
                                marker = "|",
                                color=color,
                                s=200,
                                zorder=22,
                                lw = 1.5
                            )

                            # white back
                            ax.scatter(
                                plotdata_oi["position"],
                                [1.] * len(plotdata_oi),
                                transform=mpl.transforms.blended_transform_factory(
                                    ax.transData, ax.transAxes
                                ),
                                marker = marker,
                                color=blend_with_white(color, 0.3),
                                s=250,
                                zorder=19,
                                lw=0.0,
                                edgecolor="white",
                            )
                            ax.scatter(
                                plotdata_not_oi["position"],
                                [1.] * len(plotdata_not_oi),
                                transform=mpl.transforms.blended_transform_factory(
                                    ax.transData, ax.transAxes
                                ),
                                # marker="v",
                                marker = "|",
                                color=blend_with_white(color, 0.3),
                                s=200,
                                zorder=18,
                                lw = 1.
                            )



def _setup_group(ax, group_info_oi, group_motifs):
    ax.set_xticks([])
    ax.set_yticks([])

    if "label" in group_info_oi.keys():
        color = group_motifs["color"].tolist()[0]
        ax.text(
            s=group_info_oi["label"],
            color=color,
            x=0.0,
            y=0.0,
            ha="right",
            transform=ax.transAxes,
            fontsize=8,
        )
    else:
        rainbow_text(
            ax=ax,
            x=0.0,
            y=0.0,
            strings=group_motifs["label"].tolist(),
            colors=group_motifs["color"].tolist(),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )


def rainbow_text(x, y, strings, colors, ax, transform=None, ha="right", **kw):
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
            prev = ax.text(x, y, s + " ", color=c, transform=t, ha=ha, **kw)
        else:
            if ha == "right":
                prev = ax.annotate(
                    s + " ", xycoords=prev, xy=(0, 0), color=c, ha=ha, **kw
                )
            elif ha == "left":
                prev = ax.annotate(
                    s + " ", xycoords=prev, xy=(1.0, 0), color=c, ha=ha, **kw
                )
        # text = ax.text(x, y, s + " ", color=c, transform=t, **kw)
        # text.draw(canvas.get_renderer())
        # ex = text.get_window_extent()
        # print(ex.width)
        # t = mpl.transforms.offset_copy(text._transform, x=ex.width, units="dots")

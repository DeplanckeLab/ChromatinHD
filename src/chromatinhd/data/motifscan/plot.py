import matplotlib as mpl
import numpy as np
import pandas as pd

from chromatinhd.grid import Grid, Panel
from chromatinhd.utils import indptr_to_indices


class Motifs(Grid):
    def __init__(self, motifscan, gene, motifs_oi=None, width=1.0, panel_height=0.1):
        """
        Plot the location of motifs in a region
        """

        super().__init__()

        window = motifscan.regions.window

        region_ix = motifscan.regions.coordinates.index.get_loc(gene)

        # get motif data
        window_width = window[1] - window[0]
        indptr_start = region_ix * window_width
        indptr_end = (region_ix + 1) * window_width
        motif_indices = motifscan.indices[motifscan.indptr[indptr_start] : motifscan.indptr[indptr_end]]
        position_indices = indptr_to_indices(motifscan.indptr[indptr_start : indptr_end + 1])

        # get motifs oi
        if motifs_oi is None:
            motifs_oi = motifscan.motifs
        else:
            if not isinstance(motifs_oi, pd.DataFrame):
                raise ValueError("motifs_oi should be a dataframe")
            elif not motifs_oi.index.isin(motifscan.motifs.index).all():
                raise ValueError("motifs_oi should be a dataframe with indices in motifscan.motifs")

        motifscan.motifs["ix"] = np.arange(motifscan.motifs.shape[0])
        motifdata = []
        for motif in motifs_oi.index:
            motif_ix = motifscan.motifs.loc[motif, "ix"]
            positions_oi = position_indices[motif_indices == motif_ix]
            motifdata.extend([{"position": pos + window[0], "motif": motif} for pos in positions_oi])
        motifdata = self.motifdata = pd.DataFrame(motifdata, columns=["position", "motif"])

        for motif in motifs_oi.itertuples():
            panel, ax = self.add_under(Panel((width, panel_height)), padding=0)
            ax.set_xlim(window)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.annotate(
                motif.Index,
                (1.0, 0.5),
                xycoords="axes fraction",
                xytext=(2, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                ha="left",
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
                    # color=plotdata["color"],
                    alpha=1,
                    s=100,
                    zorder=20,
                )

from typing import List
from .motifscan import Motifscan
import numpy as np


class BinnedMotifCounts:
    """
    Provides binned motif counts per fragment
    """

    def __init__(
        self,
        motifscan: Motifscan,
        motif_binsizes: List[int],
        fragment_binsizes: List[int],
    ):
        self.motifscan = motifscan
        self.region_width = motifscan.regions.window[1] - motifscan.regions.window[0]
        self.motif_binsizes = motif_binsizes
        self.fragment_binsizes = fragment_binsizes
        self.n_motifs = motifscan.motifs.shape[0]

        n_genes = len(motifscan.regions.coordinates)

        self.fragment_widths = []
        self.motifcount_sizes = []
        self.motif_widths = []
        self.fragmentprob_sizes = []
        width = self.region_width
        for motif_binsize, fragment_binsize in zip(self.motif_binsizes, self.fragment_binsizes):
            assert fragment_binsize % motif_binsize == 0, (
                "motif_binsize must be a multiple of fragment_binsize",
                motif_binsize,
                fragment_binsize,
            )
            self.motifcount_sizes.append(width // motif_binsize)
            self.motif_widths.append(self.region_width // motif_binsize)
            self.fragmentprob_sizes.append(width // fragment_binsize)
            self.fragment_widths.append(self.region_width // fragment_binsize)
            width = fragment_binsize

        precomputed = []
        for motif_binsize, motif_width, fragment_binsize, fragment_width in zip(
            self.motif_binsizes,
            self.motif_widths,
            self.fragment_binsizes,
            [1, *self.fragment_widths[:-1]],
        ):
            precomputed.append(
                np.bincount(motifscan.positions // motif_binsize, minlength=(n_genes * motif_width)).reshape(
                    (n_genes * fragment_width, -1)
                )
            )
        self.precomputed = precomputed

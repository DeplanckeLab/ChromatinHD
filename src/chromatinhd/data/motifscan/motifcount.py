from typing import List, Union
from .motifscan import Motifscan
from .view import MotifscanView
import numpy as np
import tqdm.auto as tqdm


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
        self.width = motifscan.regions.width
        assert self.width is not None, "Regions must have a width"
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


class BinnedMotifCounts:
    """
    Provides binned motif counts per fragment
    """

    def __init__(
        self,
        motifscan: Union[Motifscan, MotifscanView],
        binsize: int,
    ):
        self.motifscan = motifscan
        self.width = motifscan.regions.width
        assert self.width is not None, "Regions must have a width"
        self.binsize = binsize
        self.n_motifs = motifscan.motifs.shape[0]

        n_regions = motifscan.regions.n_regions
        motif_width = motifscan.regions.width // binsize

        precomputed = np.zeros((n_regions, motif_width), dtype=np.int32)

        for region_ix in tqdm.tqdm(range(n_regions)):
            if isinstance(motifscan, Motifscan):
                indptr_start = motifscan.region_indptr[region_ix]
                indptr_end = motifscan.region_indptr[region_ix + 1]
                coordinates = motifscan.coordinates[indptr_start:indptr_end]
            else:
                indptr_start, indptr_end = motifscan.region_indptr[region_ix]
                coordinates = (
                    motifscan.coordinates[indptr_start:indptr_end] - motifscan.regions.region_starts[region_ix]
                )
            precomputed[region_ix] = np.bincount(
                coordinates // binsize,
                minlength=(motif_width),
            )
        self.precomputed = precomputed

        # self.fragment_binsizes = fragment_binsizes
        # self.n_motifs = motifscan.motifs.shape[0]

        # self.fragment_widths = []
        # self.fragmentprob_sizes = []
        # width = self.region_width
        # for fragment_binsize in zip(self.fragment_binsizes):
        #     self.fragmentprob_sizes.append(width // fragment_binsize)
        #     self.fragment_widths.append(self.region_width // fragment_binsize)
        #     width = fragment_binsize

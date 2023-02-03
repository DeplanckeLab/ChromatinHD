import sys
import numpy as np
import pyximport

pyximport.install(
    reload_support=True,
    language_level=3,
    setup_args=dict(include_dirs=[np.get_include()]),
)
import chromatinhd.loaders.extraction.fragments
import chromatinhd.loaders.extraction.motifs

import torch
from multiprocessing import shared_memory

import dataclasses

from .fragments import Fragments

from .fragments import Result


@dataclasses.dataclass
class FullResult(Result):
    motifcounts: torch.Tensor
    n_motifs: int


class Full(Fragments):
    def __init__(self, fragments, motifscan, cellxgene_batch_size, window, cutwindow):
        super().__init__(fragments, cellxgene_batch_size, window)

        # store auxilliary information
        self.cutwindow = cutwindow
        self.cutwindow_width = cutwindow[1] - cutwindow[0]

        # create buffers for motifs
        n_motifs = motifscan.shape[1]
        n_motifs_per_fragment = 1000  # 400 motifs
        motif_buffer_size = self.fragment_buffer_size * n_motifs_per_fragment

        self.motifscan_indptr = motifscan.indptr.astype(np.int64)
        self.motifscan_indices = motifscan.indices.astype(np.int64)
        self.motifscan_data = motifscan.data.astype(np.float64)

        self.out_fragment_indptr = torch.from_numpy(
            np.zeros(motif_buffer_size, dtype=int)
        )  # .pin_memory()
        self.out_motif_ix = torch.from_numpy(
            np.zeros(motif_buffer_size, dtype=int)
        )  # .pin_memory()
        self.out_score = torch.from_numpy(
            np.zeros(motif_buffer_size, dtype=np.float64)
        )  # .pin_memory()
        self.out_distance = torch.from_numpy(
            np.zeros(motif_buffer_size, dtype=int)
        )  # .pin_memory()
        self.out_motifcounts = torch.from_numpy(
            np.ascontiguousarray(
                np.zeros((self.fragment_buffer_size, n_motifs), dtype=int)
            )
        )  # .pin_memory()

    def load(self, minibatch, **kwargs):
        super().load(minibatch, **kwargs)

        n_fragments = self.out_coordinates.shape[0]

        self.out_motifcounts.data.zero_()  # this is a big slow down (~20% of function cpu time) but unsure how to fix

        n_motifs = chromatinhd.loaders.extraction.motifs.extract_all(
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.motifscan_indptr,
            self.motifscan_indices,
            self.motifscan_data,
            *self.window,
            self.window_width,
            *self.cutwindow,
            self.out_fragment_indptr.numpy(),
            self.out_motif_ix.numpy(),
            self.out_score.numpy(),
            self.out_distance.numpy(),
            self.out_motifcounts.numpy()
        )
        self.out_fragment_indptr.resize_(n_fragments + 1)
        self.out_motif_ix.resize_(n_motifs)
        self.out_score.resize_(n_motifs)
        self.out_distance.resize_(n_motifs)
        self.out_motifcounts.resize_((n_fragments, self.out_motifcounts.shape[1]))

        return FullResult(
            motifcounts=self.out_motifcounts,
            local_cellxgene_ix=self.out_local_cellxgene_ix,
            n_fragments=n_fragments,
            **minibatch.items()
        )


@dataclasses.dataclass
class MotifcountsResult(Result):
    motifcounts: torch.Tensor


class Motifcounts(Fragments):
    def __init__(
        self, fragments, motifscan, cellxgene_batch_size, window, cutwindow, **kwargs
    ):
        super().__init__(fragments, cellxgene_batch_size, window, **kwargs)

        # store auxilliary information
        self.cutwindow = cutwindow
        self.cutwindow_width = cutwindow[1] - cutwindow[0]

        # create buffers for motifs
        self.n_motifs = motifscan.shape[1]
        self.n_features = self.n_motifs

        self.motifscan_indptr = motifscan.indptr.astype(np.int64)
        self.motifscan_indices = motifscan.indices.astype(np.int64)
        self.motifscan_data = motifscan.data.astype(np.float64)

    def load(self, minibatch, **kwargs):
        super().load(minibatch, **kwargs)

        n_fragments = self.out_coordinates.shape[0]

        out_motifcounts = np.zeros((n_fragments, self.n_features), dtype=np.int64)

        n_motifs = chromatinhd.loaders.extraction.motifs.extract_motifcounts(
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.motifscan_indptr,
            self.motifscan_indices,
            self.motifscan_data,
            *self.window,
            self.window_width,
            *self.cutwindow,
            out_motifcounts
        )
        out_motifcounts.resize((n_fragments, self.n_features))

        return MotifcountsResult(
            motifcounts=torch.from_numpy(out_motifcounts).to(torch.float),
            local_cellxgene_ix=self.out_local_cellxgene_ix,
            coordinates=self.out_coordinates,
            genemapping=self.out_genemapping,
            n_fragments=n_fragments,
            **minibatch.items()
        )


class MotifcountsSplit(Fragments):
    def __init__(
        self, fragments, motifscan, cellxgene_batch_size, window, cutwindow, **kwargs
    ):
        super().__init__(fragments, cellxgene_batch_size, window, **kwargs)

        # store auxilliary information
        self.cutwindow = cutwindow
        self.cutwindow_width = cutwindow[1] - cutwindow[0]

        # create buffers for motifs
        self.n_motifs = motifscan.shape[1]
        self.n_features = self.n_motifs * 2

        self.motifscan_indptr = motifscan.indptr
        self.motifscan_indices = motifscan.indices
        self.motifscan_data = motifscan.data

    def load(self, minibatch, **kwargs):
        super().load(minibatch, **kwargs)

        n_fragments = self.out_coordinates.shape[0]

        out_motifcounts = np.zeros((n_fragments, self.n_features), dtype=np.int64)

        n_motifs = chromatinhd.loaders.extraction.motifs.extract_motifcounts_split(
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.motifscan_indptr,
            self.motifscan_indices,
            self.motifscan_data,
            self.n_motifs,
            *self.window,
            self.window_width,
            *self.cutwindow,
            out_motifcounts
        )
        out_motifcounts.resize((n_fragments, self.n_features))

        return MotifcountsResult(
            motifcounts=torch.from_numpy(out_motifcounts).to(torch.float),
            local_cellxgene_ix=self.out_local_cellxgene_ix,
            coordinates=self.out_coordinates,
            genemapping=self.out_genemapping,
            n_fragments=n_fragments,
            **minibatch.items()
        )


class MotifcountsMultiple(Fragments):
    def __init__(
        self, fragments, motifscan, cellxgene_batch_size, window, cutwindows, **kwargs
    ):
        super().__init__(fragments, cellxgene_batch_size, window, **kwargs)

        # store auxilliary information
        self.cutwindows = cutwindows

        # create buffers for motifs
        self.n_motifs = motifscan.shape[1]
        self.n_features = self.n_motifs * (cutwindows.shape[0] - 1)

        self.motifscan_indptr = motifscan.indptr
        self.motifscan_indices = motifscan.indices
        self.motifscan_data = motifscan.data

    def load(self, minibatch, **kwargs):
        super().load(minibatch, **kwargs)

        n_fragments = self.out_coordinates.shape[0]

        out_motifcounts = np.zeros((n_fragments, self.n_features), dtype=np.int64)

        n_motifs = chromatinhd.loaders.extraction.motifs.extract_motifcounts_multiple(
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.motifscan_indptr,
            self.motifscan_indices,
            self.motifscan_data,
            self.n_motifs,
            *self.window,
            self.window_width,
            self.cutwindows,
            out_motifcounts
        )
        out_motifcounts.resize((n_fragments, self.n_features))

        return MotifcountsResult(
            motifcounts=torch.from_numpy(out_motifcounts).to(torch.float),
            local_cellxgene_ix=self.out_local_cellxgene_ix,
            coordinates=self.out_coordinates,
            genemapping=self.out_genemapping,
            n_fragments=n_fragments,
            **minibatch.items()
        )


class MotifcountsRelative(Fragments):
    def __init__(
        self,
        fragments,
        motifscan,
        cellxgene_batch_size,
        window,
        cutwindow,
        promoter_width,
        **kwargs
    ):
        super().__init__(fragments, cellxgene_batch_size, window, **kwargs)

        # store auxilliary information
        self.cutwindow = cutwindow
        self.cutwindow_width = cutwindow[1] - cutwindow[0]

        # create buffers for motifs
        self.n_motifs = motifscan.shape[1]
        self.n_features = self.n_motifs * 2

        self.promoter_width = promoter_width

        self.motifscan_indptr = motifscan.indptr.astype(np.int64)
        self.motifscan_indices = motifscan.indices.astype(np.int64)
        self.motifscan_data = motifscan.data.astype(np.float64)

    def load(self, minibatch, **kwargs):
        super().load(minibatch, **kwargs)

        n_fragments = self.out_coordinates.shape[0]

        out_motifcounts = np.zeros((n_fragments, self.n_features), dtype=np.int64)

        n_motifs = chromatinhd.loaders.extraction.motifs.extract_motifcounts_relative(
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.motifscan_indptr,
            self.motifscan_indices,
            self.motifscan_data,
            self.n_motifs,
            *self.window,
            self.window_width,
            *self.cutwindow,
            self.promoter_width,
            out_motifcounts
        )
        out_motifcounts.resize((n_fragments, self.n_features))

        return MotifcountsResult(
            motifcounts=torch.from_numpy(out_motifcounts).to(torch.float),
            local_cellxgene_ix=self.out_local_cellxgene_ix,
            coordinates=self.out_coordinates,
            genemapping=self.out_genemapping,
            n_fragments=n_fragments,
            **minibatch.items()
        )

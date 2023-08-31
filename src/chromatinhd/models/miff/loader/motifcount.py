import chromatinhd.data.fragments
import dataclasses
import numpy as np
import torch

from chromatinhd.loaders.fragments import Fragments
from chromatinhd.loaders.minibatches import Minibatch
from .motifs import Motifs
from typing import List


@dataclasses.dataclass
class Result:
    genecounts: torch.tensor = None
    global_binixs: torch.tensor = None
    binixs: torch.tensor = None
    bincounts: List[torch.tensor] = None

    def to(self, device):
        self.genecounts = self.genecounts.to(device)
        # if self.bincounts is not None:
        #     self.bincounts = [x.to(device) for x in self.bincounts]
        # self.binixs = self.binixs.to(device)
        # self.global_binixs = self.global_binixs.to(device)
        return self


# class BinnedMotifCounts:
#     """
#     Provides binned motif counts per fragment
#     """

#     def __init__(
#         self,
#         motifcounts,
#     ):
#         self.motifcounts = motifcounts
#         self.precomputed = [x.copy() for x in self.motifcounts.precomputed]
#         self.width = self.motifcounts.width

#     def load(self, minibatch, fragments):
#         # import time

#         genecounts = self.motifcounts.precomputed[0][minibatch.genes_oi, :].astype(np.float32)

#         # create bin counts
#         # bincounts = []
#         binixs = []
#         global_binixs = []
#         coordinates = fragments.coordinates[:, 0].numpy() - fragments.window[0]

#         for binset_ix, (
#             fragment_binsize,
#             parent_fragment_binsize,
#             parent_fragment_width,
#         ) in enumerate(
#             zip(
#                 self.motifcounts.fragment_binsizes,
#                 [self.width, *self.motifcounts.fragment_binsizes[:-1]],
#                 [1, *self.motifcounts.fragment_widths[:-1]],
#             )
#         ):
#             global_binix = (
#                 fragments.genemapping.cpu().numpy() * parent_fragment_width + coordinates // parent_fragment_binsize
#             )
#             # get fragment bin ixs
#             global_binixs.append(global_binix)
#             binixs.append((coordinates % parent_fragment_binsize) // fragment_binsize)

#             # bincounts.append(self.precomputed[binset_ix][global_binix])

#         binixs = np.array(binixs, dtype=int).T
#         global_binixs = np.array(global_binixs, dtype=int).T

#         return Result(
#             # bincounts=[torch.from_numpy(x) for x in bincounts],
#             binixs=torch.from_numpy(binixs),
#             global_binixs=torch.from_numpy(global_binixs),
#             genecounts=torch.from_numpy(genecounts),
#         )


class BinnedMotifCounts:
    """
    Provides binned motif counts per fragment
    """

    def __init__(
        self,
        motifcounts,
    ):
        self.motifcounts = motifcounts
        self.precomputed = [x.copy() for x in self.motifcounts.precomputed]
        self.width = self.motifcounts.width

    def load(self, minibatch, fragments):
        genecounts = self.motifcounts.precomputed[minibatch.genes_oi, :].astype(np.float32)

        global_binixs = (fragments.coordinates[:, 0].numpy() - fragments.window[0]) // self.motifcounts.binsize

        return Result(
            genecounts=torch.from_numpy(genecounts),
            global_binixs=torch.from_numpy(global_binixs),
        )


@dataclasses.dataclass
class MotifCountsFragmentsResult:
    fragments: Fragments
    motifcounts: Result
    minibatch: Minibatch

    def to(self, device):
        self.fragments.to(device)
        self.motifcounts.to(device)
        self.minibatch.to(device)
        return self


class MotifCountsFragments:
    """
    Provides both motifcounts and fragments data for a minibatch.
    """

    def __init__(
        self,
        motifcounts,
        fragments: chromatinhd.data.fragments.Fragments,
        cellxregion_batch_size: int,
    ):
        # ensure that the order of motifs and fragment.obs is the same
        self.fragments = Fragments(fragments, cellxregion_batch_size=cellxregion_batch_size, fully_contained=True)
        self.motifcounts = BinnedMotifCounts(motifcounts)

    def load(self, minibatch):
        fragments = self.fragments.load(minibatch)
        return MotifCountsFragmentsResult(
            fragments=fragments,
            motifcounts=self.motifcounts.load(minibatch, fragments),
            minibatch=minibatch,
        )

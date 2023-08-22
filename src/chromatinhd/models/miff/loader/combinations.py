import chromatinhd.data.fragments
import dataclasses
import numpy as np
import torch

from chromatinhd.models.pred.loader import Fragments as FragmentsLoader
from .minibatches import Minibatch
from .binnedmotifcounts import BinnedMotifCounts as BinnedMotifCountsLoader
from .clustering import Clustering as ClusteringLoader
from typing import List


@dataclasses.dataclass
class MotifCountsFragmentsClusteringResult:
    fragments: any
    motifcounts: any
    clustering: any
    minibatch: any

    def to(self, device):
        self.fragments.to(device)
        self.motifcounts.to(device)
        self.clustering.to(device)
        self.minibatch.to(device)
        return self


class MotifCountsFragmentsClustering:
    """
    Provides both motifcounts and fragments data for a minibatch.
    """

    def __init__(
        self,
        motifcounts,
        fragments: chromatinhd.data.fragments.Fragments,
        clustering: chromatinhd.data.clustering.Clustering,
        cellxregion_batch_size: int,
    ):
        # ensure that the order of motifs and fragment.obs is the same
        self.fragments = FragmentsLoader(fragments, cellxregion_batch_size=cellxregion_batch_size, fully_contained=True)
        self.motifcounts = BinnedMotifCountsLoader(motifcounts)
        self.clustering = ClusteringLoader(clustering)

    def load(self, minibatch):
        fragments = self.fragments.load(minibatch)
        return MotifCountsFragmentsClusteringResult(
            fragments=fragments,
            motifcounts=self.motifcounts.load(minibatch, fragments),
            clustering=self.clustering.load(minibatch),
            minibatch=minibatch,
        )

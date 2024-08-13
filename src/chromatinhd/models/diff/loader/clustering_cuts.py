import chromatinhd.data.fragments
import dataclasses

from chromatinhd.loaders.clustering import Clustering
from chromatinhd.loaders.minibatches import Minibatch
from chromatinhd.loaders.fragments import Cuts


@dataclasses.dataclass
class Result:
    cuts: Cuts
    clustering: Clustering
    minibatch: Minibatch

    def to(self, device):
        self.cuts.to(device)
        self.clustering.to(device)
        self.minibatch.to(device)
        return self


class ClusteringCuts:
    """
    Provides both clustering and cuts data for a minibatch.
    """

    def __init__(
        self,
        clustering: chromatinhd.data.clustering.Clustering,
        fragments: chromatinhd.data.fragments.Fragments,
        cellxregion_batch_size: int,
    ):
        # ensure that the order of clustering and fragment.obs is the same
        if not all(clustering.labels.index == fragments.obs.index):
            raise ValueError("Clustering and fragments should have the same obs index. ")
        self.clustering = Clustering(clustering)
        self.cuts = Cuts(fragments, cellxregion_batch_size=cellxregion_batch_size)

    def load(self, minibatch):
        return Result(
            cuts=self.cuts.load(minibatch),
            clustering=self.clustering.load(minibatch),
            minibatch=minibatch,
        )

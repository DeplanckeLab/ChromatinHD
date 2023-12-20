import chromatinhd.data.clustering
import dataclasses

from .fragments import Cuts, CutsRegional, CutsResult
from chromatinhd.loaders.minibatches import Minibatch
from .clustering import Clustering


@dataclasses.dataclass
class Result:
    clustering: Clustering
    cuts: CutsResult
    minibatch: Minibatch

    def to(self, device):
        self.clustering.to(device)
        self.cuts.to(device)
        self.minibatch.to(device)
        return self


class ClusteringCuts:
    def __init__(
        self,
        fragments: chromatinhd.data.fragments.Fragments,
        clustering: chromatinhd.data.clustering.Clustering,
        cellxregion_batch_size: int,
        layer: str = None,
        region_oi=None,
    ):
        # ensure that clustering and fragments have the same obs
        # if not all(clustering.obs.index == fragments.obs.index):
        #     raise ValueError("Clustering and fragments should have the same obs index. ")

        if region_oi is None:
            self.cuts = Cuts(fragments, cellxregion_batch_size=cellxregion_batch_size)
            self.clustering = Clustering(clustering)
        else:
            self.cuts = CutsRegional(fragments, cellxregion_batch_size=cellxregion_batch_size, region_oi=region_oi)
            self.clustering = Clustering(clustering)

    def load(self, minibatch):
        return Result(
            clustering=self.clustering.load(minibatch),
            cuts=self.cuts.load(minibatch),
            minibatch=minibatch,
        )

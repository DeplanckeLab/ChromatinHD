import chromatinhd.data.fragments
import dataclasses

from chromatinhd.loaders.fragments import Fragments
from chromatinhd.loaders.clustering import Clustering
from chromatinhd.loaders.minibatches import Minibatch


@dataclasses.dataclass
class Result:
    fragments: Fragments
    clustering: Clustering
    minibatch: Minibatch

    def to(self, device):
        self.fragments.to(device)
        self.clustering.to(device)
        self.minibatch.to(device)
        return self


class ClusteringFragments:
    """
    Provides both clustering and fragments data for a minibatch.
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
        self.fragments = Fragments(fragments, cellxregion_batch_size=cellxregion_batch_size)

    def load(self, minibatch):
        return Result(
            fragments=self.fragments.load(minibatch),
            clustering=self.clustering.load(minibatch),
            minibatch=minibatch,
        )

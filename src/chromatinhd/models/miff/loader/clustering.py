import chromatinhd.data.clustering
import dataclasses
import torch


@dataclasses.dataclass
class Result:
    labels: torch.Tensor

    def to(self, device):
        self.labels = self.labels.to(device)
        return self


class Clustering:
    """
    Provides clustering data for a minibatch.
    """

    def __init__(
        self,
        clustering: chromatinhd.data.clustering.Clustering,
    ):
        assert (clustering.labels.cat.categories == clustering.cluster_info.index).all(), (
            clustering.labels.cat.categories,
            clustering.cluster_info.index,
        )
        self.labels = torch.from_numpy(clustering.labels.cat.codes.values.copy()).to(torch.int64)

    def load(self, minibatch):
        return Result(labels=self.labels[minibatch.cells_oi])

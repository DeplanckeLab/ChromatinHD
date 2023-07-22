import chromatinhd.data.clustering
import dataclasses
import torch


@dataclasses.dataclass
class Result:
    onehot: torch.Tensor

    def to(self, device):
        self.onehot = self.onehot.to(device)
        return self


class Clustering:
    """
    Provides clustering data for a minibatch.
    """

    def __init__(
        self,
        clustering: chromatinhd.data.clustering.Clustering,
    ):
        assert (
            clustering.labels.cat.categories == clustering.cluster_info.index
        ).all(), (
            clustering.labels.cat.categories,
            clustering.cluster_info.index,
        )
        self.onehot = torch.nn.functional.one_hot(
            torch.from_numpy(clustering.labels.cat.codes.values.copy()).to(torch.int64),
            clustering.n_clusters,
        )

    def load(self, minibatch):
        onehot = self.onehot[minibatch.cells_oi, :]
        return Result(onehot=onehot)

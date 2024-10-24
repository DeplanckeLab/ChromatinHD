import chromatinhd.data.clustering
import dataclasses
import torch


@dataclasses.dataclass
class Result:
    # onehot: torch.Tensor
    indices: torch.Tensor

    def to(self, device):
        self.indices = self.indices.to(device)
        # self.onehot = self.onehot.to(device)
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
        self.onehot = torch.nn.functional.one_hot(
            torch.from_numpy(clustering.labels.cat.codes.values.copy()).to(torch.int64),
            clustering.n_clusters,
        ).to(torch.float)

    def load(self, minibatch):
        # onehot = self.onehot[minibatch.cells_oi, :]
        indices = torch.argmax(self.onehot[minibatch.cells_oi, :], dim=1)
        return Result(indices=indices)

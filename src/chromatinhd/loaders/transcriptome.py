import chromatinhd.data.transcriptome
import dataclasses
import torch
import chromatinhd.sparse
from chromatinhd.flow.tensorstore import TensorstoreInstance


@dataclasses.dataclass
class Result:
    value: torch.Tensor

    def to(self, device):
        self.value = self.value.to(device)
        return self


class Transcriptome:
    def __init__(
        self,
        transcriptome: chromatinhd.data.transcriptome.Transcriptome,
        layer: str = None,
    ):
        if layer is None:
            layer = list(transcriptome.layers.keys())[0]

        X = transcriptome.layers[layer]
        if chromatinhd.sparse.is_sparse(X):
            self.X = X.dense()
        elif torch.is_tensor(X):
            self.X = X
        elif isinstance(X, TensorstoreInstance):
            # self.X = X
            self.X = X.oindex  # open a tensorstore reader with orthogonal indexing
        else:
            self.X = torch.from_numpy(X)

    def load(self, minibatch):
        X = torch.from_numpy(self.X[minibatch.cells_oi, minibatch.genes_oi])
        if X.ndim == 1:
            X = X.unsqueeze(1)
        return Result(value=X)

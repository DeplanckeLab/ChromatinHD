import chromatinhd.data.transcriptome
import dataclasses
import torch
import chromatinhd.sparse


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
            X = transcriptome.X
        else:
            X = transcriptome.layers[layer]
        if chromatinhd.sparse.is_sparse(X):
            self.X = X.dense()
        elif torch.is_tensor(X):
            self.X = X
        else:
            self.X = torch.from_numpy(X)

    def load(self, minibatch):
        X = self.X[minibatch.cells_oi, :][:, minibatch.genes_oi]
        return Result(value=X)

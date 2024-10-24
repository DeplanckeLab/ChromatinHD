import chromatinhd.data.transcriptome
import dataclasses
import torch
import chromatinhd.sparse
from chromatinhd.flow.tensorstore import TensorstoreInstance
import numpy as np


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
            self.X = X.numpy()
        elif isinstance(X, TensorstoreInstance):
            # self.X = X
            self.X = X.oindex  # open a tensorstore reader with orthogonal indexing
        else:
            self.X = X

    def load(self, minibatch):
        X = torch.from_numpy(self.X[minibatch.cells_oi, minibatch.genes_oi].astype(np.float32))
        if X.ndim == 1:
            X = X.unsqueeze(1)
        return Result(value=X)


class TranscriptomeGene:
    def __init__(
        self,
        transcriptome: chromatinhd.data.transcriptome.Transcriptome,
        gene_oi,
        layer: str = None,
    ):
        if layer is None:
            layer = list(transcriptome.layers.keys())[0]

        gene_ix = transcriptome.var.index.get_loc(gene_oi)

        X = transcriptome.layers[layer]
        if chromatinhd.sparse.is_sparse(X):
            self.X = X[:, gene_ix].dense()[:, 0]
        elif torch.is_tensor(X):
            self.X = X[:, gene_ix].numpy()
        elif isinstance(X, TensorstoreInstance):
            # self.X = X
            self.X = X.oindex[:, gene_ix]  # open a tensorstore reader with orthogonal indexing
        else:
            self.X = X[:, gene_ix]

    def load(self, minibatch):
        X = torch.from_numpy(self.X[minibatch.cells_oi].astype(np.float32))
        if X.ndim == 1:
            X = X.unsqueeze(1)
        return Result(value=X)

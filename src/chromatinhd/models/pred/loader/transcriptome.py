import chromatinhd.data.transcriptome
import dataclasses
import torch


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
        layer: str = "X",
    ):
        self.X = torch.from_numpy(transcriptome.layers[layer])

    def load(self, minibatch):
        X = self.X[minibatch.cells_oi, :][:, minibatch.genes_oi]
        return Result(value=X)

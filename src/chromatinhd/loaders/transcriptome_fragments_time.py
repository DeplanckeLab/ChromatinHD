import chromatinhd.data.transcriptome
import chromatinhd.data.gradient
import dataclasses
import numpy as np
import torch

from .fragments import Fragments
from chromatinhd.loaders.minibatches import Minibatch
from .transcriptome import Result as TranscriptomeResult


@dataclasses.dataclass
class Result:
    transcriptome: TranscriptomeResult
    fragments: Fragments
    minibatch: Minibatch

    def to(self, device):
        self.transcriptome.to(device)
        self.fragments.to(device)
        self.minibatch.to(device)
        return self


class TranscriptomeTime:
    def __init__(
        self,
        transcriptome: chromatinhd.data.transcriptome.Transcriptome,
        gradient: chromatinhd.data.gradient.Gradient,
        layer: str = None,
        delta_time=0.25,
        n_bins=20,
        delta_expression=False,
    ):
        bins = self.bins = np.array([0] + list(np.linspace(0.1, 0.9, n_bins - 1)) + [1])

        x = self.x = gradient.values[:, 0]
        if layer is None:
            layer = list(transcriptome.layers.keys())[0]
        y = transcriptome.layers[layer][:]

        x_binned = self.x_binned = np.clip(np.searchsorted(bins, x) - 1, 0, bins.size - 2)
        x_onehot = np.zeros((x_binned.size, x_binned.max() + 1))
        x_onehot[np.arange(x_binned.size), x_binned] = 1
        y_binned = (x_onehot.T @ y) / x_onehot.sum(axis=0)[:, None]
        self.y_binned = (y_binned - y_binned.min(0)) / (y_binned.max(0) - y_binned.min(0))

        self.delta_time = delta_time
        self.delta_expression = delta_expression

    def load(self, minibatch):
        x = self.x[minibatch.cells_oi]
        x_desired = x + self.delta_time
        x_desired_bin = np.clip(np.searchsorted(self.bins, x_desired) - 1, 0, self.bins.size - 2)
        if self.delta_expression:
            x_desired_bin2 = np.clip(np.searchsorted(self.bins, x_desired + self.delta_time) - 1, 0, self.bins.size - 2)
            y_desired = (
                self.y_binned[x_desired_bin2, :][:, minibatch.genes_oi]
                - self.y_binned[x_desired_bin, :][:, minibatch.genes_oi]
            )
        else:
            y_desired = self.y_binned[x_desired_bin, :][:, minibatch.genes_oi]

        return TranscriptomeResult(value=torch.from_numpy(y_desired))


class TranscriptomeFragmentsTime:
    def __init__(
        self,
        fragments: chromatinhd.data.fragments.Fragments,
        transcriptome: chromatinhd.data.transcriptome.Transcriptome,
        gradient: chromatinhd.data.gradient.Gradient,
        cellxregion_batch_size: int,
        layer: str = None,
        delta_time=0.25,
        n_bins=20,
        delta_expression=False,
    ):
        # ensure that transcriptome and fragments have the same var
        if not all(transcriptome.var.index == fragments.var.index):
            raise ValueError("Transcriptome and fragments should have the same var index.")

        self.fragments = Fragments(fragments, cellxregion_batch_size=cellxregion_batch_size)
        self.transcriptome = TranscriptomeTime(
            transcriptome,
            gradient,
            layer=layer,
            delta_expression=delta_expression,
            delta_time=delta_time,
            n_bins=n_bins,
        )

    def load(self, minibatch):
        return Result(
            transcriptome=self.transcriptome.load(minibatch),
            fragments=self.fragments.load(minibatch),
            minibatch=minibatch,
        )

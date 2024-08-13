import chromatinhd.data.fragments
import dataclasses
import numpy as np

from chromatinhd.loaders.fragments import Fragments
from chromatinhd.loaders.minibatches import Minibatch
from .motifs import Motifs
from typing import List


@dataclasses.dataclass
class Result:
    fragments: Fragments
    motifs: Motifs
    minibatch: Minibatch

    def to(self, device):
        self.fragments.to(device)
        self.motifs.to(device)
        self.minibatch.to(device)
        return self


class MotifsFragments:
    """
    Provides both motifs and fragments data for a minibatch.
    """

    def __init__(
        self,
        motifscan: chromatinhd.data.motifscan.Motifscan,
        fragments: chromatinhd.data.fragments.Fragments,
        cellxregion_batch_size: int,
    ):
        self.motifs = Motifs(motifscan)
        self.fragments = Fragments(fragments, cellxregion_batch_size=cellxregion_batch_size)

    def load(self, minibatch):
        return Result(
            fragments=self.fragments.load(minibatch),
            motifs=self.motifs.load(minibatch),
            minibatch=minibatch,
        )

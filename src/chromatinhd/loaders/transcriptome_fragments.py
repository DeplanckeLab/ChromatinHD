import chromatinhd.data.transcriptome
import dataclasses

from .fragments import Fragments
from chromatinhd.loaders.minibatches import Minibatch
from .transcriptome import Transcriptome


@dataclasses.dataclass
class Result:
    transcriptome: Transcriptome
    fragments: Fragments
    minibatch: Minibatch

    def to(self, device):
        self.transcriptome.to(device)
        self.fragments.to(device)
        self.minibatch.to(device)
        return self


class TranscriptomeFragments:
    def __init__(
        self,
        fragments: chromatinhd.data.fragments.Fragments,
        transcriptome: chromatinhd.data.transcriptome.Transcriptome,
        cellxregion_batch_size: int,
        layer: str = None,
    ):
        # ensure that transcriptome and fragments have the same var
        if not all(transcriptome.var.index == fragments.var.index):
            raise ValueError("Transcriptome and fragments should have the same var index. ")

        self.fragments = Fragments(fragments, cellxregion_batch_size=cellxregion_batch_size)
        self.transcriptome = Transcriptome(transcriptome, layer=layer)

    def load(self, minibatch):
        return Result(
            transcriptome=self.transcriptome.load(minibatch),
            fragments=self.fragments.load(minibatch),
            minibatch=minibatch,
        )

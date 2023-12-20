import chromatinhd.data.transcriptome
import dataclasses

from .fragments2 import Fragments
from chromatinhd.loaders.minibatches import Minibatch
from .transcriptome import Transcriptome, TranscriptomeGene


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
        regionxcell_batch_size: int,
        layer: str = None,
        region_oi=None,
    ):
        # ensure that transcriptome and fragments have the same var
        if not all(transcriptome.var.index == fragments.var.index):
            raise ValueError("Transcriptome and fragments should have the same var index. ")

        self.fragments = Fragments(fragments, regionxcell_batch_size=regionxcell_batch_size)
        self.transcriptome = Transcriptome(transcriptome, layer=layer)

    def load(self, minibatch):
        return Result(
            transcriptome=self.transcriptome.load(minibatch),
            fragments=self.fragments.load(minibatch),
            minibatch=minibatch,
        )

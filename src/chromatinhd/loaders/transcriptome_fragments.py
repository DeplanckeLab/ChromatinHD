import chromatinhd.data.transcriptome
import dataclasses

from .fragments import Fragments, FragmentsRegional
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
        cellxregion_batch_size: int,
        layer: str = None,
        region_oi=None,
    ):
        # ensure that transcriptome and fragments have the same var
        if not all(transcriptome.var.index == fragments.var.index):
            raise ValueError("Transcriptome and fragments should have the same var index. ")

        if region_oi is None:
            self.fragments = Fragments(
                fragments,
                cellxregion_batch_size=cellxregion_batch_size,
                provide_multiplets=False,
                provide_libsize=True,
            )
            self.transcriptome = Transcriptome(transcriptome, layer=layer)
        else:
            self.fragments = FragmentsRegional(
                fragments,
                cellxregion_batch_size=cellxregion_batch_size,
                region_oi=region_oi,
                provide_libsize=True,
            )
            self.transcriptome = TranscriptomeGene(transcriptome, gene_oi=region_oi, layer=layer)

    def load(self, minibatch):
        return Result(
            transcriptome=self.transcriptome.load(minibatch),
            fragments=self.fragments.load(minibatch),
            minibatch=minibatch,
        )

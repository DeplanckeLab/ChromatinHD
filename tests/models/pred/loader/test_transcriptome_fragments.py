import numpy as np
import chromatinhd as chd


class TestTranscriptomeFragments:
    def test_example(self, example_fragments, example_transcriptome):
        loader = chd.models.pred.loader.TranscriptomeFragments(
            fragments=example_fragments,
            transcriptome=example_transcriptome,
            cellxgene_batch_size=10000,
        )

        minibatch = chd.models.pred.loader.Minibatch(
            cells_oi=np.arange(20), genes_oi=np.arange(5), phase="train"
        )
        result = loader.load(minibatch)

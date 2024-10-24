import numpy as np
import chromatinhd as chd


class TestTranscriptomeFragments:
    def test_example(self, example_fragments, example_transcriptome):
        loader = chd.loaders.TranscriptomeFragments(
            fragments=example_fragments,
            transcriptome=example_transcriptome,
            cellxregion_batch_size=10000,
        )

        minibatch = chd.loaders.minibatches.Minibatch(cells_oi=np.arange(20), regions_oi=np.arange(5), phase="train")
        loader.load(minibatch)

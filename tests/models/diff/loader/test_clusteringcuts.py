import numpy as np
import chromatinhd as chd


class TestClusteringCuts:
    def test_example(self, example_fragments, example_clustering):
        loader = chd.models.diff.loader.ClusteringCuts(
            fragments=example_fragments,
            clustering=example_clustering,
            cellxgene_batch_size=10000,
        )

        minibatch = chd.models.diff.loader.Minibatch(
            cells_oi=np.arange(20), genes_oi=np.arange(5), phase="train"
        )
        result = loader.load(minibatch)

import numpy as np


class ClusteringCuts:
    def test_example(self):
        loader = chd.models.diff.loader.ClusteringCuts(
            fragments=fragments,
            clustering=clustering,
            cellxgene_batch_size=10000,
        )

        minibatch = chd.models.diff.loader.Minibatch(
            cells_oi=np.arange(20), genes_oi=np.arange(5), phase="train"
        )
        result = loader.load(minibatch)

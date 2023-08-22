import numpy as np
import chromatinhd as chd


class TestFragments:
    def test_example(self, example_fragments):
        loader = chd.loaders.fragments.Fragments(
            fragments=example_fragments,
            cellxregion_batch_size=10000,
        )

        minibatch = chd.loaders.minibatches.Minibatch(cells_oi=np.arange(20), regions_oi=np.arange(5), phase="train")
        data = loader.load(minibatch)

        assert data.coordinates.shape[0] == data.n_fragments
        assert data.coordinates.shape[0] == data.local_cellxregion_ix.shape[0]


class TestCuts:
    def test_example(self, example_fragments):
        loader = chd.loaders.fragments.Cuts(
            fragments=example_fragments,
            cellxregion_batch_size=10000,
        )

        minibatch = chd.loaders.minibatches.Minibatch(cells_oi=np.arange(20), regions_oi=np.arange(5), phase="train")
        data = loader.load(minibatch)

        assert data.coordinates.shape[0] == data.n_cuts
        assert data.coordinates.shape[0] == data.local_cellxregion_ix.shape[0]

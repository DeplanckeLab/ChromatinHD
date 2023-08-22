import chromatinhd.loaders.fragments_helpers
import numpy as np


def test_multiple_arange():
    a = np.array([10, 50, 60], dtype=np.int64)
    b = np.array([20, 52, 62], dtype=np.int64)
    ix = np.zeros(100, dtype=np.int64)
    local_cellxregion_ix = np.zeros(100, dtype=np.int64)

    n_fragments = chromatinhd.loaders.fragments_helpers.multiple_arange(a, b, ix, local_cellxregion_ix)
    ix.resize(n_fragments, refcheck=False)
    local_cellxregion_ix.resize(n_fragments, refcheck=False)

    assert ix.shape == (14,)
    assert local_cellxregion_ix.shape == (14,)
    assert local_cellxregion_ix.max() < 3
    assert np.all(ix == np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 50, 51, 60, 61], dtype=np.int64))
    assert np.all(local_cellxregion_ix == np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2], dtype=np.int64))

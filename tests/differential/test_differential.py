
from peakfreeatac.differential import DifferentialSlices
import numpy as np

class TestDifferentialSlices:
    def test_from_positions(self):
        positions = np.array([0, 1, 2, 3, 100, 101, 105])
        gene_ixs = np.array([1, 1, 3, 3, 1, 1, 2])
        cluster_ixs = np.array([1, 1, 1, 1, 2, 2, 0])
        result = DifferentialSlices.from_positions(positions, gene_ixs, cluster_ixs, np.array([0, 200]), n_genes = 4, n_clusters = 3)

        assert np.array_equal(result.positions, np.array([[0, 2], [2, 4], [100, 102], [105, 106]])), result.positions
        assert np.array_equal(result.gene_ixs, np.array([1, 3, 1, 2]))
        assert np.array_equal(result.cluster_ixs, np.array([1, 1, 2, 0]))
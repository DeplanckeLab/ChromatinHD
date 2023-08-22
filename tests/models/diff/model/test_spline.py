import chromatinhd.models.diff.model.spline
import torch
import numpy as np


class TestDifferentialQuadraticSplineStack:
    def test_basic(self):
        transform = chromatinhd.models.diff.model.spline.DifferentialQuadraticSplineStack(nbins=(128,), n_regions=1)

        x = torch.linspace(0, 1, 100)
        genes_oi = torch.tensor([0])
        local_gene_ix = torch.zeros(len(x), dtype=torch.int)

        delta = torch.zeros((len(x), np.sum(transform.split_deltas)))
        delta[:, :30] = 0
        ouput, logabsdet = transform.transform_forward(x, genes_oi, local_gene_ix, delta)

        assert np.isclose(np.trapz(torch.exp(logabsdet).detach().numpy(), x), 1, atol=5e-2)

        delta[:, :30] = 1
        ouput, logabsdet = transform.transform_forward(x, genes_oi, local_gene_ix, delta)

        assert np.isclose(np.trapz(torch.exp(logabsdet).detach().numpy(), x), 1, atol=5e-2)

        delta[:, :] = torch.from_numpy(np.random.normal(size=(1, delta.shape[1])))
        ouput, logabsdet = transform.transform_forward(x, genes_oi, local_gene_ix, delta)

        assert np.isclose(np.trapz(torch.exp(logabsdet).detach().numpy(), x), 1, atol=5e-2)

import torch
import numpy as np

def normalize(X, target_sum = 1e6):
    X = np.array(X.todense())
    X = X / (X.sum(1, keepdims = True) + 1e-3) * target_sum
    X = np.log1p(X)
    return X

import sklearn.decomposition
class Model(torch.nn.Module):
    def __init__(
        self,
        n_components = 50
    ):
        super().__init__()

        self.pca_model = sklearn.decomposition.PCA(n_components)

    def fit(self, data):
        self.pca_model.fit(normalize(data.counts))

    def forward(self, data):
        return self.pca_model.transform(normalize(data.counts))

    def evaluate_latent(self, data):
        return self.pca_model.transform(normalize(data.counts))

    @property
    def n_latent_dimensions(self):
        return self.pca_model.n_components

    # def evaluate_pseudo(self, coordinates, latent, cells_oi, n, gene_oi = None):
    #     device = coordinates.device
    #     if gene_oi is None:
    #         gene_oi = 0
    #     genes_oi = torch.tensor([gene_oi], device = coordinates.device, dtype = torch.long)
    #     cut_local_gene_ix = torch.zeros_like(coordinates).to(torch.long)
    #     n_cells = len(cells_oi)
    #     cut_local_cellxgene_ix = torch.arange(n_cells, device = coordinates.device).tile(n)

    #     data = FullDict(
    #         cut_local_gene_ix = cut_local_gene_ix.to(device),
    #         cut_local_cellxgene_ix = cut_local_cellxgene_ix.to(device),
    #         cut_coordinates = coordinates.to(device),
    #         n_cells = n_cells,
    #         n_genes = 1,
    #         genes_oi_torch = genes_oi.to(device),
    #         cells_oi_torch = cells_oi.to(device),
    #         latent = latent.to(device)
    #     )
    #     return self.forward_likelihood_mixture(data)

class FullDict(dict):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
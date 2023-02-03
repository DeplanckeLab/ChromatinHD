import torch
import numpy as np
import math
import tqdm.auto as tqdm
import torch_scatter

from chromatinhd.embedding import EmbeddingTensor
from .spline import DifferentialQuadraticSplineStack


class Decoder(torch.nn.Module):
    def __init__(
        self, n_latent, n_genes, n_output_components, n_layers=1, n_hidden_dimensions=32
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                torch.nn.Linear(
                    n_hidden_dimensions if i > 0 else n_latent, n_hidden_dimensions
                )
            )
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(n_hidden_dimensions))
            layers.append(torch.nn.Dropout(0.2))
        self.nn = torch.nn.Sequential(*layers)

        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_output_components = n_output_components

        self.logit_weight = EmbeddingTensor(
            n_genes,
            (n_hidden_dimensions if n_layers > 0 else n_latent, n_output_components),
            sparse=True,
        )
        stdv = 1.0 / math.sqrt(self.logit_weight.weight.size(1))
        self.logit_weight.weight.data.uniform_(-stdv, stdv)

        self.rho_weight = EmbeddingTensor(
            n_genes, (n_hidden_dimensions if n_layers > 0 else n_latent,), sparse=True
        )
        stdv = 1.0 / math.sqrt(self.rho_weight.weight.size(1)) / 100
        self.rho_weight.weight.data.uniform_(-stdv, stdv)

    def forward(self, latent, genes_oi):
        logit_weight = self.logit_weight(genes_oi)
        rho_weight = self.rho_weight(genes_oi)
        nn_output = self.nn(latent)

        # nn_output is broadcasted across genes and across components
        logit = torch.matmul(nn_output.unsqueeze(1).unsqueeze(2), logit_weight).squeeze(
            -2
        )

        rho = torch.matmul(nn_output, rho_weight.T).squeeze(-1)
        return logit, rho

    def parameters_dense(self):
        return [*self.nn.parameters()]

    def parameters_sparse(self):
        return [self.logit_weight.weight, self.rho_weight.weight]


class BaselineDecoder(torch.nn.Module):
    def __init__(
        self, n_latent, n_genes, n_output_components, n_layers=1, n_hidden_dimensions=32
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                torch.nn.Linear(
                    n_hidden_dimensions if i > 0 else n_latent, n_hidden_dimensions
                )
            )
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(n_hidden_dimensions))
            layers.append(torch.nn.Dropout(0.2))
        self.nn = torch.nn.Sequential(*layers)

        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_output_components = n_output_components

        self.rho_weight = EmbeddingTensor(
            n_genes, (n_hidden_dimensions if n_layers > 0 else n_latent,), sparse=True
        )
        stdv = 1.0 / math.sqrt(self.rho_weight.weight.size(1)) / 100
        self.rho_weight.weight.data.uniform_(-stdv, stdv)

    def forward(self, latent, genes_oi):
        rho_weight = self.rho_weight(genes_oi)
        nn_output = self.nn(latent)

        # nn_output is broadcasted across genes and across components
        logit = torch.zeros(
            (latent.shape[0], len(genes_oi), self.n_output_components),
            device=latent.device,
        )

        rho = torch.matmul(nn_output, rho_weight.T).squeeze(-1)
        return logit, rho

    def parameters_dense(self):
        return [*self.nn.parameters()]

    def parameters_sparse(self):
        return [self.rho_weight.weight]


class Decoding(torch.nn.Module):
    def __init__(
        self,
        fragments,
        cell_latent_space,
        nbins=(128,),
        decoder_n_layers=0,
        baseline=False,
    ):
        super().__init__()

        n_latent_dimensions = cell_latent_space.shape[1]

        x = fragments.coordinates.flatten()
        x = (x - fragments.window[0]) / (fragments.window[1] - fragments.window[0])
        keep_cuts = (x >= 0) & (x <= 1)
        x = x[keep_cuts]
        local_gene_ix = fragments.genemapping.expand(2, -1).T.flatten()[keep_cuts]

        from .spline import DifferentialQuadraticSplineStack, TransformedDistribution

        transform = DifferentialQuadraticSplineStack(
            x.cpu(),
            nbins=nbins,
            local_gene_ix=local_gene_ix.cpu(),
            n_genes=fragments.n_genes,
        )
        self.mixture = TransformedDistribution(transform)
        n_delta_mixture_components = sum(transform.split_deltas)

        if not baseline:
            self.decoder = Decoder(
                n_latent_dimensions,
                fragments.n_genes,
                n_delta_mixture_components,
                n_layers=decoder_n_layers,
            )
        else:
            self.decoder = BaselineDecoder(
                n_latent_dimensions,
                fragments.n_genes,
                n_delta_mixture_components,
                n_layers=decoder_n_layers,
            )

        libsize = torch.bincount(fragments.mapping[:, 0], minlength=fragments.n_cells)
        rho_bias = (
            torch.bincount(fragments.mapping[:, 1], minlength=fragments.n_genes)
            / fragments.n_cells
            / libsize.to(torch.float).mean()
        )

        self.register_buffer("libsize", libsize)
        self.register_buffer("rho_bias", rho_bias)

        self.register_buffer("cell_latent_space", cell_latent_space)

        self.track = {}

    def forward_(
        self,
        local_cellxgene_ix,
        cut_coordinates,
        latent,
        genes_oi,
        cells_oi,
        cut_local_cellxgene_ix,
        cut_local_gene_ix,
        n_cells,
        n_genes,
    ):
        # decode
        mixture_delta, rho_delta = self.decoder(latent, genes_oi)

        # fragment counts
        mixture_delta_cellxgene = mixture_delta.view(
            np.prod(mixture_delta.shape[:2]), mixture_delta.shape[-1]
        )
        mixture_delta = mixture_delta_cellxgene[cut_local_cellxgene_ix]

        likelihood_mixture = self.track["likelihood_mixture"] = self.mixture.log_prob(
            cut_coordinates, genes_oi, cut_local_gene_ix, mixture_delta
        )

        # expression
        fragmentexpression = (
            self.rho_bias[genes_oi] * torch.exp(rho_delta)
        ) * self.libsize[cells_oi].unsqueeze(1)
        fragmentcounts_p = torch.distributions.Poisson(fragmentexpression)
        fragmentcounts = torch.reshape(
            torch.bincount(local_cellxgene_ix, minlength=n_cells * n_genes),
            (n_cells, n_genes),
        )
        likelihood_fragmentcounts = self.track[
            "likelihood_fragmentcounts"
        ] = fragmentcounts_p.log_prob(fragmentcounts)

        # likelihood
        likelihood = likelihood_mixture.sum() + likelihood_fragmentcounts.sum()

        # ELBO
        elbo = -likelihood

        return elbo

    def forward(self, data):
        if not hasattr(data, "latent"):
            data.latent = self.cell_latent_space[data.cells_oi]
        return self.forward_(
            cut_coordinates=data.cut_coordinates,
            latent=data.latent,
            cells_oi=data.cells_oi_torch,
            genes_oi=data.genes_oi_torch,
            cut_local_cellxgene_ix=data.cut_local_cellxgene_ix,
            n_cells=data.n_cells,
            n_genes=data.n_genes,
            cut_local_gene_ix=data.cut_local_gene_ix,
            local_cellxgene_ix=data.local_cellxgene_ix,
        )

    def parameters_dense(self):
        return [
            parameter
            for module in self._modules.values()
            for parameter in (
                module.parameters_dense()
                if hasattr(module, "parameters_dense")
                else module.parameters()
            )
        ]

    def parameters_sparse(self):
        return [
            parameter
            for module in self._modules.values()
            for parameter in (
                module.parameters_sparse()
                if hasattr(module, "parameters_sparse")
                else []
            )
        ]

    def _get_likelihood_mixture_cell_gene(
        self, likelihood_mixture, cut_local_cellxgene_ix, n_cells, n_genes
    ):
        return torch_scatter.segment_sum_coo(
            likelihood_mixture, cut_local_cellxgene_ix, dim_size=n_cells * n_genes
        ).reshape((n_cells, n_genes))

    def forward_likelihood_mixture(self, data):
        self.forward(data)
        return self.track["likelihood_mixture"]

    def forward_likelihood(self, data):
        self.forward(data)
        likelihood_mixture_cell_gene = self._get_likelihood_mixture_cell_gene(
            self.track["likelihood_mixture"],
            data.cut_local_cellxgene_ix,
            data.n_cells,
            data.n_genes,
        )

        likelihood = (
            self.track["likelihood_fragmentcounts"] + likelihood_mixture_cell_gene
        )

        return likelihood

    def evaluate_pseudo(self, coordinates, latent=None, gene_oi=None):
        device = coordinates.device
        if not torch.is_tensor(latent):
            if latent is None:
                latent = 0.0
            latent = torch.ones((1, self.n_latent), device=coordinates.device) * latent
        if gene_oi is None:
            gene_oi = 0
        genes_oi = torch.tensor([gene_oi], device=coordinates.device, dtype=torch.long)
        cells_oi = torch.ones((1,), dtype=torch.long)
        local_cellxgene_ix = torch.tensor([], dtype=torch.long)
        cut_local_gene_ix = torch.zeros_like(coordinates).to(torch.long)
        cut_local_cellxgene_ix = torch.zeros_like(coordinates).to(torch.long)

        data = FullDict(
            local_cellxgene_ix=local_cellxgene_ix.to(device),
            cut_local_gene_ix=cut_local_gene_ix.to(device),
            cut_local_cellxgene_ix=cut_local_cellxgene_ix.to(device),
            cut_coordinates=coordinates.to(device),
            n_cells=1,
            n_genes=1,
            genes_oi_torch=genes_oi.to(device),
            cells_oi_torch=cells_oi.to(device),
            latent=latent.to(device),
        )
        return self.forward_likelihood_mixture(data)


class FullDict(dict):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

import chromatinhd.models.positional.v16 as positional_model

import torch
import numpy as np
import math
import tqdm.auto as tqdm
from normflows.utils import splines

from chromatinhd.embedding import EmbeddingTensor


class Decoder(torch.nn.Module):
    def __init__(
        self, n_latent, n_genes, n_spline_dimensions, n_layers=1, n_hidden_dimensions=32
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
            # layers.append(torch.nn.Dropout(0.5))
        self.nn = torch.nn.Sequential(*layers)

        self.n_hidden_dimensions = n_hidden_dimensions

        self.spline_weight = EmbeddingTensor(
            n_genes,
            (n_hidden_dimensions if n_layers > 0 else n_latent, n_spline_dimensions),
            sparse=True,
        )
        stdv = 1.0 / math.sqrt(self.spline_weight.weight.size(1)) / 100
        self.spline_weight.weight.data.uniform_(-stdv, stdv)

        self.rho_weight = EmbeddingTensor(
            n_genes, (n_hidden_dimensions if n_layers > 0 else n_latent,), sparse=True
        )
        stdv = 1.0 / math.sqrt(self.rho_weight.weight.size(1)) / 100
        self.rho_weight.weight.data.uniform_(-stdv, stdv)

    def forward(self, latent, genes_oi):
        nn_output = self.nn(latent)

        # spline
        # nn_output is broadcasted across genes and across components
        spline_weight = self.spline_weight(genes_oi)
        spline = torch.matmul(
            nn_output.unsqueeze(1).unsqueeze(2), spline_weight
        ).squeeze(-2)

        # expression
        rho_weight = self.rho_weight(genes_oi)
        rho = torch.matmul(nn_output, rho_weight.T).squeeze(-1)
        return spline, rho

    def parameters_dense(self):
        return [*self.nn.parameters()]

    def parameters_sparse(self):
        return [self.spline_weight.weight, self.rho_weight.weight]


class Mixture(torch.nn.Module):
    def __init__(
        self,
        n_bins,
        n_genes,
        window,
        loc_init=None,
        scale_init=None,
        logit_init=None,
        debug=False,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.register_buffer("a", torch.tensor(window[0]).unsqueeze(-1))
        self.register_buffer("b", torch.tensor(window[1]).unsqueeze(-1))
        self.register_buffer("ab", self.b - self.a)

        default_init = np.log(np.exp(1 - splines.DEFAULT_MIN_DERIVATIVE) - 1)

        self.n_spline_dimensions = 3 * n_bins - 1

        self.spline = EmbeddingTensor(n_genes, (self.n_spline_dimensions,), sparse=True)
        self.spline.weight.data[:, :] = 1.0 * default_init

    def log_prob(self, value, delta_spline, genes_oi, local_gene_ix):
        spline = self.spline(genes_oi)[local_gene_ix] + delta_spline
        widths = spline[..., : self.n_bins]  # * 10
        heights = spline[..., self.n_bins : self.n_bins * 2]  # * 10
        derivatives = spline[..., self.n_bins * 2 :]

        x = (((value - self.a) / self.ab) - 0.5) * 2
        x = x.unsqueeze(1)

        y_, logdet = splines.unconstrained_rational_quadratic_spline(
            x,
            widths.unsqueeze(1),
            heights.unsqueeze(1),
            derivatives.unsqueeze(1),
            tail_bound=1.0,
            inverse=True,
        )
        return math.log(0.5) + logdet - math.log(self.ab)

    def parameters_dense(self):
        return []

    def parameters_sparse(self):
        return [self.spline.weight]


class CellGeneEmbeddingToCellEmbedding(torch.nn.Module):
    def __init__(self, n_input_features, n_genes, n_output_features=None):
        if n_output_features is None:
            n_output_features = n_input_features

        super().__init__()

        # default initialization same as a torch.nn.Linear
        self.bias1 = torch.nn.Parameter(
            torch.empty(n_output_features, requires_grad=True)
        )
        self.bias1.data.zero_()

        self.n_output_features = n_output_features

        self.weight1 = EmbeddingTensor(
            n_genes, (n_input_features, self.n_output_features), sparse=True
        )
        stdv = 1.0 / math.sqrt(self.weight1.weight.size(-1))
        self.weight1.weight.data.uniform_(-stdv, stdv)

    def forward(self, cellgene_embedding, genes_oi):
        weight1 = self.weight1(genes_oi)
        bias1 = self.bias1
        return torch.einsum("abc,bcd->ad", cellgene_embedding, weight1) + bias1

    def parameters_dense(self):
        return [self.bias1]

    def parameters_sparse(self):
        return [self.weight1.weight]


class Encoder(torch.nn.Module):
    def __init__(self, n_input_features, n_latent, n_layers=1, n_hidden_dimensions=16):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                torch.nn.Linear(
                    n_hidden_dimensions if i > 0 else n_input_features,
                    n_hidden_dimensions,
                )
            )
            layers.append(torch.nn.ReLU())
            # layers.append(torch.nn.Dropout(0.5))
        self.nn = torch.nn.Sequential(*layers)

        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_latent = n_latent

        self.linear_loc = torch.nn.Linear(
            n_hidden_dimensions if n_layers > 0 else n_input_features, n_latent
        )
        self.linear_scale = torch.nn.Linear(
            n_hidden_dimensions if n_layers > 0 else n_input_features, n_latent
        )

    def forward(self, x):
        nn_output = self.nn(x)
        loc = self.linear_loc(nn_output)
        scale = self.linear_scale(nn_output).exp()
        return loc, scale

    def parameters_dense(self):
        return self.parameters()

    def parameters_sparse(self):
        return []


class VAE(torch.nn.Module):
    def __init__(
        self,
        fragments,
        n_latent_dimensions=10,
        n_bins=100,
        decoder_n_layers=2,
        n_frequencies=50,
    ):
        super().__init__()

        self.fragment_embedder = positional_model.FragmentEmbedder(
            fragments.n_genes, n_frequencies=n_frequencies, n_embedding_dimensions=20
        )
        self.embedding_gene_pooler = positional_model.EmbeddingGenePooler()

        self.cellgene_embedding_to_cell_embedding = CellGeneEmbeddingToCellEmbedding(
            self.fragment_embedder.n_embedding_dimensions, fragments.n_genes
        )

        self.n_latent_dimensions = n_latent_dimensions
        self.encoder = Encoder(
            self.cellgene_embedding_to_cell_embedding.n_output_features,
            self.n_latent_dimensions,
        )

        self.mixture = Mixture(n_bins, fragments.n_genes, fragments.window)

        self.decoder = Decoder(
            n_latent_dimensions,
            fragments.n_genes,
            self.mixture.n_spline_dimensions,
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

    def forward(self, data, track_elbo=True):
        # encode
        fragment_embedding = self.fragment_embedder.forward(
            data.coordinates, data.genemapping
        )
        cell_gene_embedding = self.embedding_gene_pooler.forward(
            fragment_embedding, data.local_cellxgene_ix, data.n_cells, data.n_genes
        )
        cell_embedding = self.cellgene_embedding_to_cell_embedding.forward(
            cell_gene_embedding,
            torch.from_numpy(data.genes_oi).to(cell_gene_embedding.device),
        )

        loc, scale = self.encoder(cell_embedding)

        # latent
        latent_q = torch.distributions.Normal(loc, scale * 0.1)
        latent_p = torch.distributions.Normal(0, 1)
        latent = latent_q.rsample()

        latent_kl = (latent_p.log_prob(latent) - latent_q.log_prob(latent)).sum()

        # decode
        genes_oi = torch.from_numpy(data.genes_oi).to(latent.device)
        spline, rho_change = self.decoder(latent, genes_oi)

        # fragment counts
        spline_cellxgene = spline.view(np.prod(spline.shape[:2]), spline.shape[-1])
        spline_fragments = spline_cellxgene[data.local_cellxgene_ix]

        likelihood_left = self.mixture.log_prob(
            data.coordinates[:, 0], spline_fragments, genes_oi, data.local_gene_ix
        )
        likelihood_right = self.mixture.log_prob(
            data.coordinates[:, 1], spline_fragments, genes_oi, data.local_gene_ix
        )

        likelihood_loci = likelihood_left.sum() + likelihood_right.sum()

        # expression
        # fragmentexpression = (self.rho_bias[data.genes_oi] * torch.exp(rho_change)) * self.libsize[data.cells_oi].unsqueeze(1)
        # fragmentcounts_p = torch.distributions.Poisson(fragmentexpression)
        # fragmentcounts = torch.reshape(torch.bincount(data.local_cellxgene_ix, minlength = data.n_cells * data.n_genes), (data.n_cells, data.n_genes))
        # likelihood_fragmentcounts = fragmentcounts_p.log_prob(fragmentcounts)

        # likelihood
        likelihood = likelihood_loci.sum()  # + likelihood_fragmentcounts.sum()

        # ELBO
        elbo = -likelihood - latent_kl

        return elbo

    def evaluate_latent(self, data):
        fragment_embedding = self.fragment_embedder.forward(
            data.coordinates, data.genemapping
        )

        cell_gene_embedding = self.embedding_gene_pooler.forward(
            fragment_embedding, data.local_cellxgene_ix, data.n_cells, data.n_genes
        )

        cell_embedding = self.cellgene_embedding_to_cell_embedding.forward(
            cell_gene_embedding,
            torch.from_numpy(data.genes_oi).to(cell_gene_embedding.device),
        )

        loc, scale = self.encoder(cell_embedding)

        return loc

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


class Decoding(torch.nn.Module):
    def __init__(self, fragments, cell_latent_space, n_bins=500, decoder_n_layers=2):
        super().__init__()

        self.mixture = Mixture(n_bins, fragments.n_genes, fragments.window)

        n_latent_dimensions = cell_latent_space.shape[1]

        self.decoder = Decoder(
            n_latent_dimensions,
            fragments.n_genes,
            self.mixture.n_spline_dimensions,
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

    def forward(self, data, latent=None, track_elbo=True):
        if latent is None:
            latent = self.cell_latent_space[data.cells_oi]

        # decode
        genes_oi = torch.from_numpy(data.genes_oi).to(latent.device)
        spline, rho_change = self.decoder(latent, genes_oi)

        # fragment counts
        spline_cellxgene = spline.view(np.prod(spline.shape[:2]), spline.shape[-1])
        spline_fragments = spline_cellxgene[data.local_cellxgene_ix]

        likelihood_left = self.mixture.log_prob(
            data.coordinates[:, 0], spline_fragments, genes_oi, data.local_gene_ix
        )
        likelihood_right = self.mixture.log_prob(
            data.coordinates[:, 1], spline_fragments, genes_oi, data.local_gene_ix
        )

        likelihood_loci = likelihood_left.sum() + likelihood_right.sum()

        # expression
        fragmentexpression = (
            self.rho_bias[data.genes_oi] * torch.exp(rho_change)
        ) * self.libsize[data.cells_oi].unsqueeze(1)
        fragmentcounts_p = torch.distributions.Poisson(fragmentexpression)
        fragmentcounts = torch.reshape(
            torch.bincount(
                data.local_cellxgene_ix, minlength=data.n_cells * data.n_genes
            ),
            (data.n_cells, data.n_genes),
        )
        likelihood_fragmentcounts = fragmentcounts_p.log_prob(fragmentcounts)

        # likelihood
        likelihood = likelihood_loci.sum() + likelihood_fragmentcounts.sum()

        # ELBO
        elbo = -likelihood

        return elbo

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

    def evaluate_pseudo(self, pseudocoordinates, pseudolatent, gene_oi):
        local_cellxgene_ix = torch.arange(len(pseudocoordinates)).to(
            pseudocoordinates.device
        )
        local_gene_ix = torch.zeros(len(pseudocoordinates), dtype=torch.long).to(
            pseudocoordinates.device
        )
        # decode
        genes_oi = torch.tensor(
            [gene_oi], dtype=torch.long, device=pseudocoordinates.device
        )

        spline, rho_change = self.decoder(pseudolatent, genes_oi)

        # fragment counts
        spline_cellxgene = spline.view(np.prod(spline.shape[:2]), spline.shape[-1])
        spline_fragments = spline_cellxgene[local_cellxgene_ix]

        # fragment counts
        likelihood_left = self.mixture.log_prob(
            pseudocoordinates, spline_fragments, genes_oi, local_gene_ix
        )

        return likelihood_left

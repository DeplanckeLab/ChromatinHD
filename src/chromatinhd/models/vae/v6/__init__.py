import torch
import numpy as np
import math
import tqdm.auto as tqdm
import torch_scatter

from chromatinhd.embedding import EmbeddingTensor

import chromatinhd.models.likelihood.v4 as likelihood_model


class Encoder(torch.nn.Module):
    def __init__(self, n_genes, n_latent, nbins=64, n_layers=1, n_hidden_dimensions=16):
        super().__init__()

        n_input_features = n_genes * nbins
        self.nbins = nbins

        layers = []
        # layers.append(torch.nn.BatchNorm1d(n_input_features))
        for i in range(n_layers):
            layers.append(
                torch.nn.Linear(
                    n_hidden_dimensions if i > 0 else n_input_features,
                    n_hidden_dimensions,
                )
            )
            layers.append(torch.nn.BatchNorm1d(n_hidden_dimensions))
            layers.append(torch.nn.ReLU())
        self.nn = torch.nn.Sequential(*layers)

        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_latent = n_latent

        self.linear_loc = torch.nn.Linear(
            n_hidden_dimensions if n_layers > 0 else n_input_features, n_latent
        )
        self.linear_scale = torch.nn.Linear(
            n_hidden_dimensions if n_layers > 0 else n_input_features, n_latent
        )

    def forward(self, cut_coordinates, cut_local_cellxgene_ix, genes_oi, cells_oi):
        assert (genes_oi.diff() == 1).all(), genes_oi
        digitized = (
            torch.clamp_max(
                torch.floor(cut_coordinates * self.nbins), self.nbins - 1
            ).int()
            + (cut_local_cellxgene_ix.int() * self.nbins).int()
        ).int()
        counts = torch.bincount(
            digitized, minlength=len(cells_oi) * len(genes_oi) * self.nbins
        )
        assert counts.shape == (len(cells_oi) * len(genes_oi) * self.nbins,), (
            len(cells_oi) * len(genes_oi) * self.nbins,
            counts.shape,
        )
        x = torch.log1p(
            counts.reshape((len(cells_oi), len(genes_oi) * self.nbins)).to(
                torch.float32
            )
        )

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
        n_encoder_bins=16,
        n_decoder_bins=(16,),
        encoder_n_hidden_dimensions=16,
        n_latent_dimensions=50,
        decoder_n_layers=0,
        decoder_n_hidden_dimensions=64,
        baseline=False,
    ):
        super().__init__()

        self.n_latent_dimensions = n_latent_dimensions
        self.encoder = Encoder(
            fragments.n_genes,
            self.n_latent_dimensions,
            nbins=n_encoder_bins if not baseline else 1,
            n_hidden_dimensions=encoder_n_hidden_dimensions,
        )

        transform = likelihood_model.spline.DifferentialQuadraticSplineStack(
            fragments.cut_coordinates,
            nbins=n_decoder_bins,
            local_gene_ix=fragments.cut_local_gene_ix.cpu(),
            n_genes=fragments.n_genes,
        )
        self.mixture = likelihood_model.spline.TransformedDistribution(transform)
        n_delta_mixture_components = sum(transform.split_deltas)

        if not baseline:
            self.decoder = likelihood_model.Decoder(
                n_latent_dimensions,
                fragments.n_genes,
                n_delta_mixture_components,
                n_layers=decoder_n_layers,
                n_hidden_dimensions=decoder_n_hidden_dimensions,
            )
        else:
            self.decoder = likelihood_model.BaselineDecoder(
                n_latent_dimensions,
                fragments.n_genes,
                n_delta_mixture_components,
                n_layers=decoder_n_layers,
                n_hidden_dimensions=decoder_n_hidden_dimensions,
            )

        libsize = torch.bincount(fragments.mapping[:, 0], minlength=fragments.n_cells)
        rho_bias = (
            torch.bincount(fragments.mapping[:, 1], minlength=fragments.n_genes)
            / fragments.n_cells
            / libsize.to(torch.float).mean()
        )

        self.register_buffer("libsize", libsize)
        self.register_buffer("rho_bias", rho_bias)

        self.track = {}

        self.baseline = baseline

        self.n_total_cells = fragments.n_cells
        self.n_total_cuts = fragments.cut_coordinates.shape[0]

    def forward_(
        self,
        coordinates,
        genemapping,
        local_cellxgene_ix,
        cut_coordinates,
        genes_oi,
        cells_oi,
        cut_local_cellxgene_ix,
        cut_local_gene_ix,
        n_cells,
        n_genes,
    ):
        # encode
        loc, scale = self.encoder(
            cut_coordinates, cut_local_cellxgene_ix, genes_oi, cells_oi
        )

        self.track["latent"] = loc

        # latent
        latent_q = torch.distributions.Normal(loc, scale * 0.1)
        latent_p = torch.distributions.Normal(0, 1)
        latent = latent_q.rsample()

        latent_kl = (
            (latent_p.log_prob(latent) - latent_q.log_prob(latent))
            * self.n_total_cells
            / n_cells
        )

        # decode
        mixture_delta, rho_delta = self.decoder(latent, genes_oi)

        # fragment counts
        mixture_delta_cellxgene = mixture_delta.view(
            np.prod(mixture_delta.shape[:2]), mixture_delta.shape[-1]
        )
        mixture_delta = mixture_delta_cellxgene[cut_local_cellxgene_ix]

        if self.baseline:
            mixture_delta_kl = torch.tensor(0.0, device=coordinates.device)
        else:
            # mixture_delta_kl = torch.tensor(0., device = coordinates.device)
            mixture_delta_p = torch.distributions.Normal(0.0, 0.1)
            mixture_delta_kl = mixture_delta_p.log_prob(
                self.decoder.logit_weight(genes_oi)
            )

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

        rho_delta_p = torch.distributions.Normal(0.0, 0.1)
        rho_delta_kl = rho_delta_p.log_prob(self.decoder.rho_weight(genes_oi))

        # likelihood
        likelihood = (
            likelihood_mixture.sum() * self.n_total_cuts / cut_coordinates.shape[0]
            + likelihood_fragmentcounts.sum() * self.n_total_cells / n_cells
        )

        # ELBO
        elbo = (
            -likelihood - latent_kl.sum() - mixture_delta_kl.sum() - rho_delta_kl.sum()
        )

        return elbo

    def forward(self, data):
        return self.forward_(
            coordinates=data.coordinates,
            genemapping=data.genemapping,
            cut_coordinates=data.cut_coordinates,
            cells_oi=data.cells_oi_torch,
            genes_oi=data.genes_oi_torch,
            cut_local_cellxgene_ix=data.cut_local_cellxgene_ix,
            n_cells=data.n_cells,
            n_genes=data.n_genes,
            cut_local_gene_ix=data.cut_local_gene_ix,
            local_cellxgene_ix=data.local_cellxgene_ix,
        )

    def forward_likelihood_mixture(self, data):
        # decode
        mixture_delta, rho_delta = self.decoder(data.latent, data.genes_oi_torch)

        # fragment counts
        mixture_delta_cellxgene = mixture_delta.view(
            np.prod(mixture_delta.shape[:2]), mixture_delta.shape[-1]
        )
        mixture_delta = mixture_delta_cellxgene[data.cut_local_cellxgene_ix]

        likelihood_mixture = self.track["likelihood_mixture"] = self.mixture.log_prob(
            data.cut_coordinates,
            data.genes_oi_torch,
            data.cut_local_gene_ix,
            mixture_delta,
        )

        return likelihood_mixture

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

    def evaluate_latent(self, data):
        # return self.latent[data.cells_oi]

        # fragment_embedding = self.fragment_embedder.forward(data.coordinates, data.genemapping)

        # cell_gene_embedding = self.embedding_gene_pooler.forward(fragment_embedding, data.local_cellxgene_ix, data.n_cells, data.n_genes)

        # cell_embedding = self.cellgene_embedding_to_cell_embedding.forward(cell_gene_embedding, torch.from_numpy(data.genes_oi).to(cell_gene_embedding.device))

        loc, scale = self.encoder(
            data.cut_coordinates,
            data.cut_local_cellxgene_ix,
            data.genes_oi_torch,
            data.cells_oi_torch,
        )

        return loc

    def evaluate_pseudo(self, coordinates, latent, cells_oi, n, gene_oi=None):
        device = coordinates.device
        if gene_oi is None:
            gene_oi = 0
        genes_oi = torch.tensor([gene_oi], device=coordinates.device, dtype=torch.long)
        cut_local_gene_ix = torch.zeros_like(coordinates).to(torch.long)
        n_cells = len(cells_oi)
        cut_local_cellxgene_ix = torch.arange(n_cells, device=coordinates.device).tile(
            n
        )

        data = FullDict(
            cut_local_gene_ix=cut_local_gene_ix.to(device),
            cut_local_cellxgene_ix=cut_local_cellxgene_ix.to(device),
            cut_coordinates=coordinates.to(device),
            n_cells=n_cells,
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

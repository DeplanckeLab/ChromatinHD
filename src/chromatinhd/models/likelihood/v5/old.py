import torch
import numpy as np
import math
import tqdm.auto as tqdm
import torch_scatter

from chromatinhd.embedding import EmbeddingTensor
from . import spline


class Decoder(torch.nn.Module):
    def __init__(
        self,
        n_latent,
        n_genes,
        n_output_components,
        n_layers=1,
        n_hidden_dimensions=32,
        dropout_rate=0.2,
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                torch.nn.Linear(
                    n_hidden_dimensions if i > 0 else n_latent, n_hidden_dimensions
                )
            )
            layers.append(torch.nn.BatchNorm1d(n_hidden_dimensions))
            layers.append(torch.nn.ReLU())
            if dropout_rate > 0.0:
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
        # self.logit_weight.weight.data.uniform_(-stdv, stdv)
        self.logit_weight.weight.data.zero_()

        self.rho_weight = EmbeddingTensor(
            n_genes, (n_hidden_dimensions if n_layers > 0 else n_latent,), sparse=True
        )
        stdv = 1.0 / math.sqrt(self.rho_weight.weight.size(1))
        self.rho_weight.weight.data.uniform_(-stdv, stdv)

    def forward(self, latent, genes_oi):
        logit_weight = self.logit_weight(genes_oi)
        rho_weight = self.rho_weight(genes_oi)
        nn_output = self.nn(latent)

        # nn_output is broadcasted across genes and across components
        logit = torch.matmul(nn_output.unsqueeze(1).unsqueeze(2), logit_weight).squeeze(
            -2
        )

        # nn_output has to be broadcasted across genes
        rho = torch.matmul(nn_output.unsqueeze(1), rho_weight.T).squeeze(-2)

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
            layers.append(torch.nn.BatchNorm1d(n_hidden_dimensions))
            layers.append(torch.nn.ReLU())
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
    # mixture_delta_p_scale = 1.0
    def __init__(
        self,
        fragments,
        cell_latent_space,
        nbins=(128,),
        decoder_n_layers=0,
        baseline=False,
        mixture_delta_p_scale_free=False,
        scale_likelihood=False,
        rho_delta_p_scale_free=False,
        mixture_delta_p_scale_dist="normal",
        mixture_delta_p_scale=1.0,
    ):
        super().__init__()

        n_latent_dimensions = cell_latent_space.shape[1]

        from .spline import DifferentialQuadraticSplineStack, TransformedDistribution

        transform = DifferentialQuadraticSplineStack(
            fragments.cut_coordinates,
            nbins=nbins,
            local_gene_ix=fragments.cut_local_gene_ix.cpu(),
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

        if mixture_delta_p_scale_free:
            self.mixture_delta_p_scale = torch.nn.Parameter(
                torch.log(
                    torch.tensor(math.log(mixture_delta_p_scale), requires_grad=True)
                )
            )
        else:
            self.register_buffer(
                "mixture_delta_p_scale", torch.tensor(math.log(mixture_delta_p_scale))
            )

        self.n_total_cells = fragments.n_cells

        self.scale_likelihood = scale_likelihood

        if rho_delta_p_scale_free:
            self.rho_delta_p_scale = torch.nn.Parameter(
                torch.log(torch.tensor(0.1, requires_grad=True))
            )
        else:
            self.register_buffer("rho_delta_p_scale", torch.tensor(0.0))

        self.mixture_delta_p_scale_dist = mixture_delta_p_scale_dist

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

        # scale likelihoods
        scale = 1.0
        if hasattr(self, "scale_likelihood") and self.scale_likelihood:
            scale = self.n_total_cells / n_cells

        # mixture kl
        # mixture_delta_p = torch.distributions.Laplace(0., 0.1)
        # mixture_delta_p = torch.distributions.Normal(0., 1.0)
        if self.mixture_delta_p_scale_dist == "normal":
            mixture_delta_p = torch.distributions.Normal(
                0.0, torch.exp(self.mixture_delta_p_scale)
            )
        else:
            mixture_delta_p = torch.distributions.Laplace(
                0.0, torch.exp(self.mixture_delta_p_scale)
            )
        mixture_delta_kl = mixture_delta_p.log_prob(self.decoder.logit_weight(genes_oi))

        # expression
        rho_delta_p = torch.distributions.Normal(0.0, torch.exp(self.rho_delta_p_scale))
        rho_delta_kl = rho_delta_p.log_prob(self.decoder.rho_weight(genes_oi))

        fragmentexpression = (
            self.rho_bias[genes_oi] * torch.exp(rho_delta)
        ) * self.libsize[cells_oi].unsqueeze(1)
        self.track["rho_delta"] = rho_delta
        fragmentcounts_p = torch.distributions.Poisson(fragmentexpression)
        fragmentcounts = torch.reshape(
            torch.bincount(local_cellxgene_ix, minlength=n_cells * n_genes),
            (n_cells, n_genes),
        )
        likelihood_fragmentcounts = self.track[
            "likelihood_fragmentcounts"
        ] = fragmentcounts_p.log_prob(fragmentcounts)

        # likelihood
        # likelihood = likelihood_mixture.sum() * scale + likelihood_fragmentcounts.sum() * scale + mixture_delta_kl.sum() * scale# * 0.5# + rho_delta_kl.sum()
        likelihood = (
            likelihood_mixture.sum() * scale + likelihood_fragmentcounts.sum() * scale
        )

        # ELBO
        elbo = -likelihood - mixture_delta_kl.sum() - rho_delta_kl.sum()

        return elbo / self.n_total_cells

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
        parameters = [
            parameter
            for module in self._modules.values()
            for parameter in (
                module.parameters_dense()
                if hasattr(module, "parameters_dense")
                else module.parameters()
            )
        ]

        def contains(x, y):
            return any([x is y_ for y_ in y])

        parameters.extend(
            [
                p
                for p in self.parameters()
                if (not contains(p, self.parameters_sparse()))
                and (not contains(p, parameters))
            ]
        )
        return parameters

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

    def evaluate_pseudo(self, coordinates, latent=None, gene_oi=None, gene_ix=None):
        device = coordinates.device
        if not torch.is_tensor(latent):
            if latent is None:
                latent = 0.0
            latent = torch.ones((1, self.n_latent), device=device) * latent

        cells_oi = torch.ones((1,), dtype=torch.long)

        local_cellxgene_ix = torch.tensor([], dtype=torch.long)
        if gene_ix is None:
            if gene_oi is None:
                gene_oi = 0
            genes_oi = torch.tensor([gene_oi], device=device, dtype=torch.long)
            # genes_oi = torch.arange(self.rho_bias.shape[0])
            cut_local_gene_ix = torch.zeros_like(coordinates).to(torch.long)
            cut_local_cellxgene_ix = torch.zeros_like(coordinates).to(torch.long)
        else:
            assert len(gene_ix) == len(coordinates)
            genes_oi = torch.unique(gene_ix)

            local_gene_mapping = torch.zeros(
                genes_oi.max() + 1, device=device, dtype=torch.long
            )
            local_gene_mapping.index_add_(
                0, genes_oi, torch.arange(len(genes_oi), device=device)
            )

            cut_local_gene_ix = local_gene_mapping[gene_ix]
            cut_local_cell_ix = torch.arange(
                latent.shape[0], device=cut_local_gene_ix.device
            )
            cut_local_cellxgene_ix = (
                cut_local_cell_ix * len(genes_oi) + cut_local_gene_ix
            )

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
        self.forward_likelihood_mixture(data)

        rho_delta = self.track["rho_delta"].flatten()[data.cut_local_cellxgene_ix]
        rho = torch.log(
            (self.rho_bias[genes_oi][data.cut_local_gene_ix].exp() * rho_delta.exp())
            * self.libsize.to(torch.float).mean()
        )
        rho2 = (
            self.rho_bias[genes_oi][data.cut_local_gene_ix] * rho_delta
        ) * self.libsize.to(torch.float).mean()

        return (
            self.track["likelihood_mixture"].detach().cpu(),
            rho_delta.detach().cpu(),
            rho.detach().cpu(),
            rho.detach().cpu() + self.track["likelihood_mixture"].detach().cpu(),
            rho2.detach().cpu() + self.track["likelihood_mixture"].detach().cpu(),
        )

    def rank(self, window, n_latent, device="cuda"):
        n_genes = self.rho_bias.shape[0]

        import pandas as pd
        from chromatinhd.utils import crossing

        self = self.to(device).eval()

        # create design for inference
        design_gene = pd.DataFrame({"gene_ix": np.arange(n_genes)})
        design_latent = pd.DataFrame({"active_latent": np.arange(n_latent)})
        design_coord = pd.DataFrame(
            {"coord": np.arange(window[0], window[1] + 1, step=25)}
        )
        design = crossing(design_gene, design_latent, design_coord)
        design["batch"] = np.floor(np.arange(design.shape[0]) / 10000).astype(int)

        # infer
        mixtures = []
        rho_deltas = []
        rhos = []
        probs = []
        probs2 = []
        for _, design_subset in tqdm.tqdm(design.groupby("batch")):
            pseudocoordinates = torch.from_numpy(design_subset["coord"].values).to(
                device
            )
            pseudocoordinates = (pseudocoordinates - window[0]) / (
                window[1] - window[0]
            )
            pseudolatent = torch.nn.functional.one_hot(
                torch.from_numpy(design_subset["active_latent"].values).to(device),
                n_latent,
            ).to(torch.float)
            gene_ix = torch.from_numpy(design_subset["gene_ix"].values).to(device)

            mixture, rho_delta, rho, prob, prob2 = self.evaluate_pseudo(
                pseudocoordinates.to(device),
                latent=pseudolatent.to(device),
                gene_ix=gene_ix,
            )

            mixtures.append(mixture.numpy())
            rho_deltas.append(rho_delta.numpy())
            rhos.append(rho.numpy())
            probs.append(prob.numpy())
            probs2.append(prob2.numpy())
        mixtures = np.hstack(mixtures)
        rho_deltas = np.hstack(rho_deltas)
        rhos = np.hstack(rhos)
        probs = np.hstack(probs)
        probs2 = np.hstack(probs2)

        mixtures = mixtures.reshape(
            (design_gene.shape[0], design_latent.shape[0], design_coord.shape[0])
        )
        rho_deltas = rho_deltas.reshape(
            (design_gene.shape[0], design_latent.shape[0], design_coord.shape[0])
        )
        rhos = rhos.reshape(
            (design_gene.shape[0], design_latent.shape[0], design_coord.shape[0])
        )
        probs = probs.reshape(
            (design_gene.shape[0], design_latent.shape[0], design_coord.shape[0])
        )
        probs2 = probs2.reshape(
            (design_gene.shape[0], design_latent.shape[0], design_coord.shape[0])
        )

        mixture_diff = mixtures - mixtures.mean(-2, keepdims=True)
        probs_diff = mixture_diff + rho_deltas  # - rho_deltas.mean(-2, keepdims = True)

        # apply a mask to regions with very low likelihood of a cut
        rho_cutoff = np.log(1.0)
        rho_cutoff = -np.inf
        mask = probs >= rho_cutoff

        probs_diff_masked = probs_diff.copy()
        probs_diff_masked[~mask] = -np.inf

        ## Single base-pair resolution
        # interpolate the scoring from above but now at single base pairs
        # we may have to smooth this in the future, particularly for very detailed models that already look at base pair resolution
        x = (design["coord"].values).reshape(
            (design_gene.shape[0], design_latent.shape[0], design_coord.shape[0])
        )

        def interpolate(
            x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
        ) -> torch.Tensor:
            a = (fp[..., 1:] - fp[..., :-1]) / (xp[..., 1:] - xp[..., :-1])
            b = fp[..., :-1] - (a.mul(xp[..., :-1]))

            indices = (
                torch.searchsorted(xp.contiguous(), x.contiguous(), right=False) - 1
            )
            indices = torch.clamp(indices, 0, a.shape[-1] - 1)
            slope = a.index_select(a.ndim - 1, indices)
            intercept = b.index_select(a.ndim - 1, indices)
            return x * slope + intercept

        desired_x = torch.arange(*window)

        probs_diff_interpolated = interpolate(
            desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs_diff)
        ).numpy()
        rhos_interpolated = interpolate(
            desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(rhos)
        ).numpy()

        # again apply a mask
        rho_cutoff = np.log(1.0)
        probs_diff_interpolated_masked = probs_diff_interpolated.copy()
        mask_interpolated = rhos_interpolated >= rho_cutoff
        probs_diff_interpolated_masked[~mask_interpolated] = -np.inf

        return probs_diff_interpolated_masked


class FullDict(dict):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

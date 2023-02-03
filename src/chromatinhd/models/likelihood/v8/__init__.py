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

        self.logit_weight = torch.nn.Parameter(
            torch.zeros(
                (
                    n_genes,
                    n_hidden_dimensions if n_layers > 0 else n_latent,
                    n_output_components,
                ),
                requires_grad=True,
            )
        )

        stdv = 1.0 / math.sqrt(self.logit_weight.size(1))
        if n_layers > 1:
            self.logit_weight.weight.data.uniform_(-stdv, stdv)
        else:
            self.logit_weight.data.zero_()

    def forward(self, latent):
        logit_weight = self.logit_weight
        nn_output = self.nn(latent)

        # nn_output is broadcasted across genes and across components
        logit = torch.matmul(nn_output.unsqueeze(1).unsqueeze(2), logit_weight).squeeze(
            -2
        )

        return (logit,)

    def parameters_dense(self):
        return [
            *self.nn.parameters(),
            self.logit_weight,
        ]

    def parameters_sparse(self):
        return []


class Decoding(torch.nn.Module):
    # mixture_delta_p_scale = 1.0
    def __init__(
        self,
        fragments,
        reflatent,
        reflatent_idx,
        nbins=(128,),
        decoder_n_layers=0,
        mixture_delta_p_scale_free=False,
        scale_likelihood=False,
        mixture_delta_p_scale_dist="normal",
        mixture_delta_p_scale=1.0,
    ):
        super().__init__()

        self.n_genes = fragments.n_genes

        self.n_latent = reflatent.shape[1]

        from .spline import DifferentialQuadraticSplineStack, TransformedDistribution

        transform = DifferentialQuadraticSplineStack(
            # fragments.cut_coordinates,
            nbins=nbins,
            # local_gene_ix=fragments.cut_local_gene_ix.cpu(),
            n_genes=fragments.n_genes,
        )
        self.mixture = TransformedDistribution(transform)
        n_delta_mixture_components = sum(transform.split_deltas)

        self.decoder = Decoder(
            self.n_latent,
            fragments.n_genes,
            n_delta_mixture_components,
            n_layers=decoder_n_layers,
        )

        libsize = torch.bincount(fragments.mapping[:, 0], minlength=fragments.n_cells)

        eps = 1e-5
        rho_bias = (
            torch.bincount(fragments.mapping[:, 1], minlength=fragments.n_genes)
            / fragments.n_cells
            / libsize.to(torch.float).mean()
        ) + eps

        self.register_buffer("libsize", libsize)
        self.register_buffer("rho_bias", rho_bias)

        self.register_buffer("reflatent", reflatent)
        self.register_buffer("reflatent_idx", reflatent_idx)

        self.track = {}

        if mixture_delta_p_scale_free:
            self.mixture_delta_p_scale = torch.nn.Parameter(
                torch.tensor(math.log(mixture_delta_p_scale), requires_grad=True)
            )
        else:
            self.register_buffer(
                "mixture_delta_p_scale", torch.tensor(math.log(mixture_delta_p_scale))
            )

        self.n_total_cells = fragments.n_cells

        self.scale_likelihood = scale_likelihood

        self.mixture_delta_p_scale_dist = mixture_delta_p_scale_dist

    def forward_(
        self,
        local_cellxgene_ix,
        cut_coordinates,
        cut_reflatent_idx,
        cells_oi,
        cut_local_cellxgene_ix,
        cut_local_gene_ix,
        cut_local_cell_ix,
        n_cells,
        lib=None,
    ):
        # decode
        (mixture_delta,) = self.decoder(self.reflatent)

        cut_reflatent_idx = cut_reflatent_idx.to(cut_coordinates.device)

        cut_local_reflatentxgene_ix = (
            cut_reflatent_idx * self.n_genes + cut_local_gene_ix
        )

        if lib is None:
            cut_lib = torch.log(self.libsize[cells_oi[cut_local_cell_ix]])
        else:
            cut_lib = torch.log(lib)

        cut_positions = (cut_coordinates + cut_local_gene_ix) / self.n_genes

        likelihood_mixture = self.track["likelihood_mixture"] = (
            self.mixture.log_prob(
                cut_positions,
                cut_local_reflatentxgene_ix,
                cut_local_gene_ix,
                cut_reflatent_idx,
                mixture_delta,
            )
            # + cut_lib
        )
        # print(cut_lib)

        # scale likelihoods
        scale = 1.0
        if hasattr(self, "scale_likelihood") and self.scale_likelihood:
            scale = self.n_total_cells / n_cells

        # mixture kl
        # mixture_delta_p = torch.distributions.Laplace(0.0, 0.1)
        # mixture_delta_p = torch.distributions.Normal(0.0, 1.0)
        if self.mixture_delta_p_scale_dist == "normal":
            mixture_delta_p = torch.distributions.Normal(
                0.0, torch.exp(self.mixture_delta_p_scale)
            )
        else:
            mixture_delta_p = torch.distributions.Laplace(
                0.0, torch.exp(self.mixture_delta_p_scale)
            )
        mixture_delta_kl = mixture_delta_p.log_prob(self.decoder.logit_weight)

        # likelihood
        likelihood = (likelihood_mixture).sum() * scale

        # ELBO
        elbo = -likelihood - mixture_delta_kl.sum()

        # return elbo / self.n_total_cells
        return elbo / n_cells

    def forward(self, data):
        if not hasattr(data, "cut_reflatent_idx"):
            data.cut_coordinates
            data.cut_reflatent_idx = self.reflatent_idx[
                data.cells_oi_torch[data.cut_local_cell_ix]
            ]
        return self.forward_(
            cut_coordinates=data.cut_coordinates,
            cut_reflatent_idx=data.cut_reflatent_idx,
            cells_oi=data.cells_oi_torch,
            cut_local_cellxgene_ix=data.cut_local_cellxgene_ix,
            n_cells=data.n_cells,
            cut_local_gene_ix=data.cut_local_gene_ix,
            local_cellxgene_ix=data.local_cellxgene_ix,
            cut_local_cell_ix=data.cut_local_cell_ix,
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

        # extend with any left over parameters that were not specified in parameters_dense or parameters_sparse
        def contains(x, y):
            return any([x is y_ for y_ in y])

        # parameters.extend(
        #     [
        #         p
        #         for p in self.parameters()
        #         if (not contains(p, self.parameters_sparse()))
        #         and (not contains(p, parameters))
        #     ]
        # )
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

    def evaluate_pseudo(self, coordinates, latent=None, gene_oi=None):
        device = coordinates.device
        if not torch.is_tensor(latent):
            if latent is None:
                latent = 0.0
            latent = (
                torch.ones((len(coordinates), self.n_latent), device=device) * latent
            )

        cut_reflatent_idx = torch.where(latent)[1]
        assert cut_reflatent_idx.shape == coordinates.shape

        cells_oi = torch.ones((1,), dtype=torch.long)

        local_cellxgene_ix = torch.tensor([], dtype=torch.long)
        # if gene_ix is None:
        if gene_oi is None:
            gene_oi = 0
        # genes_oi = torch.tensor([gene_oi], device=device, dtype=torch.long)
        genes_oi = torch.arange(self.n_genes, device=device)
        cut_local_gene_ix = torch.zeros_like(coordinates).to(torch.long) + gene_oi
        cut_local_cell_ix = torch.zeros_like(coordinates).to(torch.long)
        cut_local_cellxgene_ix = torch.zeros_like(coordinates).to(torch.long) + gene_oi

        data = FullDict(
            local_cellxgene_ix=local_cellxgene_ix.to(device),
            cut_local_gene_ix=cut_local_gene_ix.to(device),
            cut_local_cell_ix=cut_local_cell_ix.to(device),
            cut_local_cellxgene_ix=cut_local_cellxgene_ix.to(device),
            cut_coordinates=coordinates.to(device),
            cut_reflatent_idx=cut_reflatent_idx.to(device),
            n_cells=1,
            genes_oi_torch=genes_oi.to(device),
            cells_oi_torch=cells_oi.to(device),
        )
        with torch.no_grad():
            self.forward_likelihood_mixture(data)

        # rho_delta = self.track["rho_delta"].flatten()[data.cut_local_cellxgene_ix]
        rho_delta = torch.tensor(0.0)
        rho = torch.log(
            (self.rho_bias[genes_oi][data.cut_local_gene_ix].exp())
            * self.libsize.to(torch.float).mean()
        )
        prob2 = (
            self.rho_bias[genes_oi][data.cut_local_gene_ix] * rho_delta
        ) * self.libsize.to(torch.float).mean()

        return (
            self.track["likelihood_mixture"].detach().cpu(),
            rho_delta.detach().cpu(),
            rho.detach().cpu(),
            rho.detach().cpu() + self.track["likelihood_mixture"].detach().cpu(),
            prob2.detach().cpu() + self.track["likelihood_mixture"].detach().cpu(),
        )

    def rank(
        self, window, n_latent, device="cuda", how="probs_diff_masked", prob_cutoff=None
    ):
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
        probs = []
        rho_deltas = []
        rhos = []
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

            likelihood_mixture, rho_delta, rho, prob, prob2 = self.evaluate_pseudo(
                pseudocoordinates.to(device),
                latent=pseudolatent.to(device),
                gene_ix=gene_ix,
            )
            prob_mixture = likelihood_mixture.detach().cpu().numpy()
            rho_delta = rho_delta.detach().cpu().numpy()
            rho = rho.detach().cpu().numpy()

            probs.append(prob)
            probs2.append(prob2)
            rho_deltas.append(rho_delta)
            rhos.append(rho)
        probs = np.hstack(probs)
        rho_deltas = np.hstack(rho_deltas)
        rhos = np.hstack(rhos)
        probs2 = np.hstack(probs2)

        probs = probs.reshape(
            (design_gene.shape[0], design_latent.shape[0], design_coord.shape[0])
        )
        rho_deltas = rho_deltas.reshape(
            (design_gene.shape[0], design_latent.shape[0], design_coord.shape[0])
        )
        rhos = rhos.reshape(
            (design_gene.shape[0], design_latent.shape[0], design_coord.shape[0])
        )
        probs2 = probs2.reshape(
            (design_gene.shape[0], design_latent.shape[0], design_coord.shape[0])
        )

        # calculate the score we're gonna use: how much does the likelihood of a cut in a window change compared to the "mean"?
        probs_diff = probs - probs.mean(-2, keepdims=True)

        # apply a mask to regions with very low likelihood of a cut
        if prob_cutoff is None:
            prob_cutoff = np.log(1.0)
        mask = probs >= prob_cutoff

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
        # rhos_interpolated = interpolate(
        #     desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(rhos)
        # ).numpy()
        probs2_interpolated = interpolate(
            desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs2)
        ).numpy()

        if how == "probs_diff_masked":
            # again apply a mask
            rho_cutoff = np.log(1.0)
            probs_diff_interpolated_masked = probs_diff_interpolated.copy()
            mask_interpolated = probs2_interpolated >= rho_cutoff
            probs_diff_interpolated_masked[~mask_interpolated] = -np.inf

            return probs_diff_interpolated_masked
        elif how == "probs_diff":
            return probs_diff_interpolated
        elif how == "probs":
            return interpolate(
                desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs)
            ).numpy()
        elif how == "mixtures":
            return interpolate(
                desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs)
            ).numpy()


class FullDict(dict):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

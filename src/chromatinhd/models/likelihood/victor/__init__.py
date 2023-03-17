import torch
import numpy as np
import math
import tqdm.auto as tqdm
import torch_scatter

from chromatinhd.embedding import EmbeddingTensor
from . import spline
from chromatinhd.models import (
    HybridModel,
)  # HybridModel = Model with both sparse and dense parameters instead of just one set of parameters. This is custom made by me but based on code from the internet
from .spline import DifferentialQuadraticSplineStack, TransformedDistribution


class Decoder(torch.nn.Module):
    def __init__(
        self,
        n_genes,
        n_output_components,
        n_layers=1,
        n_hidden_dimensions=32,
        dropout_rate=0.2,
    ):
        super().__init__()

        # the normal way
        # self.height_baseline = torch.nn.Parameter(
        #     torch.zeros(
        #         (
        #             n_genes,
        #             n_output_components,
        #         )
        #     )
        # )

        # the sparse gradient way
        self.height_slope = EmbeddingTensor(
            n_genes,
            (1, n_output_components),
            sparse=True,
        )

        # the actual tensor underyling this embedding can be accessed using
        # self.height_slope.weight

        # initialize
        # all function ending with a _ work inplace, so they will replace the
        # .data is necessary here because self.height_slope.weight is a tensor that has `requires_grad == True`
        # but we don't want to track the gradients of the .zero_() operation, we actually want to set the data to zero
        # we can do that by using the .data attribute, which gives you a view of the tensor where gradients are no longer tracked
        # a view of a tensor is a tensor which accesses the same data (so the same memory) as the original tensor, without making a copy
        self.height_slope.weight.data.zero_()

        # do the same but for the "overall" slope
        self.overall_slope = EmbeddingTensor(n_genes, (1,), sparse=True)
        self.overall_slope.weight.data.zero_()

    def forward(self, latent, genes_oi):
        # genes oi is only used to get the heights
        # we extract the overall slope for all genes because we have to do softmax later
        height_slope = self.height_slope(genes_oi)
        overall_slope = self.overall_slope.get_full_weight()

        # you may have to do some broadcasting here
        # what we need is for each cell x gene x knot its height
        # and for each cell x gene its overall
        height = height_slope * latent
        overall = overall_slope * latent

        return height, overall

    def parameters_dense(self):
        # this parameter is optimized for all genes, because of library size
        # if one gene gets more accessible, all the other genes should get less accessible
        # NOTE: this is not standard torch procedures, as torch typically only works with .parameters()
        # We have to split the parameters in dense and sparse here because we want need two different torch optimizers for these two sets of parameters
        return [self.overall_slope.weight]

    def parameters_sparse(self):
        return [self.height_slope.weight]


class Decoding(torch.nn.Module, HybridModel):
    def __init__(
        self,
        fragments,
        latent,
        nbins=(128,),
        decoder_n_layers=0,
        baseline=False,
        height_slope_p_scale_free=False,
        scale_likelihood=False,
        rho_delta_p_scale_free=False,
        mixture_delta_p_scale_dist="normal",
        height_slope_p_scale=1.0,
    ):
        super().__init__()

        self.n_total_genes = fragments.n_genes

        transform = DifferentialQuadraticSplineStack(
            nbins=nbins, n_genes=fragments.n_genes
        )
        self.mixture = TransformedDistribution(transform)
        n_delta_mixture_components = sum(transform.split_deltas)

        self.decoder = Decoder(
            fragments.n_genes,
            n_delta_mixture_components,
            n_layers=decoder_n_layers,
        )

        libsize = torch.bincount(fragments.mapping[:, 0], minlength=fragments.n_cells)

        # calculate and store the baseline accessibility for each gene
        overall_baseline = (
            torch.bincount(fragments.mapping[:, 1], minlength=fragments.n_genes)
            / fragments.n_cells
            / libsize.to(torch.float).mean()
        )
        min_rho_bias = 1e-5
        overall_baseline = torch.log(
            min_rho_bias + (1 - min_rho_bias) * overall_baseline
        )

        # register some tensors as buffers, so they will be put on the correct device when `.to(device = ...)` is called
        self.register_buffer("overall_baseline", overall_baseline)
        self.register_buffer("libsize", libsize)

        self.register_buffer("latent", latent)

        self.register_buffer(
            "height_slope_p_scale", torch.tensor(math.log(height_slope_p_scale))
        )

        self.n_total_cells = fragments.n_cells

        self.scale_likelihood = scale_likelihood

        self.register_buffer("overall_slope_p_scale", torch.tensor(math.log(1.0)))

        self.overall_p_scale_dist = mixture_delta_p_scale_dist

    def forward_(
        self,
        local_cellxgene_ix,
        cut_coordinates,
        latent,
        genes_oi,
        cells_oi,
        cut_local_cellxgene_ix,
        cut_localcellxgene_ix,
        cut_local_gene_ix,
        n_cells,
        n_genes,
    ):
        # overall_delta: [cells, genes (all)]
        # height_delta: [cells, genes (oi), knots (32+64+128)]
        overall_delta, height_delta = self.decoder(latent.to(torch.float), genes_oi)

        # overall
        # overall_cellxgene: [cells, genes (all)]
        # cut_localcellxgene_ix: for each cut, the index
        overall_cellxgene = torch.nn.functional.softmax(
            self.overall_baseline + overall_delta, -1
        )
        overall = overall_cellxgene.flatten()[cut_localcellxgene_ix]

        # overall delta kl
        # overall_delta_p = torch.distributions.Normal(
        #     0.0, torch.exp(self.overall_slope_p_scale)
        # )
        # overall_delta_kl = overall_delta_p.log_prob(self.decoder.overall_slope.weight)

        # heights
        #
        height_cellxgene = height_delta.view(
            np.prod(height_delta.shape[:2]), height_delta.shape[-1]
        )
        height = height_cellxgene[cut_local_cellxgene_ix]

        likelihood_height = self.mixture.log_prob(
            cut_coordinates, genes_oi, cut_local_gene_ix, height
        )

        # overall likelihood
        likelihood = self.track["likelihood"] = (
            likelihood_height + torch.log(rho_cuts) + math.log(self.n_total_genes)
        )
        likelihood_scale = 1.0

        # height kl
        height_delta_p = torch.distributions.Normal(
            0.0, torch.exp(self.height_slope_p_scale)
        )
        height_delta_kl = height_delta_p.log_prob(
            self.decoder.height_baseline(genes_oi)
        )

        # ELBO
        elbo = (
            -likelihood.sum()
            * likelihood_scale
            # - overall_delta_kl.sum()
            # - height_delta_kl.sum()
        )

        return elbo

    def forward(self, data):
        if not hasattr(data, "latent"):
            data.latent = self.latent[data.cells_oi]
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
            cut_localcellxgene_ix=data.cut_localcellxgene_ix,
        )

    def _get_likelihood_mixture_cell_gene(
        self, likelihood_mixture, cut_local_cellxgene_ix, n_cells, n_genes
    ):
        return torch_scatter.segment_sum_coo(
            likelihood_mixture, cut_local_cellxgene_ix, dim_size=n_cells * n_genes
        ).reshape((n_cells, n_genes))

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
            # genes_oi = torch.arange(self.overall_baseline.shape[0])
            cut_local_gene_ix = torch.zeros_like(coordinates).to(torch.long)
            cut_local_cellxgene_ix = torch.zeros_like(coordinates).to(torch.long)
            cut_localcellxgene_ix = (
                torch.ones_like(coordinates).to(torch.long) * gene_oi
            )
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
            cut_localcellxgene_ix = cut_local_cell_ix * self.n_total_genes + gene_ix

        data = FullDict(
            local_cellxgene_ix=local_cellxgene_ix.to(device),
            cut_local_gene_ix=cut_local_gene_ix.to(device),
            cut_local_cellxgene_ix=cut_local_cellxgene_ix.to(device),
            cut_localcellxgene_ix=cut_localcellxgene_ix.to(device),
            cut_coordinates=coordinates.to(device),
            n_cells=1,
            n_genes=1,
            genes_oi_torch=genes_oi.to(device),
            cells_oi_torch=cells_oi.to(device),
            latent=latent.to(device),
        )
        with torch.no_grad():
            self.forward(data)

        prob = self.track["likelihood"].detach().cpu()
        return prob.detach().cpu()

    def rank(
        self,
        window: np.ndarray,
        n_latent: int,
        device: torch.DeviceObjType = "cuda",
        how: str = "probs_diff_masked",
        prob_cutoff: float = None,
    ) -> np.ndarray:
        n_genes = self.overall_baseline.shape[0]

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
        batch_size = 100000
        design["batch"] = np.floor(np.arange(design.shape[0]) / batch_size).astype(int)

        # infer
        probs = []
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

            prob = self.evaluate_pseudo(
                pseudocoordinates.to(device),
                latent=pseudolatent.to(device),
                gene_ix=gene_ix,
            )

            probs.append(prob)
        probs = np.hstack(probs)

        probs = probs.reshape(
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
        probs_interpolated = interpolate(
            desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs)
        ).numpy()

        if how == "probs_diff_masked":
            # again apply a mask
            probs_diff_interpolated_masked = probs_diff_interpolated.copy()
            mask_interpolated = probs_interpolated >= prob_cutoff
            probs_diff_interpolated_masked[~mask_interpolated] = -np.inf

            return probs_diff_interpolated_masked
        else:
            raise NotImplementedError()


class FullDict(dict):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

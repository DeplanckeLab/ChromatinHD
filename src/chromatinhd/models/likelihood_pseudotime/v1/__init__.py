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
        n_delta_height,
    ):
        super().__init__()

        # the sparse gradient way
        self.delta_height_slope = EmbeddingTensor(
            n_genes,
            (n_delta_height,),
            sparse=True,
        )

        # initialize
        # all function ending with a _ work inplace, so they will replace the
        # .data is necessary here because self.delta_height_slope.weight is a tensor that has `requires_grad == True`
        # but we don't want to track the gradients of the .zero_() operation, we actually want to set the data to zero
        # we can do that by using the .data attribute, which gives you a view of the tensor where gradients are no longer tracked
        # a view of a tensor is a tensor which accesses the same data (so the same memory) as the original tensor, without making a copy
        self.delta_height_slope.weight.data.zero_()

        # do the same but for the "delta overall" slope
        self.delta_overall_slope = EmbeddingTensor(n_genes, (1,), sparse=True)
        self.delta_overall_slope.weight.data.zero_()

    def forward(self, latent, genes_oi):
        # genes oi is only used to get the deltas
        # we extract the overall slope for all genes because we have to do softmax later
        delta_height_slope = self.delta_height_slope(genes_oi)
        delta_overall_slope = self.delta_overall_slope.get_full_weight()

        #! you will have to do some broadcasting here
        # what we need is for each cell x gene x knot its delta
        # and for each cell x gene its overall
        delta_height = delta_height_slope * latent
        delta_overall = delta_overall_slope * latent

        return delta_height, delta_overall

    def parameters_dense(self):
        # this parameter is optimized for all genes, because of library size
        # if one gene gets more accessible, all the other genes should get less accessible
        # NOTE: this is not standard torch procedures, as torch typically only works with .parameters()
        # We have to split the parameters in dense and sparse here because we want need two different torch optimizers for these two sets of parameters
        return [self.delta_overall_slope.weight]

    def parameters_sparse(self):
        return [self.delta_height_slope.weight]


class Model(torch.nn.Module, HybridModel):
    def __init__(
        self,
        fragments,
        latent,
        nbins=(128,),
        scale_likelihood=False,
        height_slope_p_scale=1.0,
    ):
        super().__init__()

        self.n_total_genes = fragments.n_genes

        transform = DifferentialQuadraticSplineStack(
            nbins=nbins, n_genes=fragments.n_genes
        )
        self.nf = TransformedDistribution(transform)

        n_delta_height = sum(transform.split_deltas)

        self.decoder = Decoder(
            fragments.n_genes,
            n_delta_height=n_delta_height,
        )

        # calculate library size for each cell
        libsize = torch.bincount(fragments.mapping[:, 0], minlength=fragments.n_cells)
        self.register_buffer("libsize", libsize)

        # calculate and store the baseline accessibility for each gene
        # overall_baseline: [genes]
        overall_baseline = (
            torch.bincount(fragments.mapping[:, 1], minlength=fragments.n_genes)
            / fragments.n_cells
            / libsize.to(torch.float).mean()
        )
        min_overall_baseline = 1e-5
        overall_baseline = torch.log(
            min_overall_baseline + (1 - min_overall_baseline) * overall_baseline
        )
        self.register_buffer("overall_baseline", overall_baseline)

        self.register_buffer("latent", latent)

        self.register_buffer(
            "height_slope_p_scale", torch.tensor(math.log(height_slope_p_scale))
        )

        self.n_total_cells = fragments.n_cells

        self.scale_likelihood = scale_likelihood

        self.register_buffer("overall_slope_p_scale", torch.tensor(math.log(1.0)))

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
        self.track = {}

        # overall_delta: [cells, genes (all)]
        # height_delta: [cells, genes (oi), knots (32+64+128)]
        overall_delta, height_delta = self.decoder(latent.to(torch.float), genes_oi)

        # overall_cellxgene: [cells, genes (all)]
        # cut_localcellxgene_ix: for each cut, the global cell index x local gene index
        # overall: [cellxgene]
        overall_cellxgene = torch.nn.functional.log_softmax(
            self.overall_baseline + overall_delta, -1
        )
        overall_likelihood = overall_cellxgene.flatten()[
            cut_localcellxgene_ix
        ] + math.log(self.n_total_genes)

        # overall delta kl
        # overall_delta_p = torch.distributions.Normal(
        #     0.0, torch.exp(self.overall_slope_p_scale)
        # )
        # overall_delta_kl = overall_delta_p.log_prob(self.decoder.overall_slope.weight)

        # height
        height_delta_cellxgene = height_delta.view(
            np.prod(height_delta.shape[:2]), height_delta.shape[-1]
        )
        height_delta = height_delta_cellxgene[cut_local_cellxgene_ix]

        height_likelihood = self.nf.log_prob(
            cut_coordinates, genes_oi, cut_local_gene_ix, height_delta=height_delta
        )

        # overall likelihood
        likelihood = self.track["likelihood"] = height_likelihood + overall_likelihood

        # height kl
        # delta_p = torch.distributions.Normal(
        #     0.0, torch.exp(self.height_slope_p_scale)
        # )
        # delta_kl = delta_p.log_prob(
        #     self.decoder.height_baseline(genes_oi)
        # )

        # ELBO
        elbo = (
            -likelihood.sum()
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

import math
import torch
import torch_scatter
import dataclasses
import numpy as np
import tqdm.auto as tqdm

# HybridModel = Model with both sparse and dense parameters instead of just one set of parameters. This is custom made by me but based on code from the internet
from chromatinhd.models import HybridModel
from chromatinhd.embedding import EmbeddingTensor
from . import spline
from .spline import DifferentialQuadraticSplineStack, TransformedDistribution


class Decoder(torch.nn.Module):
    def __init__(self, n_genes, n_delta_height):
        # print("---  Decoder.__init__()  ---")
        super().__init__()

        # the sparse gradient way
        # initialize
        # all function ending with a _ work inplace, so they will replace the
        # .data is necessary here because self.delta_height_slope.weight is a tensor that has `requires_grad == True`
        # but we don't want to track the gradients of the .zero_() operation, we actually want to set the data to zero
        # we can do that by using the .data attribute, which gives you a view of the tensor where gradients are no longer tracked
        # a view of a tensor is a tensor which accesses the same data (so the same memory) as the original tensor, without making a copy
        self.delta_height_slope = EmbeddingTensor(n_genes, (n_delta_height,), sparse=True)
        self.delta_height_slope.weight.data.zero_()
        # print(f"{self.delta_height_slope.shape=}")

        self.delta_height_scale = EmbeddingTensor(n_genes, (n_delta_height,), sparse=True)
        self.delta_height_scale.weight.data.zero_()
        # print(f"{self.delta_height_scale.shape=}")

        self.delta_height_shift = EmbeddingTensor(n_genes, (n_delta_height,), sparse=True)
        self.delta_height_shift.weight.data.zero_()
        # print(f"{self.delta_height_shift.shape=}")

        # do the same but for the "delta overall" slope
        self.delta_overall_slope = EmbeddingTensor(n_genes, (1,), sparse=True)
        self.delta_overall_slope.weight.data.zero_()
        # print(f"{self.delta_overall_slope.shape=}")

    def forward(self, latent, genes_oi):
        # print("---  Decoder.forward()  ---")
        # genes oi is only used to get the deltas
        # we extract the overall slope for all genes because we have to do softmax later
        delta_height_slope = self.delta_height_slope(genes_oi)
        delta_height_scale = self.delta_height_scale(genes_oi)
        delta_height_shift = self.delta_height_shift(genes_oi)
        delta_overall_slope = self.delta_overall_slope.get_full_weight()

        #! you will have to do some broadcasting here
        # what we need is for each cell x gene x knot its delta
        # and for each cell x gene its overall
        latent = latent.unsqueeze(0).unsqueeze(0)
        # print(f"{latent.shape=}")

        delta_overall_slope = delta_overall_slope.unsqueeze(2)
        # print(f"{delta_overall_slope.shape=}")

        delta_overall = delta_overall_slope * latent
        # print(f"{delta_overall.shape=}")

        ###
        delta_height_slope = delta_height_slope.unsqueeze(2)
        # print(f"{delta_height_slope.shape=}")

        delta_height_scale = delta_height_scale.unsqueeze(2)
        # print(f"{delta_height_scale.shape=}")

        delta_height_shift = delta_height_shift.unsqueeze(2)
        # print(f"{delta_height_shift.shape=}")

        # https://pytorch.org/docs/stable/special.html#torch.special.expit
        delta_height = delta_height_slope/(1+torch.exp(-torch.exp(delta_height_scale) * latent - delta_height_shift))
        # print(f"{delta_height.shape=}")

        return delta_height, delta_overall

    def parameters_dense(self):
        # this parameter is optimized for all genes, because of library size
        # if one gene gets more accessible, all the other genes should get less accessible
        # NOTE: this is not standard torch procedures, as torch typically only works with .parameters()
        # We have to split the parameters in dense and sparse here because we want need two different torch optimizers for these two sets of parameters
        return [self.delta_overall_slope.weight]

    def parameters_sparse(self):
        return [self.delta_height_slope.weight, self.delta_height_scale.weight, self.delta_height_shift.weight]


# likelihoodresult = namedtuple("likelihoodresult", ["likelihood", "height_likelihood", "overall_likelihood"])
# result = likelihoodresult(likelihood=torch.tensor(1), height_likelihood=torch.tensor(1), overall_likelihood=torch.tensor(1))
# overall, height, total = result


@dataclasses.dataclass
class LikelihoodResult():
    overall:torch.Tensor
    height:torch.Tensor
    total:torch.Tensor


class Model(torch.nn.Module, HybridModel):
    def __init__(self, fragments, latent, nbins=(128,), scale_likelihood=False, height_slope_p_scale=1.0):
        super().__init__()
        # print("---  Model.__init__()  ---")

        transform = DifferentialQuadraticSplineStack(nbins=nbins, n_genes=fragments.n_genes)
        n_delta_height = sum(transform.split_deltas)

        self.nf = TransformedDistribution(transform)
        self.decoder = Decoder(fragments.n_genes, n_delta_height=n_delta_height)
        self.n_total_genes = fragments.n_genes
        self.n_total_cells = fragments.n_cells
        self.scale_likelihood = scale_likelihood

        # calculate library size for each cell
        libsize = torch.bincount(fragments.mapping[:, 0], minlength=fragments.n_cells)

        # calculate and store the baseline accessibility for each gene
        # overall_baseline: [genes]
        min_overall_baseline = 1e-5
        overall_baseline = (torch.bincount(fragments.mapping[:, 1], minlength=fragments.n_genes) / fragments.n_cells / libsize.to(torch.float).mean())
        overall_baseline = torch.log(min_overall_baseline + (1 - min_overall_baseline) * overall_baseline)

        self.register_buffer("latent", latent)
        self.register_buffer("libsize", libsize)
        self.register_buffer("overall_baseline", overall_baseline)
        self.register_buffer("height_slope_p_scale", torch.tensor(math.log(height_slope_p_scale)))
        self.register_buffer("overall_slope_p_scale", torch.tensor(math.log(1.0)))

    def forward_(self, cut_coordinates, latent, genes_oi, cut_local_cellxgene_ix, cut_localcellxgene_ix, cut_local_gene_ix, return_likelihood=False):
        # print("---  Model.forward_()  ---")

        # import matplotlib.pyplot as plt
        # plt.hist(latent)
        # raise ValueError()

        self.track = {}

        # overall_delta: [cells, genes (all)]
        # height_delta: [cells, genes (oi), knots (32+64+128)]
        height_delta, overall_delta = self.decoder(latent.to(torch.float), genes_oi)

        # overall_cellxgene: [cells, genes (all)]
        # cut_localcellxgene_ix: for each cut, the global cell index x local gene index
        # overall: [cellxgene]
        overall_baseline = self.overall_baseline.unsqueeze(0)
        overall_cellxgene = torch.nn.functional.log_softmax(overall_baseline + overall_delta.squeeze(1).transpose(1, 0), -1)
        overall_likelihood = overall_cellxgene.flatten()[cut_localcellxgene_ix] + math.log(self.n_total_genes)

        # height
        height_delta = height_delta.permute(2, 0, 1)
        height_delta_cellxgene = height_delta.reshape(np.prod(height_delta.shape[:2]), height_delta.shape[-1])
        height_delta = height_delta_cellxgene[cut_local_cellxgene_ix]
        height_likelihood = self.nf.log_prob(cut_coordinates, genes_oi, cut_local_gene_ix, height_delta)

        # overall likelihood
        # TODO: check this variable
        # print(f"{height_likelihood=}")
        # print(f"{overall_likelihood=}")
        likelihood = height_likelihood + overall_likelihood

        # ELBO
        elbo = (-likelihood.sum())

        if return_likelihood:
            return LikelihoodResult(overall=overall_likelihood, height=height_likelihood, total=likelihood)
        else:
            return elbo

    def forward(self, data, return_likelihood=False):
        # print("---  Model.forward()  ---")
        if not hasattr(data, "latent"):
            data.latent = self.latent[data.cells_oi]
        return self.forward_(
            cut_coordinates=data.cut_coordinates,
            latent=data.latent,
            genes_oi=data.genes_oi_torch,
            cut_local_cellxgene_ix=data.cut_local_cellxgene_ix,
            cut_local_gene_ix=data.cut_local_gene_ix,
            cut_localcellxgene_ix=data.cut_localcellxgene_ix,
            return_likelihood=return_likelihood,
        )
    
    def evaluate_pseudo(self, coordinate_oi, latent_oi, gene_oi=None, gene_ix=None):
        # print("---  Model.evaluate_pseudo()  ---")

        index_tensor = torch.arange(len(latent_oi))
        index_tensor_repeat = index_tensor.repeat_interleave(len(coordinate_oi))

        index_gene =  index_tensor * self.n_total_genes + gene_oi
        index_gene_repeat = index_gene.repeat_interleave(len(coordinate_oi))

        coordinate_oi_lt = coordinate_oi.repeat(len(latent_oi))
        
        device = coordinate_oi.device # on which device is coordinates stored, other data needs to be on the same device

        if gene_ix is None:
            if gene_oi is None:
                gene_oi = 0
            # single gene provided as integer, convert to tensor
            genes_oi = torch.tensor([gene_oi], device=device, dtype=torch.long)
            # tensor with 0, same shape as coordinate_oi, in this case single value
            cut_local_gene_ix = torch.zeros_like(coordinate_oi_lt).to(torch.long)
            # tensor with 0, same shape as coordinate_oi, in this case single value
            cut_local_cellxgene_ix = index_tensor_repeat.to(torch.long)
            # tensor with 0, same shape as coordinate_oi, in this case single value
            cut_localcellxgene_ix = index_gene_repeat.to(torch.long)

        data = FullDict(
            cut_local_gene_ix=cut_local_gene_ix.to(device),
            cut_local_cellxgene_ix=cut_local_cellxgene_ix.to(device),
            cut_localcellxgene_ix=cut_localcellxgene_ix.to(device),
            cut_coordinates=coordinate_oi_lt.to(device),
            genes_oi_torch=genes_oi.to(device),
            latent=latent_oi.to(device),
        )

        with torch.no_grad():
            result = self.forward(data, return_likelihood=True)

        return result


class FullDict(dict):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return str(self.__dict__)
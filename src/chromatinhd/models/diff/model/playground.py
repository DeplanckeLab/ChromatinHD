import math
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
import torch_scatter
import tqdm.auto as tqdm
import xarray as xr

from chromatinhd import get_default_device
from chromatinhd.data.clustering import Clustering
from chromatinhd.data.fragments import Fragments
from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.flow import Flow, Stored
from chromatinhd.loaders import LoaderPool
from chromatinhd.models import HybridModel
from chromatinhd.models.diff.loader.clustering_cuts import ClusteringCuts
from chromatinhd.models.diff.loader.minibatches import Minibatcher
from chromatinhd.models.diff.trainer import Trainer
from chromatinhd.optim import SparseDenseAdam
from chromatinhd.utils import crossing

from .spline import DifferentialQuadraticSplineStack, TransformedDistribution


class Decoder(torch.nn.Module):
    def __init__(
        self,
        n_latent,
        n_genes,
        n_output_components,
    ):
        super().__init__()

        self.n_output_components = n_output_components

        self.delta_height_weight = EmbeddingTensor(
            n_genes,
            (n_output_components,),
            sparse=True,
        )
        stdv = 1.0 / math.sqrt(self.delta_height_weight.weight.size(1))
        # self.delta_height_weight.weight.data.uniform_(-stdv, stdv)
        self.delta_height_weight.weight.data.zero_()

        self.delta_baseline_weight = torch.nn.Parameter(torch.zeros((n_genes, n_latent)))
        stdv = 1.0 / math.sqrt(self.delta_baseline_weight.weight.size(1))
        self.delta_baseline_weight.data.uniform_(-stdv, stdv)

    def forward(self, latent, genes_oi):
        # genes oi is only used to get the delta_heights
        # we calculate the delta_baseline for all genes because we pool using softmax later
        delta_height_weight = self.delta_height_weight(genes_oi)
        delta_baseline_weight = self.delta_baseline_weight.get_full_weight()

        # nn_output is broadcasted across genes and across components
        delta_height = torch.matmul(latent.unsqueeze(1).unsqueeze(2), delta_height_weight).squeeze(-2)

        # nn_output has to be broadcasted across genes
        delta_baseline = torch.matmul(latent.unsqueeze(1), delta_baseline_weight.T).squeeze(-2)

        return delta_height, delta_baseline

    def parameters_dense(self):
        return [self.delta_baseline_weight]

    def parameters_sparse(self):
        return [self.delta_height_weight.weight]


class Model(torch.nn.Module, HybridModel):
    """
    A ChromatinHD-diff model that models the probability density of observing a cut site between clusterings
    """

    def __init__(
        self,
        fragments,
        clustering,
        nbins=(
            128,
            64,
            32,
        ),
        decoder_n_layers=0,
        scale_likelihood=False,
        baseline_delta_regularization=True,
        baseline_delta_p_scale_free=False,
        mixture_delta_regularization=True,
        mixture_delta_p_scale_free=False,
        mixture_delta_p_scale_dist="normal",
        mixture_delta_p_scale=1.0,
    ):
        super().__init__()

        self.n_total_genes = fragments.n_genes

        self.n_clusters = clustering.n_clusters

        transform = DifferentialQuadraticSplineStack(
            nbins=nbins,
            n_genes=fragments.n_genes,
        )
        self.mixture = TransformedDistribution(transform)
        n_delta_mixture_components = sum(transform.split_deltas)

        self.decoder = Decoder(
            self.n_clusters,
            fragments.n_genes,
            n_delta_mixture_components,
            n_layers=decoder_n_layers,
        )

        # calculate baseline bias and libsizes
        libsize = torch.bincount(fragments.mapping[:, 0], minlength=fragments.n_cells)

        baseline_bias = (
            torch.bincount(fragments.mapping[:, 1], minlength=fragments.n_genes)
            / fragments.n_cells
            / libsize.to(torch.float).mean()
        )
        min_baseline_bias = 1e-5
        baseline_bias = min_baseline_bias + (1 - min_baseline_bias) * baseline_bias
        self.register_buffer("baseline_bias", baseline_bias)

        self.track = {}

        self.mixture_delta_regularization = mixture_delta_regularization
        if self.mixture_delta_regularization:
            if mixture_delta_p_scale_free:
                self.mixture_delta_p_scale = torch.nn.Parameter(
                    torch.tensor(math.log(mixture_delta_p_scale), requires_grad=True)
                )
            else:
                self.register_buffer(
                    "mixture_delta_p_scale",
                    torch.tensor(math.log(mixture_delta_p_scale)),
                )
        self.mixture_delta_p_scale_dist = mixture_delta_p_scale_dist

        self.baseline_delta_regularization = baseline_delta_regularization
        if self.baseline_delta_regularization:
            if baseline_delta_p_scale_free:
                self.baseline_delta_p_scale = torch.nn.Parameter(torch.log(torch.tensor(0.1, requires_grad=True)))
            else:
                self.register_buffer("baseline_delta_p_scale", torch.tensor(math.log(1.0)))

    def forward_(
        self,
        coordinates,
        clustering,
        genes_oi,
        local_cellxgene_ix,
        localcellxgene_ix,
        local_gene_ix,
    ):
        # decode
        mixture_delta, baseline_delta = self.decoder(clustering, genes_oi)

        # baseline
        baseline = torch.nn.functional.softmax(torch.log(self.baseline_bias) + baseline_delta, -1)
        baseline_cuts = baseline.flatten()[localcellxgene_ix]

        # fragment counts
        mixture_delta_cellxgene = mixture_delta.view(np.prod(mixture_delta.shape[:2]), mixture_delta.shape[-1])
        mixture_delta = mixture_delta_cellxgene[local_cellxgene_ix]

        self.track["likelihood_position"] = likelihood_position = self.mixture.log_prob(
            coordinates, genes_oi, local_gene_ix, mixture_delta
        )

        self.track["likelihood_region"] = likelihood_region = torch.log(baseline_cuts) + math.log(self.n_total_genes)

        # likelihood
        likelihood = self.track["likelihood"] = likelihood_position + likelihood_region

        elbo = -likelihood.sum()

        # regularization
        # mixture
        if self.mixture_delta_regularization:
            mixture_delta_p = torch.distributions.Normal(0.0, torch.exp(self.mixture_delta_p_scale))
            mixture_delta_kl = mixture_delta_p.log_prob(self.decoder.delta_height_weight(genes_oi))

            elbo -= mixture_delta_kl.sum()

        # baseline delta
        if self.baseline_delta_regularization:
            baseline_delta_p = torch.distributions.Normal(0.0, torch.exp(self.baseline_delta_p_scale))
            baseline_delta_kl = baseline_delta_p.log_prob(self.decoder.delta_baseline_weight(genes_oi))

            elbo -= baseline_delta_kl.sum()

        return elbo

    def forward(self, data):
        return self.forward_(
            coordinates=data.cuts.coordinates,
            clustering=data.clustering.onehot,
            genes_oi=data.minibatch.genes_oi_torch,
            local_gene_ix=data.cuts.local_gene_ix,
            local_cellxgene_ix=data.cuts.local_cellxgene_ix,
            localcellxgene_ix=data.cuts.localcellxgene_ix,
        )

    def train_model(self, fragments, clustering, fold, device=None, n_epochs=30, lr=1e-2):
        """
        Trains the model
        """

        if device is None:
            device = get_default_device()

        # set up minibatchers and loaders
        minibatcher_train = Minibatcher(
            fold["cells_train"],
            range(fragments.n_genes),
            n_genes_step=500,
            n_cells_step=200,
        )
        minibatcher_validation = Minibatcher(
            fold["cells_validation"],
            range(fragments.n_genes),
            n_genes_step=10,
            n_cells_step=10000,
            permute_cells=False,
            permute_genes=False,
        )

        loaders_train = LoaderPool(
            ClusteringCuts,
            dict(
                clustering=clustering,
                fragments=fragments,
                cellxgene_batch_size=minibatcher_train.cellxgene_batch_size,
            ),
            n_workers=10,
        )
        loaders_validation = LoaderPool(
            ClusteringCuts,
            dict(
                clustering=clustering,
                fragments=fragments,
                cellxgene_batch_size=minibatcher_validation.cellxgene_batch_size,
            ),
            n_workers=5,
        )

        trainer = Trainer(
            self,
            loaders_train,
            loaders_validation,
            minibatcher_train,
            minibatcher_validation,
            SparseDenseAdam(
                self.parameters_sparse(),
                self.parameters_dense(),
                lr=lr,
                weight_decay=1e-5,
            ),
            n_epochs=n_epochs,
            checkpoint_every_epoch=1,
            optimize_every_step=1,
            device=device,
        )
        self.trace = trainer.trace

        trainer.train()

    def _get_likelihood_cell_gene(self, likelihood, local_cellxgene_ix, n_cells, n_genes):
        return torch_scatter.segment_sum_coo(likelihood, local_cellxgene_ix, dim_size=n_cells * n_genes).reshape(
            (n_cells, n_genes)
        )

    def get_prediction(
        self,
        fragments: Fragments,
        clustering: Clustering,
        cells: List[str] = None,
        cell_ixs: List[int] = None,
        genes: List[str] = None,
        gene_ixs: List[int] = None,
        device: str = None,
    ) -> xr.Dataset:
        """
        Returns the likelihoods of the observed cut sites for each cell and gene

        Parameters:
            fragments: Fragments object
            clustering: Clustering object
            cells: Cells to predict
            cell_ixs: Cell indices to predict
            genes: Genes to predict
            gene_ixs: Gene indices to predict
            device: Device to use

        Returns:
            **likelihood_position**, likelihood of the observing a cut site at the particular genomic location, conditioned on the gene region. **likelihood_region**, likelihood of observing a cut site in the gene region
        """

        if cell_ixs is None:
            if cells is None:
                cells = fragments.obs.index
            fragments.obs["ix"] = np.arange(len(fragments.obs))
            cell_ixs = fragments.obs.loc[cells]["ix"].values
        if cells is None:
            cells = fragments.obs.index[cell_ixs]

        if gene_ixs is None:
            if genes is None:
                genes = fragments.var.index
            fragments.var["ix"] = np.arange(len(fragments.var))
            gene_ixs = fragments.var.loc[genes]["ix"].values
        if genes is None:
            genes = fragments.var.index[gene_ixs]

        minibatches = Minibatcher(
            cell_ixs,
            gene_ixs,
            n_genes_step=500,
            n_cells_step=200,
            use_all_cells=True,
            use_all_genes=True,
            permute_cells=False,
            permute_genes=False,
        )
        loaders = LoaderPool(
            ClusteringCuts,
            dict(
                clustering=clustering,
                fragments=fragments,
                cellxgene_batch_size=minibatches.cellxgene_batch_size,
            ),
            n_workers=5,
        )
        loaders.initialize(minibatches)

        likelihood_position = np.zeros((len(cell_ixs), len(gene_ixs)))
        likelihood_region = np.zeros((len(cell_ixs), len(gene_ixs)))

        cell_mapping = np.zeros(fragments.n_cells, dtype=np.int64)
        cell_mapping[cell_ixs] = np.arange(len(cell_ixs))

        gene_mapping = np.zeros(fragments.n_genes, dtype=np.int64)
        gene_mapping[gene_ixs] = np.arange(len(gene_ixs))

        self.eval()
        self = self.to(device)

        for data in loaders:
            data = data.to(device)
            with torch.no_grad():
                self.forward(data)

            likelihood_position[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    gene_mapping[data.minibatch.genes_oi],
                )
            ] += (
                self._get_likelihood_cell_gene(
                    self.track["likelihood_position"],
                    data.cuts.local_cellxgene_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_genes,
                )
                .cpu()
                .numpy()
            )
            likelihood_region[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    gene_mapping[data.minibatch.genes_oi],
                )
            ] += (
                self._get_likelihood_cell_gene(
                    self.track["likelihood_region"],
                    data.cuts.local_cellxgene_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_genes,
                )
                .cpu()
                .numpy()
            )

        self = self.to("cpu")

        result = xr.Dataset(
            {
                "likelihood_position": xr.DataArray(
                    likelihood_position,
                    dims=("cell", "gene"),
                    coords={"cell": cells, "gene": fragments.var.index},
                ),
                "likelihood_region": xr.DataArray(
                    likelihood_region,
                    dims=("cell", "gene"),
                    coords={"cell": cells, "gene": fragments.var.index},
                ),
            }
        )
        return result

    def evaluate_pseudo(
        self,
        coordinates,
        clustering=None,
        gene_oi=None,
        gene_ix=None,
        device=None,
    ):
        from chromatinhd.models.diff.loader.clustering import Result as ClusteringResult
        from chromatinhd.models.diff.loader.clustering_cuts import (
            Result as ClusteringCutsResult,
        )
        from chromatinhd.models.diff.loader.cuts import Result as CutsResult
        from chromatinhd.models.diff.loader.minibatches import Minibatch

        if not torch.is_tensor(clustering):
            if clustering is None:
                clustering = 0.0
            clustering = torch.ones((1, self.n_clusters)) * clustering

            print(clustering)

        cells_oi = torch.ones((1,), dtype=torch.long)

        local_cellxgene_ix = torch.tensor([], dtype=torch.long)
        if gene_ix is None:
            if gene_oi is None:
                gene_oi = 0
            genes_oi = torch.tensor([gene_oi], dtype=torch.long)
            local_gene_ix = torch.zeros_like(coordinates).to(torch.long)
            local_cellxgene_ix = torch.zeros_like(coordinates).to(torch.long)
            localcellxgene_ix = torch.ones_like(coordinates).to(torch.long) * gene_oi
        else:
            assert len(gene_ix) == len(coordinates)
            genes_oi = torch.unique(gene_ix)

            local_gene_mapping = torch.zeros(genes_oi.max() + 1, dtype=torch.long)
            local_gene_mapping.index_add_(0, genes_oi, torch.arange(len(genes_oi)))

            local_gene_ix = local_gene_mapping[gene_ix]
            local_cell_ix = torch.arange(clustering.shape[0])
            local_cellxgene_ix = local_cell_ix * len(genes_oi) + local_gene_ix
            localcellxgene_ix = local_cell_ix * self.n_total_genes + gene_ix

        data = ClusteringCutsResult(
            cuts=CutsResult(
                coordinates=coordinates,
                local_cellxgene_ix=local_cellxgene_ix,
                localcellxgene_ix=localcellxgene_ix,
                n_genes=len(genes_oi),
            ),
            clustering=ClusteringResult(
                onehot=clustering,
            ),
            minibatch=Minibatch(
                cells_oi=cells_oi.cpu().numpy(),
                genes_oi=genes_oi.cpu().numpy(),
            ),
        ).to(device)

        self = self.to(device).eval()

        with torch.no_grad():
            self.forward(data)

        self = self.to("cpu")

        prob = self.track["likelihood"].detach().cpu()
        return prob.detach().cpu()

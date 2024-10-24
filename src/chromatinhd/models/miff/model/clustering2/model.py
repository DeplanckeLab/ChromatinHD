import math
import pickle
from typing import List, Union, Dict

import numpy as np
import pandas as pd
import torch
# import torch_scatter
import tqdm.auto as tqdm
import xarray as xr

from chromatinhd import get_default_device
from chromatinhd.data.clustering import Clustering
from chromatinhd.data.fragments import Fragments
from chromatinhd.data.motifscan import Motifscan

from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.flow import Flow, Stored, Linked
from chromatinhd.loaders import LoaderPool
from chromatinhd.models import HybridModel
from chromatinhd.loaders.minibatches import Minibatcher
from chromatinhd.models.diff.trainer import Trainer
from chromatinhd.optim import SparseDenseAdam
from torch.optim import SGD
from chromatinhd.utils import crossing

from chromatinhd.models.miff.loader.combinations import MotifCountsFragmentsClustering as MainLoader
from . import distributions


class Model(torch.nn.Module, HybridModel, Flow):
    """
    A ChromatinHD-diff model that predicts fragments within cells using motifs
    """

    fragments = Linked()
    motifcounts = Linked()
    clustering = Linked()

    def __init__(
        self,
        fragments,
        motifcounts,
        clustering,
        cls_fragment_position_distribution=distributions.FragmentPositionDistribution1,
        kwargs_fragment_position_distribution={},
    ):
        torch.nn.Module.__init__(self)
        Flow.__init__(self)

        self.fragment_position_distribution = cls_fragment_position_distribution(
            fragments,
            motifcounts,
            clustering,
            **kwargs_fragment_position_distribution,
        )

        self.clustering = clustering
        self.fragments = fragments
        self.motifcounts = motifcounts

    def forward(self, data, return_full_likelihood=False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        self.track = {}

        # p(tile|gene,motifs_tile)
        likelihood_position = self.fragment_position_distribution.log_prob(data)

        if not return_full_likelihood:
            elbo = (-likelihood_position.sum()) / (data.minibatch.n_cells * data.minibatch.n_genes)

            return elbo

        return {
            "likelihood_position": likelihood_position,
            "likelihood": likelihood_position,
        }

    def train_model(self, fold, fragments=None, motifcounts=None, clustering=None, device=None, n_epochs=30, lr=1e-2):
        """
        Trains the model
        """

        fragments = self.fragments if fragments is None else fragments
        motifcounts = self.motifcounts if motifcounts is None else motifcounts
        clustering = self.clustering if clustering is None else clustering

        if device is None:
            device = get_default_device()

        # set up minibatchers and loaders
        minibatcher_train = Minibatcher(
            fold["cells_train"],
            fold["genes_train"],
            n_regions_step=1000,
            n_cells_step=1000,
        )
        minibatcher_validation = Minibatcher(
            fold["cells_validation"],
            fold["genes_validation"],
            n_regions_step=1000,
            n_cells_step=1000,
            permute_cells=False,
            permute_regions=False,
        )

        loaders_train = LoaderPool(
            MainLoader,
            dict(
                motifcounts=motifcounts,
                fragments=fragments,
                clustering=clustering,
                cellxregion_batch_size=minibatcher_train.cellxregion_batch_size,
            ),
            n_workers=10,
        )
        loaders_validation = LoaderPool(
            MainLoader,
            dict(
                motifcounts=motifcounts,
                fragments=fragments,
                clustering=clustering,
                cellxregion_batch_size=minibatcher_validation.cellxregion_batch_size,
            ),
            n_workers=5,
        )

        trainer = Trainer(
            self,
            loaders_train,
            loaders_validation,
            minibatcher_train,
            minibatcher_validation,
            # SGD(
            #     self.parameters(),
            #     lr=lr,
            # ),
            SparseDenseAdam(
                self.parameters_sparse(),
                self.parameters_dense(),
                lr=lr,
                # weight_decay=1e-5,
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
        fragments: Fragments = None,
        motifcounts=None,
        clustering=None,
        cells: List[str] = None,
        cell_ixs: List[int] = None,
        genes: List[str] = None,
        gene_ixs: List[int] = None,
        device: str = None,
    ) -> xr.Dataset:
        fragments = self.fragments if fragments is None else fragments
        motifcounts = self.motifcounts if motifcounts is None else motifcounts
        clustering = self.clustering if clustering is None else clustering

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
            n_regions_step=1000,
            n_cells_step=1000,
            use_all_cells=True,
            use_all_regions=True,
            permute_cells=False,
            permute_regions=False,
        )
        loaders = LoaderPool(
            MainLoader,
            dict(
                motifcounts=motifcounts,
                fragments=fragments,
                clustering=clustering,
                cellxregion_batch_size=minibatches.cellxregion_batch_size,
            ),
            n_workers=5,
        )
        loaders.initialize(minibatches)

        likelihood_position = np.zeros((len(cell_ixs), len(gene_ixs)))
        cell_mapping = np.zeros(fragments.n_cells, dtype=np.int64)
        cell_mapping[cell_ixs] = np.arange(len(cell_ixs))

        gene_mapping = np.zeros(fragments.n_genes, dtype=np.int64)
        gene_mapping[gene_ixs] = np.arange(len(gene_ixs))

        self.eval()
        self = self.to(device)

        for data in loaders:
            data = data.to(device)
            with torch.no_grad():
                full_likelihood = self.forward(data, return_full_likelihood=True)

            likelihood_position[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    gene_mapping[data.minibatch.genes_oi],
                )
            ] += (
                (full_likelihood["likelihood_position"]).cpu().numpy()
            )
            likelihood = likelihood_position

        self = self.to("cpu")

        result = xr.Dataset(
            {
                "likelihood_position": xr.DataArray(
                    likelihood_position,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={fragments.obs.index.name: cells, fragments.var.index.name: genes},
                ),
                "likelihood": xr.DataArray(
                    likelihood,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={fragments.obs.index.name: cells, fragments.var.index.name: genes},
                ),
            }
        )
        return result

    def evaluate_positional(
        self,
        design,
        fragments=None,
        motifcounts=None,
        clustering=None,
        device=None,
    ):
        fragments = self.fragments if fragments is None else fragments
        motifcounts = self.motifcounts if motifcounts is None else motifcounts
        clustering = self.clustering if clustering is None else clustering

        assert "coordinate" in design.columns
        assert "gene_ix" in design.columns

        from chromatinhd.loaders.minibatches import Minibatch
        from chromatinhd.models.miff.loader.combinations import MotifCountsFragmentsClusteringResult
        from chromatinhd.models.miff.loader.binnedmotifcounts import BinnedMotifCounts
        from chromatinhd.models.miff.loader.clustering import Result as ClusteringResult
        from chromatinhd.loaders.fragments import Result as FragmentsResult

        coordinates = torch.from_numpy(design["coordinate"].values).to(torch.long)
        gene_ix = torch.from_numpy(design["gene_ix"].values).to(torch.long)

        cells_oi = torch.arange(coordinates.shape[0])

        local_cellxgene_ix = torch.tensor([], dtype=torch.long)
        assert len(gene_ix) == len(coordinates)
        genes_oi = torch.unique(gene_ix)

        local_gene_mapping = torch.zeros(genes_oi.max() + 1, dtype=torch.long)
        local_gene_mapping.index_add_(0, genes_oi, torch.arange(len(genes_oi)))

        local_gene_ix = local_gene_mapping[gene_ix]
        local_cell_ix = cells_oi
        local_cellxgene_ix = local_cell_ix * len(genes_oi) + local_gene_ix

        window = fragments.regions.window

        genemapping = genes_oi[local_gene_ix]

        coordinates = torch.stack([coordinates, coordinates], -1)

        fragments = FragmentsResult(
            coordinates=coordinates,
            local_cellxgene_ix=local_cellxgene_ix,
            genemapping=genemapping,
            n_fragments=coordinates.shape[0],
            genes_oi=genes_oi,
            cells_oi=cells_oi,
            window=window,
        )

        minibatch = Minibatch(
            cells_oi=cells_oi.cpu().numpy(),
            genes_oi=genes_oi.cpu().numpy(),
        )

        motif_loader = BinnedMotifCounts(motifcounts)
        motifcounts_result = motif_loader.load(minibatch, fragments)

        clustering_result = ClusteringResult(
            labels=torch.from_numpy(design["clustering_label"].values).to(torch.long),
        )

        data = MotifCountsFragmentsClusteringResult(
            fragments=fragments,
            motifcounts=motifcounts_result,
            clustering=clustering_result,
            minibatch=minibatch,
        ).to(device)

        self = self.to(device).eval()

        with torch.no_grad():
            full_likelihood = self.forward(data, return_full_likelihood=True)

        self = self.to("cpu")

        return full_likelihood["likelihood"].detach().cpu(), full_likelihood["likelihood_position"].detach().cpu()

    def plot_positional(self, fragments=None, motifcounts=None, clustering=None, gene_ix=None, gene=None):
        fragments = self.fragments if fragments is None else fragments
        motifcounts = self.motifcounts if motifcounts is None else motifcounts
        clustering = self.clustering if clustering is None else clustering

        import matplotlib.pyplot as plt

        if gene_ix is None:
            if gene is None:
                gene = fragments.var.index[gene_ix]
            gene_ix = fragments.var.index.get_loc(gene)
        else:
            gene = fragments.var.index[gene_ix]
        coordinates = torch.linspace(*fragments.regions.window, 1001)[:-1]

        design = crossing(
            coordinate=coordinates,
            gene_ix=torch.tensor([gene_ix]),
            clustering_label=torch.arange(clustering.n_clusters),
        )

        fig, ax = plt.subplots()

        out = self.evaluate_positional(design, fragments=fragments, motifcounts=motifcounts, clustering=clustering)
        design["logprob"] = out[0].sum(-1).squeeze(-1)
        design["logprob_positional"] = out[1].sum(-1).squeeze(-1)

        # design["logprob"], design["logprob_positional"] = self.evaluate_positional(
        #     design, fragments=fragments, motifcounts=motifcounts, clustering=clustering
        # )
        for cluster_ix, subdesign in design.groupby("clustering_label"):
            ax.plot(
                subdesign["coordinate"],
                np.exp(subdesign["logprob_positional"]),
                label=clustering.cluster_info.index[cluster_ix],
            )
        # ax.plot(design["coordinate"], np.exp(design["logprob"]))
        ax.set_ylim(0)
        ax.legend()
        ax.set_ylabel("probability")
        ax.set_xlabel("coordinate")
        ax.set_title("Positional probability of motif")
        return fig

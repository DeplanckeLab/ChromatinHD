import math
import pickle
from typing import List, Union, Dict

import numpy as np
import pandas as pd
import torch
import torch_scatter
import tqdm.auto as tqdm
import xarray as xr

from chromatinhd import get_default_device
from chromatinhd.data.clustering import Clustering
from chromatinhd.data.fragments import Fragments
from chromatinhd.data.motifscan import Motifscan

from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.flow import Flow, Stored
from chromatinhd.loaders import LoaderPool
from chromatinhd.models import HybridModel
from chromatinhd.models.diff.loader.minibatches import Minibatcher
from chromatinhd.models.diff.trainer import Trainer
from chromatinhd.optim import SparseDenseAdam
from chromatinhd.utils import crossing

from chromatinhd.models.miff.loader.binnedmotifcounts import MotifCountsFragments
from . import distributions


class Model(torch.nn.Module, HybridModel):
    """
    A ChromatinHD-diff model that predicts fragments within cells using motifs
    """

    def __init__(
        self,
        fragments,
        motifcounts,
        cls_fragment_count_distribution=distributions.FragmentCountDistribution1,
        cls_fragment_position_distribution=distributions.FragmentPositionDistribution1,
        kwargs_fragment_position_distribution={},
    ):
        super().__init__()

        self.fragment_count_distribution = cls_fragment_count_distribution(fragments, motifcounts)
        self.fragment_position_distribution = cls_fragment_position_distribution(
            fragments,
            motifcounts,
            **kwargs_fragment_position_distribution,
        )

    def forward(self, data, return_full_likelihood=False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        self.track = {}

        # p(gene|motifs_gene)
        likelihood_count = self.fragment_count_distribution.log_prob(data)

        # p(tile|gene,motifs_tile)
        likelihood_position = self.fragment_position_distribution.log_prob(data)

        if not return_full_likelihood:
            elbo = (-likelihood_position.sum() - likelihood_count.sum()) / (
                data.minibatch.n_cells * data.minibatch.n_genes
            )

            return elbo

        return {
            "likelihood_position": likelihood_position,
            "likelihood_count": likelihood_count,
        }

    def train_model(self, fragments, motifcounts, fold, device=None, n_epochs=30, lr=1e-2):
        """
        Trains the model
        """

        if device is None:
            device = get_default_device()

        # set up minibatchers and loaders
        minibatcher_train = Minibatcher(
            fold["cells_train"],
            fold["genes_train"],
            n_regions_step=500,
            n_cells_step=200,
        )
        minibatcher_validation = Minibatcher(
            fold["cells_validation"],
            fold["genes_validation"],
            n_regions_step=1000,
            n_cells_step=1000,
            permute_cells=False,
            permute_genes=False,
        )

        loaders_train = LoaderPool(
            MotifCountsFragments,
            dict(
                motifcounts=motifcounts,
                fragments=fragments,
                cellxgene_batch_size=minibatcher_train.cellxgene_batch_size,
            ),
            n_workers=10,
        )
        loaders_validation = LoaderPool(
            MotifCountsFragments,
            dict(
                motifcounts=motifcounts,
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
            torch.optim.Adam(self.parameters(), lr=lr),
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
        motifcounts,
        cells: List[str] = None,
        cell_ixs: List[int] = None,
        genes: List[str] = None,
        gene_ixs: List[int] = None,
        device: str = None,
    ) -> xr.Dataset:
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
            n_regions_step=500,
            n_cells_step=200,
            use_all_cells=True,
            use_all_genes=True,
            permute_cells=False,
            permute_genes=False,
        )
        loaders = LoaderPool(
            MotifCountsFragments,
            dict(
                motifcounts=motifcounts,
                fragments=fragments,
                cellxgene_batch_size=minibatches.cellxgene_batch_size,
            ),
            n_workers=5,
        )
        loaders.initialize(minibatches)

        likelihood_position = np.zeros((len(cell_ixs), len(gene_ixs)))
        likelihood_count = np.zeros((len(cell_ixs), len(gene_ixs)))

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
                self._get_likelihood_cell_gene(
                    full_likelihood["likelihood_position"],
                    data.fragments.local_cellxgene_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_genes,
                )
                .cpu()
                .numpy()
            )
            likelihood_count[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    gene_mapping[data.minibatch.genes_oi],
                )
            ] += (
                full_likelihood["likelihood_count"].cpu().numpy()
            )

        self = self.to("cpu")

        result = xr.Dataset(
            {
                "likelihood_position": xr.DataArray(
                    likelihood_position,
                    dims=("cell", "gene"),
                    coords={"cell": cells, "gene": genes},
                ),
                "likelihood_count": xr.DataArray(
                    likelihood_count,
                    dims=("cell", "gene"),
                    coords={"cell": cells, "gene": genes},
                ),
                "likelihood": xr.DataArray(
                    likelihood_position,
                    dims=("cell", "gene"),
                    coords={"cell": cells, "gene": genes},
                ),
            }
        )
        return result

    def evaluate_positional(
        self,
        design,
        fragments,
        motifcounts,
        device=None,
    ):
        assert "coordinate" in design.columns
        assert "gene_ix" in design.columns

        from chromatinhd.models.diff.loader.minibatches import Minibatch
        from chromatinhd.models.miff.loader.binnedmotifcounts import MotifCountsFragmentsResult
        from chromatinhd.models.miff.loader.binnedmotifcounts import BinnedMotifCounts
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

        data = MotifCountsFragmentsResult(
            fragments=fragments,
            motifcounts=motifcounts_result,
            minibatch=minibatch,
        ).to(device)

        self = self.to(device).eval()

        with torch.no_grad():
            full_likelihood = self.forward(data, return_full_likelihood=True)

        self = self.to("cpu")

        prob = full_likelihood["likelihood_position"].detach().cpu()
        return prob.detach().cpu()

    def plot_positional(self, fragments, motifcounts, gene_ix=0, gene=None):
        import matplotlib.pyplot as plt

        if gene is None:
            gene = fragments.var.index[gene_ix]
        coordinates = torch.linspace(*fragments.regions.window, 1001)[:-1]

        design = crossing(
            coordinate=coordinates,
            gene_ix=torch.tensor([gene_ix]),
        )

        fig, ax = plt.subplots()

        design["logprob"] = self.evaluate_positional(design, fragments=fragments, motifcounts=motifcounts)
        print(
            np.trapz(
                np.exp(design["logprob"]),
                np.linspace(0, 1, len(design)),
            )
        )
        ax.plot(design["coordinate"], np.exp(design["logprob"]))
        ax.set_ylim(0)
        ax.set_ylabel("probability")
        ax.set_xlabel("coordinate")
        ax.set_title("Positional probability of motif")

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

from chromatinhd.models.miff.loader.combinations import FragmentsClustering as MainLoader
from .count import FragmentCountDistribution1
from .count import Baseline as CountBaseline
from .position import FragmentPositionDistribution1
from .position import Baseline as PositionBaseline


class Model(torch.nn.Module, HybridModel, FlowModel):
    """
    A ChromatinHD-diff model that identifies differential accessibility between clusters
    """

    fragments = Linked()
    clustering = Linked()

    def __init__(
        self,
        fragments,
        clustering,
        fragment_position_distribution=None,
        fragment_count_distribution=None,
        path=None,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        Flow.__init__(self, path=path, **kwargs)

        if fragment_count_distribution is not None:
            self.fragment_count_distribution = fragment_count_distribution
        else:
            self.fragment_count_distribution = FragmentCountDistribution1(fragments, clustering)

        if fragment_position_distribution is not None:
            self.fragment_position_distribution = fragment_position_distribution
        else:
            self.fragment_position_distribution = FragmentPositionDistribution1(fragments, clustering)

        self.clustering = clustering
        self.fragments = fragments

    def forward(self, data, return_full_likelihood=False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        self.track = {}

        # p(region|motifs_region)
        likelihood_count = self.fragment_count_distribution.log_prob(data)
        # print(likelihood_count.sum())

        # p(tile|region,motifs_tile)
        likelihood_position = self.fragment_position_distribution.log_prob(data)

        if not return_full_likelihood:
            elbo = (-likelihood_position.sum() - likelihood_count.sum()) / (
                data.minibatch.n_cells * data.minibatch.n_regions
            )

            return elbo

        return {
            "likelihood_position": likelihood_position,
            "likelihood_count": likelihood_count,
            "likelihood": likelihood_position.sum(-1) + likelihood_count.flatten()[data.fragments.local_cellxregion_ix],
        }

    def train_model(self, fold, fragments=None, clustering=None, device=None, n_epochs=30, lr=1e-2, n_cells_step=1000):
        """
        Trains the model
        """

        import gc

        gc.collect()
        torch.cuda.empty_cache()

        fragments = self.fragments if fragments is None else fragments
        clustering = self.clustering if clustering is None else clustering

        if device is None:
            device = get_default_device()

        # set up minibatchers and loaders
        minibatcher_train = Minibatcher(
            fold["cells_train"],
            fold["regions_train"],
            n_regions_step=1000,
            n_cells_step=n_cells_step,
        )
        minibatcher_validation = Minibatcher(
            fold["cells_validation"],
            fold["regions_validation"],
            n_regions_step=1000,
            n_cells_step=n_cells_step,
            permute_cells=False,
            permute_regions=False,
        )

        loaders_train = LoaderPool(
            MainLoader,
            dict(
                fragments=fragments,
                clustering=clustering,
                cellxregion_batch_size=minibatcher_train.cellxregion_batch_size,
            ),
            n_workers=20,
        )
        loaders_validation = LoaderPool(
            MainLoader,
            dict(
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

    def _get_likelihood_cell_region(self, likelihood, local_cellxregion_ix, n_cells, n_regions):
        return torch_scatter.segment_sum_coo(likelihood, local_cellxregion_ix, dim_size=n_cells * n_regions).reshape(
            (n_cells, n_regions)
        )

    def get_prediction(
        self,
        fragments: Fragments = None,
        clustering=None,
        cells: List[str] = None,
        cell_ixs: List[int] = None,
        regions: List[str] = None,
        region_ixs: List[int] = None,
        device: str = None,
    ) -> xr.Dataset:
        fragments = self.fragments if fragments is None else fragments
        clustering = self.clustering if clustering is None else clustering

        if cell_ixs is None:
            if cells is None:
                cells = fragments.obs.index
            fragments.obs["ix"] = np.arange(len(fragments.obs))
            cell_ixs = fragments.obs.loc[cells]["ix"].values
        if cells is None:
            cells = fragments.obs.index[cell_ixs]

        if region_ixs is None:
            if regions is None:
                regions = fragments.var.index
            fragments.var["ix"] = np.arange(len(fragments.var))
            region_ixs = fragments.var.loc[regions]["ix"].values
        if regions is None:
            regions = fragments.var.index[region_ixs]

        minibatches = Minibatcher(
            cell_ixs,
            region_ixs,
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
                fragments=fragments,
                clustering=clustering,
                cellxregion_batch_size=minibatches.cellxregion_batch_size,
            ),
            n_workers=5,
        )
        loaders.initialize(minibatches)

        likelihood_position = np.zeros((len(cell_ixs), len(region_ixs)))
        likelihood_count = np.zeros((len(cell_ixs), len(region_ixs)))

        cell_mapping = np.zeros(fragments.n_cells, dtype=np.int64)
        cell_mapping[cell_ixs] = np.arange(len(cell_ixs))

        region_mapping = np.zeros(fragments.n_regions, dtype=np.int64)
        region_mapping[region_ixs] = np.arange(len(region_ixs))

        self.eval()
        self = self.to(device)

        progress = tqdm.tqdm(loaders, total=len(loaders), leave=False)

        for data in progress:
            data = data.to(device)
            with torch.no_grad():
                full_likelihood = self.forward(data, return_full_likelihood=True)

            likelihood_position[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    region_mapping[data.minibatch.regions_oi],
                )
            ] += (
                self._get_likelihood_cell_region(
                    full_likelihood["likelihood_position"].sum(-1),
                    data.fragments.local_cellxregion_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_regions,
                )
                .cpu()
                .numpy()
            )
            likelihood_count[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    region_mapping[data.minibatch.regions_oi],
                )
            ] += (
                full_likelihood["likelihood_count"].cpu().numpy()
            )
            likelihood = likelihood_position + likelihood_count

        self = self.to("cpu")

        result = xr.Dataset(
            {
                "likelihood_position": xr.DataArray(
                    likelihood_position,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={fragments.obs.index.name: cells, fragments.var.index.name: regions},
                ),
                "likelihood_count": xr.DataArray(
                    likelihood_count,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={fragments.obs.index.name: cells, fragments.var.index.name: regions},
                ),
                "likelihood": xr.DataArray(
                    likelihood,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={fragments.obs.index.name: cells, fragments.var.index.name: regions},
                ),
            }
        )
        return result

    def evaluate_positional(
        self,
        design,
        fragments=None,
        clustering=None,
        device=None,
    ):
        fragments = self.fragments if fragments is None else fragments
        clustering = self.clustering if clustering is None else clustering

        assert "coordinate" in design.columns
        assert "region_ix" in design.columns

        from chromatinhd.loaders.minibatches import Minibatch
        from chromatinhd.models.miff.loader.combinations import FragmentsClusteringResult
        from chromatinhd.models.miff.loader.clustering import Result as ClusteringResult
        from chromatinhd.loaders.fragments import FragmentsResult

        coordinates = torch.from_numpy(design["coordinate"].values).to(torch.long)
        region_ix = torch.from_numpy(design["region_ix"].values).to(torch.long)

        cells_oi = torch.arange(coordinates.shape[0])

        local_cellxregion_ix = torch.tensor([], dtype=torch.long)
        assert len(region_ix) == len(coordinates)
        regions_oi = torch.unique(region_ix)

        local_region_mapping = torch.zeros(regions_oi.max() + 1, dtype=torch.long)
        local_region_mapping.index_add_(0, regions_oi, torch.arange(len(regions_oi)))

        local_region_ix = local_region_mapping[region_ix]
        local_cell_ix = cells_oi
        local_cellxregion_ix = local_cell_ix * len(regions_oi) + local_region_ix

        window = fragments.regions.window

        regionmapping = regions_oi[local_region_ix]

        coordinates = torch.stack([coordinates, coordinates], -1)

        lib = torch.ones((len(cells_oi),), dtype=torch.float)

        fragments = FragmentsResult(
            coordinates=coordinates,
            local_cellxregion_ix=local_cellxregion_ix,
            regionmapping=regionmapping,
            n_fragments=coordinates.shape[0],
            regions_oi=regions_oi,
            cells_oi=cells_oi,
            window=window,
            lib=lib,
        )

        minibatch = Minibatch(
            cells_oi=cells_oi.cpu().numpy(),
            regions_oi=regions_oi.cpu().numpy(),
        )

        clustering_result = ClusteringResult(
            labels=torch.from_numpy(design["clustering_label"].values).to(torch.long),
        )

        data = FragmentsClusteringResult(
            fragments=fragments,
            clustering=clustering_result,
            minibatch=minibatch,
        ).to(device)

        self = self.to(device).eval()

        with torch.no_grad():
            full_likelihood = self.forward(data, return_full_likelihood=True)

        self = self.to("cpu")

        return (
            full_likelihood["likelihood"].detach().cpu(),
            full_likelihood["likelihood_position"][:, 0].detach().cpu(),
        )

    def plot_positional(self, region=None, region_ix=None, fragments=None, clustering=None):
        fragments = self.fragments if fragments is None else fragments
        clustering = self.clustering if clustering is None else clustering

        import matplotlib.pyplot as plt

        if region_ix is None:
            if region is None:
                region = fragments.var.index[region_ix]
            region_ix = fragments.var.index.get_loc(region)
        else:
            region = fragments.var.index[region_ix]
        coordinates = torch.linspace(*fragments.regions.window, 2001)[:-1]

        design = crossing(
            coordinate=coordinates,
            region_ix=torch.tensor([region_ix]),
            clustering_label=torch.arange(clustering.n_clusters),
        )

        fig, ax = plt.subplots()

        design["logprob"], design["logprob_positional"] = self.evaluate_positional(
            design, fragments=fragments, clustering=clustering
        )
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

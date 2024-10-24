import math
import pickle
from typing import List

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
from chromatinhd.flow import Flow, Stored
from chromatinhd.loaders import LoaderPool
from chromatinhd.models import HybridModel
from chromatinhd.loaders.minibatches import Minibatcher
from chromatinhd.models.diff.trainer import Trainer
from chromatinhd.optim import SparseDenseAdam
from chromatinhd.utils import crossing

from . import distributions
from chromatinhd.models.miff.loader.motifs_fragments import MotifsFragments


class Model(torch.nn.Module, HybridModel):
    """
    A ChromatinHD-diff model that predicts fragments within cells using motifs
    """

    def __init__(
        self,
        fragments,
        motifscan,
        cls_fragment_count_distribution=distributions.FragmentCountDistribution2,
        cls_fragment_position_distribution=distributions.FragmentPositionDistribution2,
        kwargs_fragment_count_distribution={},
    ):
        super().__init__()

        self.fragment_count_distribution = cls_fragment_count_distribution(fragments, motifscan)
        self.fragment_position_distribution = cls_fragment_position_distribution(
            fragments, motifscan, **kwargs_fragment_count_distribution
        )

    def forward(self, data, return_full_likelihood=False):
        self.track = {}

        # p(region|motifs_region)
        likelihood_count = self.fragment_count_distribution.log_prob(data)

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
        }

    def train_model(self, fragments, motifscan, fold, device=None, n_epochs=30, lr=1e-2):
        """
        Trains the model
        """

        if device is None:
            device = get_default_device()

        # set up minibatchers and loaders
        minibatcher_train = Minibatcher(
            fold["cells_train"],
            fold["regions_train"],
            n_regions_step=500,
            n_cells_step=200,
        )
        minibatcher_validation = Minibatcher(
            fold["cells_validation"],
            fold["regions_validation"],
            n_regions_step=1000,
            n_cells_step=1000,
            permute_cells=False,
            permute_regions=False,
        )

        loaders_train = LoaderPool(
            MotifsFragments,
            dict(
                motifscan=motifscan,
                fragments=fragments,
                cellxregion_batch_size=minibatcher_train.cellxregion_batch_size,
            ),
            n_workers=10,
        )
        loaders_validation = LoaderPool(
            MotifsFragments,
            dict(
                motifscan=motifscan,
                fragments=fragments,
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

    def _get_likelihood_cell_region(self, likelihood, local_cellxregion_ix, n_cells, n_regions):
        return torch_scatter.segment_sum_coo(likelihood, local_cellxregion_ix, dim_size=n_cells * n_regions).reshape(
            (n_cells, n_regions)
        )

    def get_prediction(
        self,
        fragments: Fragments,
        motifscan: Motifscan,
        cells: List[str] = None,
        cell_ixs: List[int] = None,
        regions: List[str] = None,
        region_ixs: List[int] = None,
        device: str = None,
    ) -> xr.Dataset:
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
            n_regions_step=500,
            n_cells_step=200,
            use_all_cells=True,
            use_all_regions=True,
            permute_cells=False,
            permute_regions=False,
        )
        loaders = LoaderPool(
            MotifsFragments,
            dict(
                motifscan=motifscan,
                fragments=fragments,
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

        for data in loaders:
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
                    full_likelihood["likelihood_position"],
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
                    likelihood_count + likelihood_position,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={fragments.obs.index.name: cells, fragments.var.index.name: regions},
                ),
            }
        )
        return result

    def evaluate_positional(
        self,
        design,
        motifscan,
        device=None,
    ):
        assert "coordinate" in design.columns
        assert "region_ix" in design.columns

        from chromatinhd.loaders.minibatches import Minibatch
        from chromatinhd.models.miff.loader.motifs import Motifs
        from chromatinhd.models.miff.loader.motifs_fragments import Result as MotifsFragmentsResult
        from chromatinhd.loaders.fragments import Result as FragmentsResult

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

        regionmapping = regions_oi[local_region_ix]

        coordinates = torch.stack([coordinates, coordinates], -1)

        window = motifscan.regions.window

        assert (coordinates[:, 1] <= window[1]).all()
        assert (coordinates[:, 0] >= window[0]).all()

        fragments = FragmentsResult(
            coordinates=coordinates,
            local_cellxregion_ix=local_cellxregion_ix,
            regionmapping=regionmapping,
            n_fragments=coordinates.shape[0],
            regions_oi=regions_oi,
            cells_oi=cells_oi,
        )

        minibatch = Minibatch(
            cells_oi=cells_oi.cpu().numpy(),
            regions_oi=regions_oi.cpu().numpy(),
        )

        motif_loader = Motifs(motifscan)
        motifs = motif_loader.load(minibatch)

        data = MotifsFragmentsResult(
            fragments=fragments,
            motifs=motifs,
            minibatch=minibatch,
        ).to(device)

        self = self.to(device).eval()

        with torch.no_grad():
            full_likelihood = self.forward(data, return_full_likelihood=True)

        self = self.to("cpu")

        prob = full_likelihood["likelihood_position"].detach().cpu()
        return prob.detach().cpu()

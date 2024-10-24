import math
import pickle
from typing import List

import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm
import xarray as xr

from chromatinhd import get_default_device
from chromatinhd.data.clustering import Clustering
from chromatinhd.data.fragments import Fragments
from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.flow import Flow, Stored, Linked, LinkedDict
from chromatinhd.loaders import LoaderPool
from chromatinhd.models import HybridModel
from chromatinhd.models.diff.loader.clustering_cuts import ClusteringCuts
from chromatinhd.loaders.minibatches import Minibatcher
from chromatinhd.models.diff.trainer import Trainer, TrainerPerFeature
from chromatinhd.optim import SparseDenseAdam
from chromatinhd.utils import crossing
from chromatinhd.models import FlowModel

from .encoders import Shared, Split, SharedLora


class Model(FlowModel):
    """
    A ChromatinHD-diff model that models the probability density of observing a cut site between clusterings
    """

    fragments = Linked()
    clustering = Linked()
    fold = Stored()

    @classmethod
    def create(
        cls,
        fragments: Fragments,
        clustering: Clustering,
        fold,
        path=None,
        baseline=False,
        overall_delta_regularization=True,
        overall_delta_p_scale_free=False,
        encoder="shared",
        encoder_params=dict(),
        overwrite=True,
    ):
        """
        Parameters:
            fragments:
                Fragments object
            clustering:
                Clustering object
            nbins:
                Number of bins for the spline
            decoder_n_layers:
                Number of layers in the decoder
            baseline:
                Whether to use a baseline model
        """

        self = super(Model, cls).create(path=path, fragments=fragments, clustering=clustering, reset=overwrite)

        self.n_total_regions = fragments.n_regions

        self.n_clusters = clustering.n_clusters

        libsize = torch.from_numpy(fragments.counts.sum(1)).float()
        libsize = libsize + 1  # pseudocount
        self.register_buffer("libsize", libsize)

        window = fragments.regions.window

        # create overall
        overall_bias = fragments.counts.sum(0) / fragments.n_cells / libsize.to(torch.float).mean()
        min_overall_bias = 1e-5
        overall_bias = torch.log(min_overall_bias + (1 - min_overall_bias) * overall_bias)
        self.register_buffer("overall_bias", overall_bias)

        # create overall delta
        self.overall_delta_regularization = overall_delta_regularization
        if self.overall_delta_regularization:
            if overall_delta_p_scale_free:
                self.overall_delta_p_scale = torch.nn.Parameter(torch.log(torch.tensor(0.1, requires_grad=True)))
            else:
                self.register_buffer("overall_delta_p_scale", torch.tensor(math.log(1.0)))
        n_clusters = clustering.n_clusters

        self.overall_delta = EmbeddingTensor(fragments.n_regions, (n_clusters,), sparse=True)
        # self.overall_delta = torch.nn.Parameter(torch.zeros((fragments.n_regions, n_clusters)))
        self.overall_delta.data[:] = 0.0

        if encoder == "shared":
            self.encoder = Shared(fragments, clustering, **encoder_params)
        elif encoder == "shared_lowrank":
            self.encoder = SharedLora(fragments, clustering, **encoder_params)
        elif encoder == "split":
            self.encoder = Split(fragments, clustering, **encoder_params)

        self.window = window

        self.fold = fold

        return self

    def parameters_sparse(self):
        return [*self.encoder.parameters_sparse()] + [self.overall_delta.weight]

    def forward(self, data, w_delta_multiplier=1.0, libsize=None):
        self.track = {}
        elbo = torch.tensor(0.0, device=data.cuts.coordinates.device)

        count = torch.bincount(
            data.cuts.local_cellxregion_ix, minlength=data.minibatch.n_cells * data.minibatch.n_regions
        ).reshape((data.minibatch.n_cells, data.minibatch.n_regions))

        if libsize is None:
            libsize = self.libsize
        mu = torch.exp(
            (
                self.overall_bias[data.minibatch.genes_oi].unsqueeze(1)
                + self.overall_delta(data.minibatch.regions_oi_torch)
            )[:, data.clustering.indices].T
        ) * libsize[data.minibatch.cells_oi].unsqueeze(1)

        likelihood_overall = torch.distributions.Poisson(mu).log_prob(count)
        self.track["likelihood_overall"] = likelihood_overall
        self.track["logprob_overall"] = torch.log(mu)

        elbo = elbo - likelihood_overall.sum()

        # bin
        likelihood_bin, kl_bin = self.encoder.forward(data, w_delta_multiplier=w_delta_multiplier)
        likelihood_mixture = self._get_likelihood_cell_region(
            likelihood_bin, data.cuts.local_cellxregion_ix, data.minibatch.n_cells, data.minibatch.n_regions
        )

        elbo = elbo - likelihood_mixture.sum() - kl_bin.sum()

        self.track["likelihood_mixture"] = likelihood_mixture
        self.track["kl_bin"] = kl_bin

        return elbo / data.cuts.n_cuts

    def forward_region_loss(self, data):
        self.forward(data)

        likelihood_overall = self.track["likelihood_overall"].sum(0)
        likelihood_mixture = self.track["likelihood_mixture"].sum(0)
        return -(likelihood_overall + likelihood_mixture)

    def forward_likelihood(self, data):
        _ = self.forward(data)
        return self.track["likelihood_mixture"] + self.track["likelihood_overall"]

    def train_model(
        self,
        device=None,
        n_epochs=30,
        lr=1e-2,
        pbar=True,
        early_stopping=True,
        fold=None,
        fragments: Fragments = None,
        clustering=None,
        do_validation=True,
        n_cells_step=250,
        n_regions_step=100,
        n_workers_validation=5,
        n_workers_train=10,
        regions_oi=None,
    ):
        """
        Trains the model
        """

        if fragments is None:
            fragments = self.fragments
        if clustering is None:
            clustering = self.clustering
        if fold is None:
            fold = self.fold

        if device is None:
            device = get_default_device()

        if regions_oi is not None:
            region_ixs = fragments.var.index.get_indexer(regions_oi)
        else:
            region_ixs = range(fragments.n_regions)

        # set up minibatchers and loaders
        minibatcher_train = Minibatcher(
            fold["cells_train"],
            region_ixs,
            n_regions_step=n_regions_step,
            n_cells_step=n_cells_step,
        )
        minibatcher_validation = Minibatcher(
            fold["cells_validation"],
            region_ixs,
            n_regions_step=n_regions_step,
            n_cells_step=n_cells_step,
            permute_cells=False,
            permute_regions=False,
            use_all_cells=True,
        )

        loaders_train = LoaderPool(
            ClusteringCuts,
            dict(
                clustering=clustering,
                fragments=fragments,
                cellxregion_batch_size=minibatcher_train.cellxregion_batch_size,
            ),
            n_workers=n_workers_train,
        )
        loaders_validation = LoaderPool(
            ClusteringCuts,
            dict(
                clustering=clustering,
                fragments=fragments,
                cellxregion_batch_size=minibatcher_validation.cellxregion_batch_size,
            ),
            n_workers=n_workers_validation,
        )

        trainer = Trainer(
            # trainer = TrainerPerFeature(
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
            early_stopping=early_stopping,
            do_validation=do_validation,
        )
        self.trace = trainer.trace

        trainer.train()

    def _get_likelihood_cell_region(self, likelihood, local_cellxregion_ix, n_cells, n_regions):
        likelihood_cell_region = torch.zeros(n_cells * n_regions, device=likelihood.device)
        likelihood_cell_region.scatter_add_(0, local_cellxregion_ix, likelihood)
        return likelihood_cell_region.reshape((n_cells, n_regions))
        # return torch_scatter.segment_sum_coo(likelihood, local_cellxregion_ix, dim_size=n_cells * n_regions).reshape(
        #     (n_cells, n_regions)
        # )

    def get_prediction(
        self,
        cells: List[str] = None,
        cell_ixs: List[int] = None,
        regions: List[str] = None,
        region_ixs: List[int] = None,
        device: str = None,
        fragments: Fragments = None,
        clustering: Clustering = None,
        return_raw=False,
    ) -> xr.Dataset:
        """
        Returns the likelihoods of the observed cut sites for each cell and region

        Parameters:
            cells: Cells to predict
            cell_ixs: Cell indices to predict
            regions: Genes to predict
            region_ixs: Gene indices to predict
            device: Device to use

        Returns:
            **likelihood_mixture**, likelihood of the observing a cut site at the particular genomic location, conditioned on the region region. **likelihood_overall**, likelihood of observing a cut site in the region region
        """

        if fragments is None:
            fragments = self.fragments
        if clustering is None:
            clustering = self.clustering

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
            ClusteringCuts,
            dict(
                clustering=clustering,
                fragments=fragments,
                cellxregion_batch_size=minibatches.cellxregion_batch_size,
            ),
            n_workers=5,
        )
        loaders.initialize(minibatches)

        likelihood_mixture = np.zeros((len(cell_ixs), len(region_ixs)))
        likelihood_overall = np.zeros((len(cell_ixs), len(region_ixs)))

        cell_mapping = np.zeros(fragments.n_cells, dtype=np.int64)
        cell_mapping[cell_ixs] = np.arange(len(cell_ixs))

        region_mapping = np.zeros(fragments.n_regions, dtype=np.int64)
        region_mapping[region_ixs] = np.arange(len(region_ixs))

        if device is None:
            device = get_default_device()

        self.eval()
        self = self.to(device)

        for data in loaders:
            data = data.to(device)
            with torch.no_grad():
                self.forward(data)

            likelihood_mixture[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    region_mapping[data.minibatch.regions_oi],
                )
            ] += (
                self.track["likelihood_mixture"].detach().cpu().numpy()
            )
            likelihood_overall[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    region_mapping[data.minibatch.regions_oi],
                )
            ] += (
                self.track["likelihood_overall"].detach().cpu().numpy()
            )
        likelihood = likelihood_mixture + likelihood_overall

        self = self.to("cpu")

        if return_raw:
            return likelihood

        result = xr.Dataset(
            {
                "likelihood_mixture": xr.DataArray(
                    likelihood_mixture,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={fragments.obs.index.name: cells, fragments.var.index.name: regions},
                ),
                "likelihood_overall": xr.DataArray(
                    likelihood_overall,
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

    def evaluate_pseudo(
        self,
        coordinates,
        clustering=None,
        region_oi=None,
        region_ix=None,
        device=None,
        normalize_per_bp=100,
        normalize_per_cell=100,
    ):
        from chromatinhd.loaders.clustering import Result as ClusteringResult
        from chromatinhd.models.diff.loader.clustering_cuts import (
            Result as ClusteringCutsResult,
        )
        from chromatinhd.loaders.fragments import CutsResult
        from chromatinhd.loaders.minibatches import Minibatch

        if not torch.is_tensor(clustering):
            if clustering is None:
                clustering = 0.0
            clustering = clustering
        clustering_indices = torch.argmax(clustering, dim=1).long()

        cells_oi = torch.arange(len(coordinates), dtype=torch.long)

        local_cellxregion_ix = torch.tensor([], dtype=torch.long)
        if region_ix is None:
            if region_oi is None:
                region_oi = 0
            regions_oi = torch.tensor([region_oi], dtype=torch.long)
            local_region_ix = torch.zeros_like(coordinates).to(torch.long)
            local_cellxregion_ix = torch.zeros_like(coordinates).to(torch.long)
        else:
            assert len(region_ix) == len(coordinates)
            regions_oi = torch.unique(region_ix)

            local_region_mapping = torch.zeros(regions_oi.max() + 1, dtype=torch.long)
            local_region_mapping.index_add_(0, regions_oi, torch.arange(len(regions_oi)))

            local_region_ix = local_region_mapping[region_ix]
            local_cell_ix = torch.arange(clustering.shape[0])
            local_cellxregion_ix = local_cell_ix * len(regions_oi) + local_region_ix

        data = ClusteringCutsResult(
            cuts=CutsResult(
                coordinates=coordinates,
                local_cellxregion_ix=local_cellxregion_ix,
                n_regions=len(regions_oi),
                n_fragments=len(coordinates),
                n_cuts=len(coordinates),
                window=torch.tensor([0, 1]),
            ),
            clustering=ClusteringResult(
                indices=clustering_indices,
            ),
            minibatch=Minibatch(
                cells_oi=cells_oi.cpu().numpy(),
                regions_oi=regions_oi.cpu().numpy(),
            ),
        ).to(device)

        self = self.to(device).eval()

        with torch.no_grad():
            self.forward(data, libsize=self.libsize.float().mean() * torch.ones(len(coordinates), device=device))

        self = self.to("cpu")

        prob = self.track["likelihood_mixture"].detach().cpu().sum(1)
        prob = prob + self.track["logprob_overall"].detach().cpu().sum(1)
        prob = prob + math.log(normalize_per_cell) + math.log(normalize_per_bp)
        return prob.detach().cpu()


class Models(Flow):
    models = LinkedDict()

    clustering = Linked()
    """The clustering"""

    fragments = Linked()
    """The fragments"""

    folds = Linked()
    """The folds"""

    model_params = Stored(default=dict)
    train_params = Stored(default=dict)

    @property
    def models_path(self):
        path = self.path / "models"
        path.mkdir(exist_ok=True)
        return path

    def train_models(
        self, fragments=None, clustering=None, folds=None, device=None, pbar=True, regions_oi=None, **kwargs
    ):
        if fragments is None:
            fragments = self.fragments
        if clustering is None:
            clustering = self.clustering
        if folds is None:
            folds = self.folds

        progress = tqdm.tqdm(enumerate(folds), total=len(folds)) if pbar else enumerate(folds)

        for fold_ix, fold in progress:
            model_name = f"{fold_ix}"
            model_folder = self.models_path / (model_name)
            force = False
            if model_name not in self.models:
                force = True
            elif not self.models[model_name].o.state.exists(self.models[model_name]):
                force = True

            if force:
                model = Model.create(
                    fragments=fragments,
                    clustering=clustering,
                    fold=fold,
                    path=model_folder,
                    **self.model_params,
                )
                model.train_model(device=device, pbar=True, regions_oi=regions_oi, **{**self.train_params, **kwargs})
                model.save_state()

                model = model.to("cpu")

                self.models[model_name] = model

    def __contains__(self, ix):
        return ix in self.models

    def __getitem__(self, ix):
        return self.models[ix]

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        for ix in self.models.keys():
            yield self[ix]

    @property
    def design(self):
        design_dimensions = {
            "fold": range(len(self.folds)),
        }
        design = crossing(**design_dimensions)
        design.index = design["fold"].astype(str)
        return design

    def fitted(self, fold_ix):
        return f"{fold_ix}" in self.models

    def get_prediction(self, fold_ix, **kwargs):
        model = self[f"{fold_ix}"]
        return model.get_prediction(**kwargs)

    @property
    def trained(self):
        print(len(self))
        return len(self) > 0

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
from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.flow import Flow, Stored
from chromatinhd.loaders import LoaderPool
from chromatinhd.models import HybridModel
from chromatinhd.models.diff.loader.clustering_cuts import ClusteringCuts
from chromatinhd.loaders.minibatches import Minibatcher
from chromatinhd.models.diff.trainer import Trainer
from chromatinhd.optim import SparseDenseAdam
from chromatinhd.utils import crossing

from .spline import DifferentialQuadraticSplineStack, TransformedDistribution


class Decoder(torch.nn.Module):
    def __init__(
        self,
        n_latent,
        n_regions,
        n_output_components,
        n_layers=1,
        n_hidden_dimensions=32,
        dropout_rate=0.2,
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(torch.nn.Linear(n_hidden_dimensions if i > 0 else n_latent, n_hidden_dimensions))
            layers.append(torch.nn.BatchNorm1d(n_hidden_dimensions))
            layers.append(torch.nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(torch.nn.Dropout(0.2))
        self.nn = torch.nn.Sequential(*layers)

        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_output_components = n_output_components

        self.logit_weight = EmbeddingTensor(
            n_regions,
            (n_hidden_dimensions if n_layers > 0 else n_latent, n_output_components),
            sparse=True,
        )
        stdv = 1.0 / math.sqrt(self.logit_weight.weight.size(1))
        # self.logit_weight.weight.data.uniform_(-stdv, stdv)
        self.logit_weight.weight.data.zero_()

        self.rho_weight = EmbeddingTensor(n_regions, (n_hidden_dimensions if n_layers > 0 else n_latent,), sparse=True)
        stdv = 1.0 / math.sqrt(self.rho_weight.weight.size(1))
        self.rho_weight.weight.data.uniform_(-stdv, stdv)

    def forward(self, latent, regions_oi):
        # regions oi is only used to get the logits
        # we calculate the rho for all regions because we pool using softmax later
        logit_weight = self.logit_weight(regions_oi)
        rho_weight = self.rho_weight.get_full_weight()
        nn_output = self.nn(latent)

        # nn_output is broadcasted across regions and across components
        logit = torch.matmul(nn_output.unsqueeze(1).unsqueeze(2), logit_weight).squeeze(-2)

        # nn_output has to be broadcasted across regions
        rho = torch.matmul(nn_output.unsqueeze(1), rho_weight.T).squeeze(-2)

        return logit, rho

    def parameters_dense(self):
        return [*self.nn.parameters(), self.rho_weight.weight]

    def parameters_sparse(self):
        return [self.logit_weight.weight]


class BaselineDecoder(torch.nn.Module):
    def __init__(self, n_latent, n_regions, n_output_components, n_layers=1, n_hidden_dimensions=32):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(torch.nn.Linear(n_hidden_dimensions if i > 0 else n_latent, n_hidden_dimensions))
            layers.append(torch.nn.BatchNorm1d(n_hidden_dimensions))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.2))
        self.nn = torch.nn.Sequential(*layers)

        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_output_components = n_output_components

        self.rho_weight = EmbeddingTensor(n_regions, (n_hidden_dimensions if n_layers > 0 else n_latent,), sparse=True)
        stdv = 1.0 / math.sqrt(self.rho_weight.weight.size(1)) / 100
        self.rho_weight.weight.data.uniform_(-stdv, stdv)

    def forward(self, latent, regions_oi):
        rho_weight = self.rho_weight(regions_oi)
        nn_output = self.nn(latent)

        # nn_output is broadcasted across regions and across components
        logit = torch.zeros(
            (latent.shape[0], len(regions_oi), self.n_output_components),
            device=latent.device,
        )

        rho = torch.matmul(nn_output, rho_weight.T).squeeze(-1)
        return logit, rho

    def parameters_dense(self):
        return [*self.nn.parameters()]

    def parameters_sparse(self):
        return [self.rho_weight.weight]


class Model(torch.nn.Module, HybridModel):
    """
    A ChromatinHD-diff model that models the probability density of observing a cut site between clusterings
    """

    def __init__(
        self,
        fragments: Fragments,
        clustering: Clustering,
        nbins: List[int] = (
            128,
            64,
            32,
        ),
        decoder_n_layers=0,
        baseline=False,
        rho_delta_regularization=True,
        rho_delta_p_scale_free=False,
        mixture_delta_regularization=True,
        mixture_delta_p_scale_free=False,
        mixture_delta_p_scale_dist="normal",
        mixture_delta_p_scale=1.0,
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
        super().__init__()

        self.n_total_regions = fragments.n_regions

        self.n_clusters = clustering.n_clusters

        transform = DifferentialQuadraticSplineStack(
            nbins=nbins,
            n_regions=fragments.n_regions,
        )
        self.mixture = TransformedDistribution(transform)
        n_delta_mixture_components = sum(transform.split_deltas)

        if not baseline:
            self.decoder = Decoder(
                self.n_clusters,
                fragments.n_regions,
                n_delta_mixture_components,
                n_layers=decoder_n_layers,
            )
        else:
            self.decoder = BaselineDecoder(
                self.n_clusters,
                fragments.n_regions,
                n_delta_mixture_components,
                n_layers=decoder_n_layers,
            )

        # calculate libsizes and rho bias
        libsize = torch.from_numpy(np.bincount(fragments.mapping[:, 0], minlength=fragments.n_cells))

        rho_bias = (
            torch.from_numpy(np.bincount(fragments.mapping[:, 1], minlength=fragments.n_regions))
            / fragments.n_cells
            / libsize.to(torch.float).mean()
        )
        min_rho_bias = 1e-5
        rho_bias = min_rho_bias + (1 - min_rho_bias) * rho_bias
        self.register_buffer("rho_bias", rho_bias)

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

        self.rho_delta_regularization = rho_delta_regularization
        if self.rho_delta_regularization:
            if rho_delta_p_scale_free:
                self.rho_delta_p_scale = torch.nn.Parameter(torch.log(torch.tensor(0.1, requires_grad=True)))
            else:
                self.register_buffer("rho_delta_p_scale", torch.tensor(math.log(1.0)))

    def forward_(
        self,
        coordinates,
        clustering,
        regions_oi,
        local_cellxregion_ix,
        localcellxregion_ix,
        local_region_ix,
    ):
        # decode
        mixture_delta, rho_delta = self.decoder(clustering, regions_oi)

        # rho
        rho = torch.nn.functional.softmax(torch.log(self.rho_bias) + rho_delta, -1)
        rho_cuts = rho.flatten()[localcellxregion_ix]

        # fragment counts
        mixture_delta_cellxregion = mixture_delta.view(np.prod(mixture_delta.shape[:2]), mixture_delta.shape[-1])
        mixture_delta = mixture_delta_cellxregion[local_cellxregion_ix]

        self.track["likelihood_mixture"] = likelihood_mixture = self.mixture.log_prob(
            coordinates, regions_oi=regions_oi, local_region_ix=local_region_ix, delta=mixture_delta
        )

        self.track["likelihood_overall"] = likelihood_overall = torch.log(rho_cuts) + math.log(self.n_total_regions)

        # likelihood
        likelihood = self.track["likelihood"] = likelihood_mixture + likelihood_overall

        elbo = -likelihood.sum()

        # regularization
        # mixture
        if self.mixture_delta_regularization:
            mixture_delta_p = torch.distributions.Normal(0.0, torch.exp(self.mixture_delta_p_scale))
            mixture_delta_kl = mixture_delta_p.log_prob(self.decoder.logit_weight(regions_oi))

            elbo -= mixture_delta_kl.sum()

        # rho delta
        if self.rho_delta_regularization:
            rho_delta_p = torch.distributions.Normal(0.0, torch.exp(self.rho_delta_p_scale))
            rho_delta_kl = rho_delta_p.log_prob(self.decoder.rho_weight(regions_oi))

            elbo -= rho_delta_kl.sum()

        return elbo

    def forward(self, data):
        return self.forward_(
            coordinates=(data.cuts.coordinates - data.cuts.window[0]) / (data.cuts.window[1] - data.cuts.window[0]),
            clustering=data.clustering.onehot,
            regions_oi=data.minibatch.regions_oi_torch,
            local_region_ix=data.cuts.local_region_ix,
            local_cellxregion_ix=data.cuts.local_cellxregion_ix,
            localcellxregion_ix=data.cuts.localcellxregion_ix,
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
            range(fragments.n_regions),
            n_regions_step=500,
            n_cells_step=200,
        )
        minibatcher_validation = Minibatcher(
            fold["cells_validation"],
            range(fragments.n_regions),
            n_regions_step=10,
            n_cells_step=10000,
            permute_cells=False,
            permute_regions=False,
        )

        loaders_train = LoaderPool(
            ClusteringCuts,
            dict(
                clustering=clustering,
                fragments=fragments,
                cellxregion_batch_size=minibatcher_train.cellxregion_batch_size,
            ),
            n_workers=10,
        )
        loaders_validation = LoaderPool(
            ClusteringCuts,
            dict(
                clustering=clustering,
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
        clustering: Clustering,
        cells: List[str] = None,
        cell_ixs: List[int] = None,
        regions: List[str] = None,
        region_ixs: List[int] = None,
        device: str = None,
    ) -> xr.Dataset:
        """
        Returns the likelihoods of the observed cut sites for each cell and region

        Parameters:
            fragments: Fragments object
            clustering: Clustering object
            cells: Cells to predict
            cell_ixs: Cell indices to predict
            regions: Genes to predict
            region_ixs: Gene indices to predict
            device: Device to use

        Returns:
            **likelihood_mixture**, likelihood of the observing a cut site at the particular genomic location, conditioned on the region region. **likelihood_overall**, likelihood of observing a cut site in the region region
        """

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
                self._get_likelihood_cell_region(
                    self.track["likelihood_mixture"],
                    data.cuts.local_cellxregion_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_regions,
                )
                .cpu()
                .numpy()
            )
            likelihood_overall[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    region_mapping[data.minibatch.regions_oi],
                )
            ] += (
                self._get_likelihood_cell_region(
                    self.track["likelihood_overall"],
                    data.cuts.local_cellxregion_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_regions,
                )
                .cpu()
                .numpy()
            )

        self = self.to("cpu")

        result = xr.Dataset(
            {
                "likelihood_mixture": xr.DataArray(
                    likelihood_mixture,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={fragments.obs.index.name: cells, fragments.var.index.name: fragments.var.index},
                ),
                "likelihood_overall": xr.DataArray(
                    likelihood_overall,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={fragments.obs.index.name: cells, fragments.var.index.name: fragments.var.index},
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
            clustering = torch.ones((1, self.n_clusters)) * clustering

            print(clustering)

        cells_oi = torch.ones((1,), dtype=torch.long)

        local_cellxregion_ix = torch.tensor([], dtype=torch.long)
        if region_ix is None:
            if region_oi is None:
                region_oi = 0
            regions_oi = torch.tensor([region_oi], dtype=torch.long)
            local_region_ix = torch.zeros_like(coordinates).to(torch.long)
            local_cellxregion_ix = torch.zeros_like(coordinates).to(torch.long)
            localcellxregion_ix = torch.ones_like(coordinates).to(torch.long) * region_oi
        else:
            assert len(region_ix) == len(coordinates)
            regions_oi = torch.unique(region_ix)

            local_region_mapping = torch.zeros(regions_oi.max() + 1, dtype=torch.long)
            local_region_mapping.index_add_(0, regions_oi, torch.arange(len(regions_oi)))

            local_region_ix = local_region_mapping[region_ix]
            local_cell_ix = torch.arange(clustering.shape[0])
            local_cellxregion_ix = local_cell_ix * len(regions_oi) + local_region_ix
            localcellxregion_ix = local_cell_ix * self.n_total_regions + region_ix

        data = ClusteringCutsResult(
            cuts=CutsResult(
                coordinates=coordinates,
                local_cellxregion_ix=local_cellxregion_ix,
                localcellxregion_ix=localcellxregion_ix,
                n_regions=len(regions_oi),
                n_fragments=len(coordinates),
                n_cuts=len(coordinates),
                window=torch.tensor([0, 1]),
            ),
            clustering=ClusteringResult(
                onehot=clustering,
            ),
            minibatch=Minibatch(
                cells_oi=cells_oi.cpu().numpy(),
                regions_oi=regions_oi.cpu().numpy(),
            ),
        ).to(device)

        self = self.to(device).eval()

        with torch.no_grad():
            self.forward(data)

        self = self.to("cpu")

        prob = self.track["likelihood"].detach().cpu()
        return prob.detach().cpu()


class Models(Flow):
    n_models = Stored()

    @property
    def models_path(self):
        path = self.path / "models"
        path.mkdir(exist_ok=True)
        return path

    def train_models(self, fragments, clustering, folds, device=None, n_epochs=30, **kwargs):
        """
        Trains the models

        Parameters:
            fragments:
                Fragments object
        """
        self.n_models = len(folds)
        for fold_ix, fold in [(fold_ix, fold) for fold_ix, fold in enumerate(folds)]:
            desired_outputs = [self.models_path / ("model_" + str(fold_ix) + ".pkl")]
            force = False
            if not all([desired_output.exists() for desired_output in desired_outputs]):
                force = True

            if force:
                model = Model(fragments, clustering, **kwargs)
                model.train_model(fragments, clustering, fold, device=device, n_epochs=n_epochs)

                model = model.to("cpu")

                pickle.dump(
                    model,
                    open(self.models_path / ("model_" + str(fold_ix) + ".pkl"), "wb"),
                )

    def __getitem__(self, ix):
        return pickle.load((self.models_path / ("model_" + str(ix) + ".pkl")).open("rb"))

    def __len__(self):
        return self.n_models

    def __iter__(self):
        for ix in range(len(self)):
            yield self[ix]

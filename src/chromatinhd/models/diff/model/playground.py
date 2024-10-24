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
from chromatinhd.models.diff.loader.clustering_fragments import (
    ClusteringFragments,
)
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
    ):
        super().__init__()

        self.n_output_components = n_output_components

        self.delta_height_weight = EmbeddingTensor(
            n_regions,
            (
                n_latent,
                n_output_components,
            ),
            sparse=True,
        )
        stdv = 1.0 / math.sqrt(self.delta_height_weight.weight.size(1))
        # self.delta_height_weight.weight.data.uniform_(-stdv, stdv)
        self.delta_height_weight.weight.data.zero_()

        self.delta_baseline_weight = torch.nn.Parameter(torch.zeros((n_regions, n_latent)))
        stdv = 1.0 / math.sqrt(self.delta_baseline_weight.size(1))
        self.delta_baseline_weight.data.uniform_(-stdv, stdv)

    def forward(self, latent, regions_oi):
        # regions oi is only used to get the delta_heights
        # we calculate the delta_baseline for all regions because we pool using softmax later
        delta_height_weight = self.delta_height_weight(regions_oi)
        delta_baseline_weight = self.delta_baseline_weight

        # nn_output is broadcasted across regions and across components
        delta_height = torch.matmul(latent.unsqueeze(1).unsqueeze(2), delta_height_weight).squeeze(-2)

        # nn_output has to be broadcasted across regions
        delta_baseline = torch.matmul(latent.unsqueeze(1), delta_baseline_weight.T).squeeze(-2)

        return delta_height, delta_baseline

    def parameters_dense(self):
        return [self.delta_baseline_weight]

    def parameters_sparse(self):
        return [self.delta_height_weight.weight]


class SineEncoding(torch.nn.Module):
    def __init__(self, n_frequencies, n_coordinates=2):
        super().__init__()

        self.register_buffer(
            "frequencies",
            torch.tensor([[1 / 1000 ** (2 * i / n_frequencies)] * 2 for i in range(1, n_frequencies + 1)]).flatten(-2),
        )
        self.register_buffer(
            "shifts",
            torch.tensor([[0, torch.pi / 2] for _ in range(1, n_frequencies + 1)]).flatten(-2),
        )

        self.n_embedding_dimensions = n_frequencies * 2 * n_coordinates

    def forward(self, coordinates):
        embedding = torch.sin((coordinates[..., None] * self.frequencies + self.shifts).flatten(-2))
        return embedding


class CutEmbedderSine(torch.nn.Module):
    dropout_rate = 0.0

    def __init__(
        self,
        n_regions,
        n_frequencies=10,
        n_embedding_dimensions=20,
        n_output_dimensions=1,
        n_layers=1,
        dropout_rate=0.0,
        **kwargs,
    ):
        self.n_embedding_dimensions = n_embedding_dimensions
        self.n_output_dimensions = n_output_dimensions

        self.dropout_rate = dropout_rate

        super().__init__(**kwargs)

        self.sine_encoding = SineEncoding(n_frequencies=n_frequencies, n_coordinates=1)

        layers = []
        if self.dropout_rate > 0:
            layers.append(torch.nn.Dropout(self.dropout_rate))
        for layer_ix in range(n_layers):
            if layer_ix == 0:
                layers.append(
                    torch.nn.Linear(
                        self.sine_encoding.n_embedding_dimensions,
                        self.n_embedding_dimensions,
                    )
                )
            else:
                layers.append(torch.nn.Linear(self.n_embedding_dimensions, self.n_embedding_dimensions))
            if (self.dropout_rate > 0) and (layer_ix < n_layers - 1) and (n_layers > 1):
                layers.append(torch.nn.Dropout(self.dropout_rate))
            if layer_ix == 0:
                layers.append(torch.nn.Sigmoid())
            else:
                layers.append(torch.nn.ReLU())

        self.bias1 = EmbeddingTensor(
            n_regions,
            (self.n_output_dimensions,),
            sparse=True,
        )
        self.bias1.data.zero_()

        self.weight1 = EmbeddingTensor(
            n_regions,
            (
                self.n_embedding_dimensions if n_layers > 0 else self.sine_encoding.n_embedding_dimensions,
                self.n_output_dimensions,
            ),
            sparse=True,
        )
        self.weight1.data.zero_()

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, coordinates, region_ix):
        embedding = self.sine_encoding(coordinates)
        embedding = self.nn(embedding)
        embedding = torch.einsum("ab,abc->ac", embedding, self.weight1(region_ix))  # + self.bias1(region_ix)

        return embedding

    def parameters_sparse(self):
        return [self.bias1.weight, self.weight1.weight]


class CutEmbedderDummy(torch.nn.Module):
    dropout_rate = 0.0

    def __init__(
        self,
        n_regions,
        n_frequencies=10,
        n_embedding_dimensions=20,
        n_output_dimensions=1,
        n_layers=1,
        dropout_rate=0.0,
        **kwargs,
    ):
        self.n_embedding_dimensions = n_embedding_dimensions
        self.n_output_dimensions = n_output_dimensions

        super().__init__(**kwargs)

        self.nn = torch.nn.Sequential(torch.nn.Linear(1, self.n_output_dimensions))

    def forward(self, coordinates, region_ix):
        embedding = self.nn(coordinates.float() / 20000)

        return embedding


class CutEmbedderDirect(torch.nn.Module):
    dropout_rate = 0.0

    def __init__(
        self,
        n_regions,
        n_frequencies=10,
        n_embedding_dimensions=20,
        n_output_dimensions=1,
        n_layers=1,
        dropout_rate=0.2,
        **kwargs,
    ):
        self.n_embedding_dimensions = n_embedding_dimensions
        self.n_output_dimensions = n_output_dimensions

        self.dropout_rate = dropout_rate

        super().__init__(**kwargs)

        layers = []
        for layer_ix in range(n_layers):
            if layer_ix == 0:
                layers.append(
                    torch.nn.Linear(
                        1,
                        self.n_embedding_dimensions,
                    )
                )
            else:
                layers.append(torch.nn.Linear(self.n_embedding_dimensions, self.n_embedding_dimensions))
            layers.append(torch.nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(torch.nn.Dropout(self.dropout_rate))
        # layers.append(torch.nn.Linear(self.n_embedding_dimensions, self.n_output_dimensions))

        self.bias1 = EmbeddingTensor(
            n_regions,
            (self.n_output_dimensions,),
            sparse=True,
        )
        self.bias1.data.zero_()

        self.weight1 = EmbeddingTensor(
            n_regions,
            (
                self.n_embedding_dimensions if n_layers > 0 else self.sine_encoding.n_embedding_dimensions,
                self.n_output_dimensions,
            ),
            sparse=True,
        )
        self.weight1.data.zero_()

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, coordinates, region_ix):
        embedding = self.nn(coordinates / 20000)
        embedding = torch.einsum("ab,abc->ac", embedding, self.weight1(region_ix)) + self.bias1(region_ix)

        return embedding

    def parameters_sparse(self):
        return [self.bias1.weight, self.weight1.weight]


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
        delta_baseline_regularization=True,
        delta_baseline_p_scale_free=False,
        delta_height_regularization=True,
        delta_height_p_scale_free=False,
        delta_height_p_scale_dist="normal",
        delta_height_p_scale=1.0,
        cut_embedder="sine",
        cut_embedder_dropout_rate=0.1,
    ):
        super().__init__()

        self.n_total_regions = fragments.n_regions

        self.n_clusters = clustering.n_clusters

        transform = DifferentialQuadraticSplineStack(
            nbins=nbins,
            n_regions=fragments.n_regions,
        )
        self.mixture = TransformedDistribution(transform)
        n_delta_mixture_components = sum(transform.split_deltas)

        self.decoder = Decoder(
            self.n_clusters,
            fragments.n_regions,
            n_delta_mixture_components,
        )

        # calculate baseline bias and libsizes
        libsize = torch.from_numpy(np.bincount(fragments.mapping[:, 0], minlength=fragments.n_cells))

        baseline_bias = (
            torch.from_numpy(np.bincount(fragments.mapping[:, 1], minlength=fragments.n_regions))
            / fragments.n_cells
            / libsize.to(torch.float).mean()
        )
        min_baseline_bias = 1e-5
        baseline_bias = min_baseline_bias + (1 - min_baseline_bias) * baseline_bias
        self.register_buffer("baseline_bias", baseline_bias)

        self.track = {}

        self.delta_height_regularization = delta_height_regularization
        if self.delta_height_regularization:
            if delta_height_p_scale_free:
                self.delta_height_p_scale = torch.nn.Parameter(
                    torch.tensor(math.log(delta_height_p_scale), requires_grad=True)
                )
            else:
                self.register_buffer(
                    "delta_height_p_scale",
                    torch.tensor(math.log(delta_height_p_scale)),
                )
        self.delta_height_p_scale_dist = delta_height_p_scale_dist

        self.delta_baseline_regularization = delta_baseline_regularization
        if self.delta_baseline_regularization:
            if delta_baseline_p_scale_free:
                self.delta_baseline_p_scale = torch.nn.Parameter(torch.log(torch.tensor(0.1, requires_grad=True)))
            else:
                self.register_buffer("delta_baseline_p_scale", torch.tensor(math.log(1.0)))

        self.right_normalize = 200 / 20000
        self.right_scale = torch.nn.Parameter(
            torch.tensor(math.log(50 / 20000 / self.right_normalize), requires_grad=True)
        )
        self.right_loc = torch.nn.Parameter(torch.tensor(200 / 20000 / self.right_normalize, requires_grad=True))

        if cut_embedder == "sine":
            self.right_scale_nn = CutEmbedderSine(
                n_regions=len(fragments.var),
                n_layers=3,
                n_embedding_dimensions=20,
                n_frequencies=20,
                n_output_dimensions=2,
                dropout_rate=cut_embedder_dropout_rate,
            )
        elif cut_embedder == "direct":
            self.right_scale_nn = CutEmbedderDirect(
                n_regions=len(fragments.var),
                n_layers=3,
                n_embedding_dimensions=20,
                n_output_dimensions=2,
            )
        elif cut_embedder == "dummy":
            self.right_scale_nn = CutEmbedderDummy(
                n_regions=len(fragments.var),
                n_output_dimensions=2,
            )

    def forward(self, data, shuffle_leftright=False):
        # decode
        delta_height, delta_baseline = self.decoder(data.clustering.onehot, data.minibatch.regions_oi_torch)

        # baseline
        baseline = torch.nn.functional.softmax(torch.log(self.baseline_bias) + delta_baseline, -1)
        baseline_cuts = baseline.flatten()[data.fragments.localcellxregion_ix]

        # fragment counts
        delta_height_cellxregion = delta_height.view(np.prod(delta_height.shape[:2]), delta_height.shape[-1])
        delta_height = delta_height_cellxregion[data.fragments.local_cellxregion_ix]

        # randomly select first or second column of coordinates
        if shuffle_leftright:
            selection = torch.randint(0, 2, (data.fragments.n_fragments,)).to(
                data.fragments.coordinates.device, torch.long
            )
            coordinates_left = torch.gather(data.fragments.coordinates, 1, selection[:, None])[:, 0]
            coordinates_right = torch.gather(data.fragments.coordinates, 1, 1 - selection[:, None])[:, 0]
        else:
            coordinates_left = data.fragments.coordinates[:, 0]
            coordinates_right = data.fragments.coordinates[:, 1]

        coordinates_left_unscaled = coordinates_left
        coordinates_left = (coordinates_left - data.fragments.window[0]) / (
            data.fragments.window[1] - data.fragments.window[0]
        )
        # coordinates_left = torch.clamp(coordinates_left, 0, 1)
        coordinates_right = (coordinates_right - data.fragments.window[0]) / (
            data.fragments.window[1] - data.fragments.window[0]
        )
        # coordinates_right = torch.clamp(coordinates_right, 0, 1)

        likelihood_position_left, out_left = self.mixture.log_prob(
            coordinates_left,
            regions_oi=data.minibatch.regions_oi_torch,
            local_region_ix=data.fragments.local_region_ix,
            delta=delta_height,
            return_transformed=True,
        )
        self.track["likelihood_position_left"] = likelihood_position_left

        from .truncated_normal import apply_trunc_normal, apply_normal

        # output, likelihood_position_right = apply_trunc_normal(coordinates_right, coordinates_left, self.right_scale)

        likelihood_position_right = torch.zeros_like(likelihood_position_left)
        nn_output = self.right_scale_nn(
            coordinates_left_unscaled[:, None], region_ix=data.fragments.regionmapping
        ).squeeze(-1)
        right_scale = torch.exp(self.right_scale + nn_output[:, 0] * 0.01) * self.right_normalize
        right_loc = (self.right_loc + nn_output[:, 1]) * self.right_normalize
        output, logabsdet = apply_normal(coordinates_right - coordinates_left, right_loc, right_scale)
        likelihood_position_right += logabsdet

        self.track["likelihood_position_right"] = likelihood_position_right

        self.track["likelihood_position"] = likelihood_position = likelihood_position_left + likelihood_position_right

        # for region
        self.track["likelihood_region"] = likelihood_region = torch.log(baseline_cuts) + math.log(self.n_total_regions)

        # likelihood
        likelihood = self.track["likelihood"] = likelihood_position + likelihood_region

        elbo = -likelihood.sum()

        # regularization
        # mixture
        if self.delta_height_regularization:
            delta_height_p = torch.distributions.Normal(0.0, torch.exp(self.delta_height_p_scale))
            delta_height_kl = delta_height_p.log_prob(self.decoder.delta_height_weight(data.minibatch.regions_oi_torch))

            elbo -= delta_height_kl.sum()

        # baseline delta
        if self.delta_baseline_regularization:
            delta_baseline_p = torch.distributions.Normal(0.0, torch.exp(self.delta_baseline_p_scale))
            delta_baseline_kl = delta_baseline_p.log_prob(self.decoder.delta_baseline_weight)

            elbo -= delta_baseline_kl.sum()

        return elbo

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
            ClusteringFragments,
            dict(
                clustering=clustering,
                fragments=fragments,
                cellxregion_batch_size=minibatcher_train.cellxregion_batch_size,
            ),
            n_workers=10,
        )
        loaders_validation = LoaderPool(
            ClusteringFragments,
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
            **likelihood_position**, likelihood of the observing a cut site at the particular genomic location, conditioned on the region region. **likelihood_region**, likelihood of observing a cut site in the region region
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
            ClusteringFragments,
            dict(
                clustering=clustering,
                fragments=fragments,
                cellxregion_batch_size=minibatches.cellxregion_batch_size,
            ),
            n_workers=5,
        )
        loaders.initialize(minibatches)

        likelihood_position = np.zeros((len(cell_ixs), len(region_ixs)))
        likelihood_region = np.zeros((len(cell_ixs), len(region_ixs)))
        likelihood = np.zeros((len(cell_ixs), len(region_ixs)))

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

            likelihood_position[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    region_mapping[data.minibatch.regions_oi],
                )
            ] += (
                self._get_likelihood_cell_region(
                    self.track["likelihood_position"],
                    data.fragments.local_cellxregion_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_regions,
                )
                .cpu()
                .numpy()
            )
            likelihood_region[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    region_mapping[data.minibatch.regions_oi],
                )
            ] += (
                self._get_likelihood_cell_region(
                    self.track["likelihood_region"],
                    data.fragments.local_cellxregion_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_regions,
                )
                .cpu()
                .numpy()
            )
            likelihood[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    region_mapping[data.minibatch.regions_oi],
                )
            ] += (
                self._get_likelihood_cell_region(
                    self.track["likelihood"],
                    data.fragments.local_cellxregion_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_regions,
                )
                .cpu()
                .numpy()
            )

        self = self.to("cpu")

        result = xr.Dataset(
            {
                "likelihood_position": xr.DataArray(
                    likelihood_position,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={"cell": cells, "region": fragments.var.index},
                ),
                "likelihood_region": xr.DataArray(
                    likelihood_region,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={"cell": cells, "region": fragments.var.index},
                ),
                "likelihood": xr.DataArray(
                    likelihood,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={"cell": cells, "region": fragments.var.index},
                ),
            }
        )
        return result

    def evaluate_right(
        self,
        coordinates,
        coordinates2,
        window,
        region_ix=None,
        cluster_ix=None,
        clustering=None,
        device=None,
    ):
        from chromatinhd.loaders.clustering import Result as ClusteringResult
        from chromatinhd.models.diff.loader.clustering_fragments import (
            Result as ClusteringFragmentsResult,
        )
        from chromatinhd.loaders.fragments import Result as FragmentsResult
        from chromatinhd.loaders.minibatches import Minibatch

        if not torch.is_tensor(clustering):
            if cluster_ix is not None:
                # one hot
                clustering = torch.nn.functional.one_hot(cluster_ix, self.n_clusters)
            if clustering is None:
                clustering = 0.0
            clustering = torch.ones((1, self.n_clusters)) * clustering

        cells_oi = torch.arange(coordinates.shape[0])

        local_cellxregion_ix = torch.tensor([], dtype=torch.long)
        assert len(region_ix) == len(coordinates)
        regions_oi = torch.unique(region_ix)

        local_region_mapping = torch.zeros(regions_oi.max() + 1, dtype=torch.long)
        local_region_mapping.index_add_(0, regions_oi, torch.arange(len(regions_oi)))

        local_region_ix = local_region_mapping[region_ix]
        local_cell_ix = torch.arange(clustering.shape[0])
        local_cellxregion_ix = local_cell_ix * len(regions_oi) + local_region_ix
        localcellxregion_ix = local_cell_ix * self.n_total_regions + region_ix

        regionmapping = regions_oi[local_region_ix]

        coordinates = torch.stack([coordinates, coordinates2], -1)

        assert (coordinates[:, 1] <= window[1]).all()
        assert (coordinates[:, 0] >= window[0]).all()

        data = ClusteringFragmentsResult(
            fragments=FragmentsResult(
                coordinates=coordinates,
                local_cellxregion_ix=local_cellxregion_ix,
                localcellxregion_ix=localcellxregion_ix,
                regionmapping=regionmapping,
                n_fragments=coordinates.shape[0],
                cells_oi=cells_oi,
                regions_oi=regions_oi,
                window=window,
                n_total_regions=self.n_total_regions,
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
            self.forward(data, shuffle_leftright=False)

        self = self.to("cpu")

        likelihood_position_left = self.track["likelihood_position_left"].detach().cpu()
        likelihood_position_right = self.track["likelihood_position_right"].detach().cpu()
        return likelihood_position_left.detach().cpu(), likelihood_position_right.detach().cpu()

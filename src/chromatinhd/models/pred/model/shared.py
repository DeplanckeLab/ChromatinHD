"""
Additive model for predicting region expression from fragments
"""
from __future__ import annotations

import torch
# import torch_scatter
import math
import numpy as np
import xarray as xr
import pandas as pd
import itertools

import pickle

from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.models import FlowModel
from chromatinhd.flow import Flow, Stored, StoredDict, Linked

from chromatinhd.loaders.minibatches import Minibatcher
from chromatinhd.loaders.transcriptome_fragments import (
    TranscriptomeFragments,
)
from chromatinhd.data.fragments import Fragments
from chromatinhd.data.transcriptome import Transcriptome
from chromatinhd.models.pred.trainer import SharedTrainer
from chromatinhd.loaders import LoaderPool
from chromatinhd.optim import SparseDenseAdam

from chromatinhd import get_default_device

from .loss import paircor, paircor_loss, region_paircor_loss, pairzmse_loss, region_pairzmse_loss

from typing import Any

from . import encoders


class FragmentEmbedder(torch.nn.Module):
    dropout_rate = 0.0

    def __init__(
        self,
        n_regions,
        n_frequencies=10,
        n_embedding_dimensions=5,
        n_layers=2,
        nonlinear=True,
        dropout_rate=0.0,
        residual=False,
        batchnorm=False,
        encoder="sine2",
        fragments=None,
        distance_encoder=None,
        **kwargs,
    ):
        self.nonlinear = nonlinear
        self.residual = residual

        super().__init__(**kwargs)

        # make encoding
        if encoder is None:
            self.encoder = encoders.NoneEncoding()
        elif encoder == "ones":
            self.encoder = encoders.OneEncoding()
        elif encoder == "sine2":
            self.encoder = encoders.SineEncoding2(n_frequencies=n_frequencies)
        elif encoder == "sine":
            self.encoder = encoders.SineEncoding(n_frequencies=n_frequencies)
        elif encoder == "sine3":
            self.encoder = encoders.SineEncoding3(n_frequencies=n_frequencies)
        elif encoder == "radial":
            self.encoder = encoders.RadialEncoding(n_frequencies=n_frequencies, window=fragments.regions.window)
        elif encoder == "tophat":
            self.encoder = encoders.TophatEncoding(n_frequencies=n_frequencies, window=fragments.regions.window)
        elif encoder == "direct":
            self.encoder = encoders.DirectEncoding(window=fragments.regions.window)
        elif encoder == "exponential":
            self.encoder = encoders.ExponentialEncoding(window=fragments.regions.window)
        elif encoder == "radial_binary":
            self.encoder = encoders.RadialBinaryEncoding(n_frequencies=n_frequencies, window=fragments.regions.window)
        else:
            self.encoder = encoder(n_frequencies=n_frequencies)

        self.n_embedding_dimensions = n_embedding_dimensions if n_layers > 0 else self.encoder.n_embedding_dimensions
        n_embedding_dimensions = self.encoder.n_embedding_dimensions

        # make distance encoding
        if distance_encoder is not None:
            if distance_encoder == "direct":
                self.distance_encoder = encoders.DirectDistanceEncoding()
            elif distance_encoder == "split":
                self.distance_encoder = encoders.SplitDistanceEncoding()
            else:
                raise ValueError(distance_encoder + " is not a valid distance encoder")
            n_embedding_dimensions += self.distance_encoder.n_embedding_dimensions
        else:
            self.distance_encoder = None

        # make layers
        self.layers = []

        for i in range(n_layers):
            sublayers = []
            sublayers.append(
                torch.nn.Linear(
                    self.n_embedding_dimensions if i > 0 else n_embedding_dimensions,
                    self.n_embedding_dimensions,
                )
            )
            if (i > 0) and (batchnorm):
                sublayers.append(torch.nn.BatchNorm1d(self.n_embedding_dimensions))
                # sublayers.append(torch.nn.LayerNorm((self.n_embedding_dimensions,)))
            if nonlinear is not False:
                if nonlinear is True:
                    sublayers.append(torch.nn.GELU())
                elif nonlinear == "relu":
                    sublayers.append(torch.nn.ReLU())
                else:
                    sublayers.append(nonlinear())
            if self.dropout_rate > 0:
                sublayers.append(torch.nn.Dropout(p=dropout_rate))
            nn = torch.nn.Sequential(*sublayers)
            self.add_module("nn{}".format(i), nn)
            self.layers.append(nn)

    def forward(self, coordinates, region_ix):
        embedding = self.encoder(coordinates)
        if self.distance_encoder is not None:
            embedding = torch.cat([embedding, self.distance_encoder(coordinates)], dim=-1)
        for i, layer in enumerate(self.layers):
            embedding2 = layer(embedding)
            if (self.residual) and (i > 0):
                embedding = embedding + embedding2
            else:
                embedding = embedding2

        return embedding


class EmbeddingGenePooler(torch.nn.Module):
    """
    Pools fragments across regions and cells
    """

    def __init__(self, reduce="sum"):
        self.reduce = reduce
        super().__init__()

    def forward(self, embedding, fragment_cellxregion_ix, cell_n, region_n):
        if self.reduce == "mean":
            cellxregion_embedding = torch_scatter.segment_mean_coo(
                embedding, fragment_cellxregion_ix, dim_size=cell_n * region_n
            )
        elif self.reduce == "sum":
            cellxregion_embedding = torch_scatter.segment_sum_coo(
                embedding, fragment_cellxregion_ix, dim_size=cell_n * region_n
            )
        else:
            raise ValueError()
        cell_region_embedding = cellxregion_embedding.reshape((cell_n, region_n, cellxregion_embedding.shape[-1]))
        return cell_region_embedding


class EmbeddingToExpression(torch.nn.Module):
    """
    Predicts region expression using a [cell, region, component] embedding in a region-specific manner
    """

    def __init__(
        self,
        n_regions,
        n_embedding_dimensions=5,
        initialization="default",
        n_layers=1,
        dropout_rate=0.0,
        nonlinear=True,
        residual=False,
        batchnorm=False,
        **kwargs,
    ):
        self.n_regions = n_regions
        self.n_embedding_dimensions = n_embedding_dimensions
        self.residual = residual

        super().__init__()

        # make layers
        self.layers = []

        for i in range(n_layers):
            sublayers = []
            sublayers.append(
                torch.nn.Linear(
                    self.n_embedding_dimensions,
                    self.n_embedding_dimensions,
                )
            )
            if batchnorm:
                sublayers.append(torch.nn.BatchNorm1d(n_embedding_dimensions))
            if nonlinear:
                sublayers.append(torch.nn.GELU())
            if dropout_rate > 0:
                sublayers.append(torch.nn.Dropout(p=dropout_rate))
            nn = torch.nn.Sequential(*sublayers)
            self.add_module("nn{}".format(i), nn)
            self.layers.append(nn)

        nn = torch.nn.Linear(self.n_embedding_dimensions, 1)
        self.final = nn
        nn.weight.data[:] = 0
        self.layers.append(nn)

    def forward(self, cell_region_embedding, region_ix):
        # print(cell_region_embedding.max())
        embedding = cell_region_embedding.reshape(-1, self.n_embedding_dimensions)
        # print(embedding.max())
        for i, layer in enumerate(self.layers):
            embedding2 = layer(embedding)
            if (self.residual) and (i < len(self.layers) - 1):
                embedding = embedding + embedding2
            else:
                embedding = embedding2
            # print(embedding.max())
        # print("--")

        return embedding.reshape(cell_region_embedding.shape[:-1])


class Model(FlowModel):
    """
    Predicting region expression from raw fragments using an additive model across fragments from the same cell

    Parameters:
        n_regions:
            the number of regions
        dummy:
            whether to use a dummy model that just counts fragments.
        n_frequencies:
            the number of frequencies to use for sine encoding
        reduce:
            the reduction to use for pooling fragments across regions and cells
        nonlinear:
            whether to use a non-linear activation function
        n_embedding_dimensions:
            the number of embedding dimensions
        dropout_rate:
            the dropout rate
    """

    transcriptome = Linked()
    """The transcriptome"""

    fragments = Linked()
    """The fragments"""

    fold = Stored()
    """The cells used for training, test and validation"""

    layer = Stored()
    """The layer of the transcriptome"""

    regions_oi = None

    def __init__(
        self,
        path=None,
        fragments: Fragments | None = None,
        transcriptome: Transcriptome | None = None,
        fold=None,
        dummy: bool = False,
        n_frequencies: int = 50,
        reduce: str = "sum",
        nonlinear: bool = True,
        n_embedding_dimensions: int = 10,
        embedding_to_expression_initialization: str = "default",
        dropout_rate_fragment_embedder: float = 0.0,
        n_layers_fragment_embedder=1,
        residual_fragment_embedder=False,
        batchnorm_fragment_embedder=False,
        n_layers_embedding2expression=1,
        dropout_rate_embedding2expression: float = 0.0,
        residual_embedding2expression=False,
        batchnorm_embedding2expression=False,
        layer=None,
        reset=False,
        encoder=None,
        distance_encoder=None,
        regions_oi=None,
        **kwargs: Any,
    ) -> None:
        super().__init__(path=path, reset=reset, **kwargs)

        if fragments is not None:
            self.fragments = fragments
        if transcriptome is not None:
            self.transcriptome = transcriptome
        if fold is not None:
            self.fold = fold
        if layer is not None:
            self.layer = layer
        elif not self.o.layer.exists(self):
            if transcriptome is not None:
                self.layer = list(self.transcriptome.layers.keys())[0]

        self.regions_oi = regions_oi

        if not self.o.state.exists(self):
            assert fragments is not None
            assert transcriptome is not None

            n_regions = fragments.n_regions

            self.fragment_embedder = FragmentEmbedder(
                n_frequencies=n_frequencies,
                n_regions=n_regions,
                nonlinear=nonlinear,
                n_layers=n_layers_fragment_embedder,
                n_embedding_dimensions=n_embedding_dimensions,
                dropout_rate=dropout_rate_fragment_embedder,
                residual=residual_fragment_embedder,
                batchnorm=batchnorm_fragment_embedder,
                fragments=fragments,
                encoder=encoder,
                distance_encoder=distance_encoder,
            )
            self.embedding_region_pooler = EmbeddingGenePooler(reduce=reduce)
            self.embedding_to_expression = EmbeddingToExpression(
                n_regions=n_regions,
                n_embedding_dimensions=self.fragment_embedder.n_embedding_dimensions,
                initialization=embedding_to_expression_initialization,
                n_layers=n_layers_embedding2expression,
                residual=residual_embedding2expression,
                dropout_rate=dropout_rate_embedding2expression,
                batchnorm=batchnorm_embedding2expression,
            )

    def forward(self, data):
        """
        Make a prediction given a data object
        """
        fragment_embedding = self.fragment_embedder(data.fragments.coordinates, data.fragments.regionmapping)
        cell_region_embedding = self.embedding_region_pooler(
            fragment_embedding,
            data.fragments.local_cellxregion_ix,
            data.minibatch.n_cells,
            data.minibatch.n_regions,
        )
        expression_predicted = self.embedding_to_expression(cell_region_embedding, data.minibatch.regions_oi_torch)
        return expression_predicted

    def forward_loss(self, data):
        """
        Make a prediction and calculate the loss given a data object
        """
        expression_predicted = self.forward(data)
        expression_true = data.transcriptome.value
        return paircor_loss(expression_predicted, expression_true)
        # return pairzmse_loss(expression_predicted, expression_true)

    def forward_region_loss(self, data):
        """
        Make a prediction and calculate the loss given a data object
        """
        expression_predicted = self.forward(data)
        expression_true = data.transcriptome.value
        return region_paircor_loss(expression_predicted, expression_true)
        # return region_pairzmse_loss(expression_predicted, expression_true)

    def forward_multiple(self, data, fragments_oi, min_fragments=1):
        fragment_embedding = self.fragment_embedder(data.fragments.coordinates, data.fragments.regionmapping)

        total_n_fragments = torch.bincount(
            data.fragments.local_cellxregion_ix,
            minlength=data.minibatch.n_regions * data.minibatch.n_cells,
        ).reshape((data.minibatch.n_cells, data.minibatch.n_regions))

        total_cell_region_embedding = self.embedding_region_pooler.forward(
            fragment_embedding,
            data.fragments.local_cellxregion_ix,
            data.minibatch.n_cells,
            data.minibatch.n_regions,
        )

        total_expression_predicted = self.embedding_to_expression.forward(
            total_cell_region_embedding, data.minibatch.regions_oi_torch
        )

        for fragments_oi_ in fragments_oi:
            if (fragments_oi_ is not None) and ((~fragments_oi_).sum() > min_fragments):
                lost_fragments_oi = ~fragments_oi_
                lost_local_cellxregion_ix = data.fragments.local_cellxregion_ix[lost_fragments_oi]
                n_fragments = total_n_fragments - torch.bincount(
                    lost_local_cellxregion_ix,
                    minlength=data.minibatch.n_regions * data.minibatch.n_cells,
                ).reshape((data.minibatch.n_cells, data.minibatch.n_regions))
                cell_region_embedding = total_cell_region_embedding - self.embedding_region_pooler.forward(
                    fragment_embedding[lost_fragments_oi],
                    lost_local_cellxregion_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_regions,
                )

                expression_predicted = self.embedding_to_expression.forward(
                    cell_region_embedding, data.minibatch.regions_oi_torch
                )
            else:
                n_fragments = total_n_fragments
                expression_predicted = total_expression_predicted

            yield expression_predicted, n_fragments

    def train_model(
        self,
        fold: list = None,
        fragments: Fragments = None,
        transcriptome: Transcriptome = None,
        device=None,
        lr=1e-4,
        n_epochs=1000,
        pbar=True,
        n_regions_step=50,
        n_cells_step=10000,
        weight_decay=1e-2,
        checkpoint_every_epoch=1,
    ):
        """
        Train the model
        """
        if fold is None:
            fold = self.fold
        assert fold is not None

        if fragments is None:
            fragments = self.fragments
        if transcriptome is None:
            transcriptome = self.transcriptome

        minibatcher_train = Minibatcher(
            fold["cells_train"],
            fold["regions_train"],
            n_regions_step=n_regions_step,
            n_cells_step=n_cells_step,
            permute_cells=True,
            permute_regions=True,
        )
        minibatcher_validation = Minibatcher(
            fold["cells_validation"],
            fold["regions_validation"],
            n_regions_step=100,
            n_cells_step=10000,
            permute_cells=False,
            permute_regions=False,
        )

        if device is None:
            device = get_default_device()

        loaders_train = LoaderPool(
            TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxregion_batch_size=minibatcher_train.cellxregion_batch_size,
                layer=self.layer,
            ),
            n_workers=10,
        )
        loaders_validation = LoaderPool(
            TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxregion_batch_size=minibatcher_validation.cellxregion_batch_size,
                layer=self.layer,
            ),
            n_workers=5,
        )

        trainer = SharedTrainer(
            self,
            loaders_train,
            loaders_validation,
            minibatcher_train,
            minibatcher_validation,
            # torch.optim.SGD(
            #     itertools.chain(self.parameters_sparse(), self.parameters_dense()),
            #     momentum=0.5,
            #     weight_decay=weight_decay,
            #     lr=lr,
            # ),
            SparseDenseAdam(
                self.parameters_sparse(),
                self.parameters_dense(),
                lr=lr,
                weight_decay=weight_decay,
            ),
            n_epochs=n_epochs,
            checkpoint_every_epoch=checkpoint_every_epoch,
            optimize_every_step=1,
            device=device,
            pbar=pbar,
        )

        self.trace = trainer.trace

        trainer.train()
        # trainer.trace.plot()

    def get_prediction(
        self,
        fragments=None,
        transcriptome=None,
        cells=None,
        cell_ixs=None,
        regions=None,
        region_ixs=None,
        device=None,
        return_raw=False,
    ):
        """
        Returns the prediction of a dataset
        """

        if fragments is None:
            fragments = self.fragments
        if transcriptome is None:
            transcriptome = self.transcriptome
        if cell_ixs is None:
            if cells is None:
                cells = fragments.obs.index
            fragments.obs["ix"] = np.arange(len(fragments.obs))
            cell_ixs = fragments.obs.loc[cells]["ix"].values
        if cells is None:
            cells = fragments.obs.index[cell_ixs]

        if region_ixs is None:
            if regions is None:
                if self.regions_oi is not None:
                    regions = self.regions_oi
                else:
                    regions = fragments.var.index
            fragments.var["ix"] = np.arange(len(fragments.var))
            region_ixs = fragments.var.loc[regions]["ix"].values
        if regions is None:
            regions = fragments.var.index[region_ixs]

        if device is None:
            device = get_default_device()

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
            TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxregion_batch_size=minibatches.cellxregion_batch_size,
                layer=self.layer,
            ),
            n_workers=5,
        )
        loaders.initialize(minibatches)

        predicted = np.zeros((len(cell_ixs), len(region_ixs)))
        expected = np.zeros((len(cell_ixs), len(region_ixs)))
        n_fragments = np.zeros((len(cell_ixs), len(region_ixs)))

        cell_mapping = np.zeros(fragments.n_cells, dtype=np.int64)
        cell_mapping[cell_ixs] = np.arange(len(cell_ixs))

        region_mapping = np.zeros(fragments.n_regions, dtype=np.int64)
        region_mapping[region_ixs] = np.arange(len(region_ixs))

        self.eval()
        self = self.to(device)

        for data in loaders:
            data = data.to(device)
            with torch.no_grad():
                pred_mb = self.forward(data)
            predicted[
                np.ix_(cell_mapping[data.minibatch.cells_oi], region_mapping[data.minibatch.regions_oi])
            ] = pred_mb.cpu().numpy()
            expected[
                np.ix_(cell_mapping[data.minibatch.cells_oi], region_mapping[data.minibatch.regions_oi])
            ] = data.transcriptome.value.cpu().numpy()
            n_fragments[np.ix_(cell_mapping[data.minibatch.cells_oi], region_mapping[data.minibatch.regions_oi])] = (
                torch.bincount(
                    data.fragments.local_cellxregion_ix,
                    minlength=len(data.minibatch.cells_oi) * len(data.minibatch.regions_oi),
                )
                .reshape(len(data.minibatch.cells_oi), len(data.minibatch.regions_oi))
                .cpu()
                .numpy()
            )

        self = self.to("cpu")

        if return_raw:
            return predicted, expected, n_fragments

        result = xr.Dataset(
            {
                "predicted": xr.DataArray(
                    predicted,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={fragments.obs.index.name: cells, fragments.var.index.name: regions},
                ),
                "expected": xr.DataArray(
                    expected,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={fragments.obs.index.name: cells, fragments.var.index.name: regions},
                ),
                "n_fragments": xr.DataArray(
                    n_fragments,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={fragments.obs.index.name: cells, fragments.var.index.name: regions},
                ),
            }
        )
        return result

    def get_prediction_censored(
        self,
        censorer,
        fragments=None,
        transcriptome=None,
        cells=None,
        cell_ixs=None,
        regions=None,
        region_ixs=None,
        device=None,
    ):
        """
        Returns the prediction of multiple censored dataset
        """
        if fragments is None:
            fragments = self.fragments
        if transcriptome is None:
            transcriptome = self.transcriptome

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

        if device is None:
            device = get_default_device()

        minibatcher = Minibatcher(
            cell_ixs,
            region_ixs,
            n_regions_step=500,
            n_cells_step=5000,
            use_all_cells=True,
            use_all_regions=True,
            permute_cells=False,
            permute_regions=False,
        )
        loaders = LoaderPool(
            TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxregion_batch_size=minibatcher.cellxregion_batch_size,
                layer=self.layer,
            ),
            n_workers=10,
        )
        loaders.initialize(minibatcher)

        predicted = np.zeros((len(censorer), len(cell_ixs), len(region_ixs)), dtype=float)
        expected = np.zeros((len(cell_ixs), len(region_ixs)), dtype=float)
        n_fragments = np.zeros((len(censorer), len(cell_ixs), len(region_ixs)), dtype=int)

        cell_mapping = np.zeros(fragments.n_cells, dtype=np.int64)
        cell_mapping[cell_ixs] = np.arange(len(cell_ixs))
        region_mapping = np.zeros(fragments.n_regions, dtype=np.int64)
        region_mapping[region_ixs] = np.arange(len(region_ixs))

        self.eval()
        self.to(device)
        for data in loaders:
            data = data.to(device)
            fragments_oi = censorer(data)

            with torch.no_grad():
                for design_ix, (
                    pred_mb,
                    n_fragments_oi_mb,
                ) in enumerate(self.forward_multiple(data, fragments_oi)):
                    ix = np.ix_(
                        [design_ix],
                        cell_mapping[data.minibatch.cells_oi],
                        region_mapping[data.minibatch.regions_oi],
                    )
                    predicted[ix] = pred_mb.cpu().numpy()
                    n_fragments[ix] = n_fragments_oi_mb.cpu().numpy()
            expected[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    region_mapping[data.minibatch.regions_oi],
                )
            ] = data.transcriptome.value.cpu().numpy()

        self.to("cpu")

        return predicted, expected, n_fragments

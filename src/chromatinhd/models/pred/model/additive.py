"""
Additive model for predicting region expression from fragments
"""
from __future__ import annotations

import torch
import torch_scatter
import math
import numpy as np
import xarray as xr
import pandas as pd

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
from chromatinhd.models.pred.trainer import Trainer
from chromatinhd.loaders import LoaderPool
from chromatinhd.optim import SparseDenseAdam

from chromatinhd import get_default_device

from .loss import paircor, paircor_loss, region_paircor_loss

from typing import Any


class SineEncoding(torch.nn.Module):
    def __init__(self, n_frequencies):
        super().__init__()

        self.register_buffer(
            "frequencies",
            torch.tensor([[1 / 100 ** (2 * i / n_frequencies)] * 2 for i in range(1, n_frequencies + 1)]).flatten(-2),
        )

        self.register_buffer(
            "shifts",
            torch.tensor([[0, torch.pi / 2] for _ in range(1, n_frequencies + 1)]).flatten(-2),
        )

        self.n_embedding_dimensions = n_frequencies * 2 * 2

    def forward(self, coordinates):
        embedding = torch.sin((coordinates[..., None] * self.frequencies + self.shifts).flatten(-2))
        return embedding


class DirectEncoding(torch.nn.Sequential):
    """
    Dummy encoding of fragments, simply providing the positions directly
    """

    def __init__(self, window=(-10000, 10000)):
        self.n_embedding_dimensions = 3
        self.window = window
        super().__init__()

    def forward(self, coordinates):
        return torch.cat(
            [
                torch.ones((*coordinates.shape[:-1], 1), device=coordinates.device, dtype=torch.float),
                coordinates / (self.window[1] - self.window[0]) * 2,
            ],
            dim=-1,
        )


class FragmentEmbedder(torch.nn.Module):
    dropout_rate = 0.0

    def __init__(
        self,
        n_regions,
        n_frequencies=10,
        n_embedding_dimensions=5,
        nonlinear=True,
        dropout_rate=0.0,
        **kwargs,
    ):
        self.n_embedding_dimensions = n_embedding_dimensions

        self.nonlinear = nonlinear
        self.dropout_rate = dropout_rate

        super().__init__(**kwargs)

        if n_frequencies == "direct":
            self.sine_encoding = DirectEncoding()
        else:
            self.sine_encoding = SineEncoding(n_frequencies=n_frequencies)

        # default initialization same as a torch.nn.Linear
        self.bias1 = EmbeddingTensor(
            n_regions,
            (self.n_embedding_dimensions,),
            sparse=True,
        )
        self.bias1.data.zero_()

        self.weight1 = EmbeddingTensor(
            n_regions,
            (
                self.sine_encoding.n_embedding_dimensions,
                self.n_embedding_dimensions,
            ),
            sparse=True,
        )
        stdv = 1.0 / math.sqrt(self.weight1.shape[-1])  # / 100
        self.weight1.data.uniform_(-stdv, stdv)

    def forward(self, coordinates, region_ix):
        embedding = self.sine_encoding(coordinates)
        embedding = torch.einsum("ab,abc->ac", embedding, self.weight1(region_ix)) + self.bias1(region_ix)
        # embedding = (embedding[..., None] * self.weight1[region_ix]).sum(-2)

        # non-linear
        if self.nonlinear is True:
            embedding = torch.sigmoid(embedding)
        elif isinstance(self.nonlinear, str) and self.nonlinear == "relu":
            embedding = torch.relu(embedding)
        elif isinstance(self.nonlinear, str) and self.nonlinear == "elu":
            embedding = torch.nn.functional.elu(embedding)

        if (self.dropout_rate > 0) and self.training:
            embedding = torch.nn.functional.dropout(embedding, self.dropout_rate)

        return embedding

    def parameters_sparse(self):
        return [self.bias1.weight, self.weight1.weight]


class FragmentEmbedderCounter(torch.nn.Sequential):
    """
    Dummy embedding of fragments in a single embedding dimension of ones
    """

    def __init__(self, *args, **kwargs):
        self.n_embedding_dimensions = 1
        super().__init__(*args, **kwargs)

    def forward(self, coordinates, region_ix):
        return torch.ones((*coordinates.shape[:-1], 1), device=coordinates.device, dtype=torch.float)


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

    def __init__(self, n_regions, n_embedding_dimensions=5, initialization="default", **kwargs):
        self.n_regions = n_regions
        self.n_embedding_dimensions = n_embedding_dimensions

        super().__init__()

        # set bias to empirical mean
        self.bias1 = EmbeddingTensor(
            n_regions,
            tuple(),
            sparse=True,
        )
        self.bias1.data[:] = 0.0

        self.weight1 = EmbeddingTensor(
            n_regions,
            (n_embedding_dimensions,),
            sparse=True,
        )
        if initialization == "ones":
            self.weight1.data[:] = 1.0
        elif initialization == "default":
            self.weight1.data[:, :5] = 1.0
            self.weight1.data[:, 5:] = 0.0
            self.weight1.data[:, -1] = -1.0
        elif initialization == "smaller":
            stdv = 1.0 / math.sqrt(self.weight1.data.size(-1)) / 100
            self.weight1.data.uniform_(-stdv, stdv)

    def forward(self, cell_region_embedding, region_ix):
        out = (cell_region_embedding * self.weight1(region_ix)).sum(-1) + self.bias1(region_ix).squeeze()
        return out

    def parameters_sparse(self):
        return [self.bias1.weight, self.weight1.weight]


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
        dropout_rate: float = 0.0,
        embedding_to_expression_initialization: str = "default",
        layer=None,
        reset=False,
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

        if not self.o.state.exists(self):
            assert fragments is not None
            assert transcriptome is not None

            n_regions = fragments.n_regions

            if dummy is True:
                self.fragment_embedder = FragmentEmbedderCounter()
            else:
                self.fragment_embedder = FragmentEmbedder(
                    n_frequencies=n_frequencies,
                    n_regions=n_regions,
                    nonlinear=nonlinear,
                    n_embedding_dimensions=n_embedding_dimensions,
                    dropout_rate=dropout_rate,
                )
            self.embedding_region_pooler = EmbeddingGenePooler(reduce=reduce)
            self.embedding_to_expression = EmbeddingToExpression(
                n_regions=n_regions,
                n_embedding_dimensions=self.fragment_embedder.n_embedding_dimensions,
                initialization=embedding_to_expression_initialization,
            )

            n_regions = fragments.n_regions

            if dummy is True:
                self.fragment_embedder = FragmentEmbedderCounter()
            else:
                self.fragment_embedder = FragmentEmbedder(
                    n_frequencies=n_frequencies,
                    n_regions=n_regions,
                    nonlinear=nonlinear,
                    n_embedding_dimensions=n_embedding_dimensions,
                    dropout_rate=dropout_rate,
                )
            self.embedding_region_pooler = EmbeddingGenePooler(reduce=reduce)
            self.embedding_to_expression = EmbeddingToExpression(
                n_regions=n_regions,
                n_embedding_dimensions=self.fragment_embedder.n_embedding_dimensions,
                initialization=embedding_to_expression_initialization,
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

    def forward_region_loss(self, data):
        """
        Make a prediction and calculate the loss given a data object
        """
        expression_predicted = self.forward(data)
        expression_true = data.transcriptome.value
        return region_paircor_loss(expression_predicted, expression_true)

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
        lr=1e-3,
        n_epochs=60,
        pbar=True,
        n_regions_step=500,
        n_cells_step=200,
        weight_decay=1e-5,
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

        # set up minibatchers and loaders
        minibatcher_train = Minibatcher(
            fold["cells_train"],
            range(fragments.n_regions),
            n_regions_step=n_regions_step,
            n_cells_step=n_cells_step,
        )
        minibatcher_validation = Minibatcher(
            fold["cells_validation"],
            range(fragments.n_regions),
            n_regions_step=10,
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
            predicted[np.ix_(cell_mapping[data.minibatch.cells_oi], data.minibatch.regions_oi)] = pred_mb.cpu().numpy()
            expected[
                np.ix_(cell_mapping[data.minibatch.cells_oi], data.minibatch.regions_oi)
            ] = data.transcriptome.value.cpu().numpy()
            n_fragments[np.ix_(cell_mapping[data.minibatch.cells_oi], data.minibatch.regions_oi)] = (
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


class Models(Flow):
    models = StoredDict(Stored)
    n_models = Stored()

    @property
    def models_path(self):
        path = self.path / "models"
        path.mkdir(exist_ok=True)
        return path

    def train_models(self, fragments, transcriptome, folds, device=None):
        self.n_models = len(folds)
        for fold_ix, fold in [(fold_ix, fold) for fold_ix, fold in enumerate(folds)]:
            desired_outputs = [self.models_path / ("model_" + str(fold_ix) + ".pkl")]
            force = False
            if not all([desired_output.exists() for desired_output in desired_outputs]):
                force = True

            if force:
                model = Model(
                    n_regions=fragments.n_regions,
                )
                model.train_model(fragments, transcriptome, fold, device=device)

                model = model.to("cpu")

                self.models[fold_ix] = model

    def __getitem__(self, ix):
        return self.models[ix]

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        for ix in range(len(self)):
            yield self[ix]

    def get_region_cors(self, fragments, transcriptome, folds, device=None):
        cor_predicted = np.zeros((len(fragments.var.index), len(folds)))
        cor_n_fragments = np.zeros((len(fragments.var.index), len(folds)))
        n_fragments = np.zeros((len(fragments.var.index), len(folds)))

        if device is None:
            device = get_default_device()
        for model_ix, (model, fold) in enumerate(zip(self, folds)):
            prediction = model.get_prediction(fragments, transcriptome, cell_ixs=fold["cells_test"], device=device)

            cor_predicted[:, model_ix] = paircor(prediction["predicted"].values, prediction["expected"].values)
            cor_n_fragments[:, model_ix] = paircor(prediction["n_fragments"].values, prediction["expected"].values)

            n_fragments[:, model_ix] = prediction["n_fragments"].values.sum(0)
        cor_predicted = pd.Series(cor_predicted.mean(1), index=fragments.var.index, name="cor_predicted")
        cor_n_fragments = pd.Series(cor_n_fragments.mean(1), index=fragments.var.index, name="cor_n_fragments")
        n_fragments = pd.Series(n_fragments.mean(1), index=fragments.var.index, name="n_fragments")
        result = pd.concat([cor_predicted, cor_n_fragments, n_fragments], axis=1)
        result["deltacor"] = result["cor_predicted"] - result["cor_n_fragments"]

        return result

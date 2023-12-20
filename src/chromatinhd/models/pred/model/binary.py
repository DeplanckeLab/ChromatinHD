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
import itertools
import tqdm.auto as tqdm

import pickle

from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.models import FlowModel
from chromatinhd.flow import Flow, Stored, LinkedDict, Linked

from chromatinhd.loaders.minibatches import Minibatcher
from chromatinhd.loaders.transcriptome_fragments2 import (
    TranscriptomeFragments,
)
from chromatinhd.data.fragments import Fragments
from chromatinhd.data.transcriptome import Transcriptome
from chromatinhd.models.pred.trainer import Trainer, TrainerPerFeature
from chromatinhd.loaders import LoaderPool
from chromatinhd.optim import SparseDenseAdam
from chromatinhd.utils import crossing
from torch.optim import Adam, RAdam

from chromatinhd.optim import AdamPerFeature, SGDPerFeature

from chromatinhd import get_default_device

from .loss import paircor, paircor_loss, region_paircor_loss, pairzmse_loss, region_pairzmse_loss
from .multilinear import MultiLinear

from typing import Any

import time


class catchtime(object):
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.t = time.time() - self.t
        # print(self.name, self.t)


from .encoders import (
    SineEncoding,
    SineEncoding2,
    SineEncoding3,
    RadialEncoding,
    TophatEncoding,
    DirectEncoding,
    ExponentialEncoding,
    RadialBinaryEncoding,
    RadialBinaryCenterEncoding,
    LinearBinaryEncoding,
    SplineBinaryEncoding,
    DirectDistanceEncoding,
    SplitDistanceEncoding,
    LinearDistanceEncoding,
    MultiSplineBinaryEncoding,
)


class FragmentEmbedder(torch.nn.Module):
    dropout_rate = 0.0
    attentions = None

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
        layernorm=False,
        encoder="sine2",
        fragments=None,
        distance_encoder=None,
        encoder_kwargs=None,
        attention=False,
        **kwargs,
    ):
        self.nonlinear = nonlinear
        self.residual = residual

        super().__init__(**kwargs)

        # make encoding

        if encoder is None:
            encoder = "spline_binary"

        if encoder_kwargs is None:
            encoder_kwargs = {}
        if encoder == "sine2":
            self.encoder = SineEncoding2(n_frequencies=n_frequencies)
        elif encoder == "sine":
            self.encoder = SineEncoding(n_frequencies=n_frequencies)
        elif encoder == "sine3":
            self.encoder = SineEncoding3(n_frequencies=n_frequencies)
        elif encoder == "radial":
            self.encoder = RadialEncoding(n_frequencies=n_frequencies, window=fragments.regions.window)
        elif encoder == "tophat":
            self.encoder = TophatEncoding(n_frequencies=n_frequencies, window=fragments.regions.window)
        elif encoder == "direct":
            self.encoder = DirectEncoding(window=fragments.regions.window)
        elif encoder == "exponential":
            self.encoder = ExponentialEncoding(window=fragments.regions.window)
        elif encoder == "radial_binary":
            self.encoder = RadialBinaryEncoding(
                n_frequencies=n_frequencies, window=fragments.regions.window, **encoder_kwargs
            )
        elif encoder == "radial_binary_center":
            self.encoder = RadialBinaryCenterEncoding(
                n_frequencies=n_frequencies, window=fragments.regions.window, **encoder_kwargs
            )
        elif encoder == "linear_binary":
            self.encoder = LinearBinaryEncoding(
                n_frequencies=n_frequencies, window=fragments.regions.window, **encoder_kwargs
            )
        elif encoder == "spline_binary":
            self.encoder = SplineBinaryEncoding(window=fragments.regions.window, **encoder_kwargs)
        elif encoder == "multi_spline_binary":
            self.encoder = MultiSplineBinaryEncoding(
                fragments.n_regions, window=fragments.regions.window, **encoder_kwargs
            )
        elif isinstance(encoder, str):
            raise ValueError(encoder + " is not a valid encoder")
        elif encoder is None:
            raise ValueError(encoder + " is not a valid encoder")
        else:
            self.encoder = encoder(n_frequencies=n_frequencies)

        n_input_embedding_dimensions = self.encoder.n_embedding_dimensions

        # make distance encoding
        if distance_encoder is not None:
            if distance_encoder == "direct":
                self.distance_encoder = DirectDistanceEncoding()
            elif distance_encoder == "split":
                self.distance_encoder = SplitDistanceEncoding()
            elif distance_encoder == "linear":
                self.distance_encoder = LinearDistanceEncoding()
            else:
                raise ValueError(distance_encoder + " is not a valid distance encoder")
            n_input_embedding_dimensions += self.distance_encoder.n_embedding_dimensions
        else:
            self.distance_encoder = None

        n_output_embedding_dimensions = self.n_embedding_dimensions = (
            n_embedding_dimensions if n_layers > 0 else n_input_embedding_dimensions
        )

        # make layers
        self.layers = []
        self.attentions = []

        for i in range(n_layers):
            sublayers = []
            sublayers.append(
                MultiLinear(
                    n_output_embedding_dimensions if i > 0 else n_input_embedding_dimensions,
                    n_output_embedding_dimensions,
                    n_regions,
                )
            )
            if (i > 0) and (batchnorm):
                sublayers.append(torch.nn.BatchNorm1d(n_output_embedding_dimensions))
            if (i > 0) and (layernorm):
                sublayers.append(torch.nn.LayerNorm((n_output_embedding_dimensions,)))
            if nonlinear is not False:
                if nonlinear is True:
                    sublayers.append(torch.nn.GELU())
                elif nonlinear == "relu":
                    sublayers.append(torch.nn.ReLU())
                elif nonlinear == "silu":
                    sublayers.append(torch.nn.SiLU())
                else:
                    sublayers.append(nonlinear())
            if self.dropout_rate > 0:
                sublayers.append(torch.nn.Dropout(p=dropout_rate))
            nn = torch.nn.Sequential(*sublayers)
            self.add_module("nn{}".format(i), nn)
            self.layers.append(nn)

            if (attention is not False) and (i > 0):
                attn = torch.nn.MultiheadAttention(n_output_embedding_dimensions, 1, bias=False)

                if isinstance(attention, dict):
                    if "initialization" in attention:
                        if attention["initialization"] == "eye":
                            ndim = n_output_embedding_dimensions
                            attn.in_proj_weight.data[0 * ndim : 1 * ndim] = torch.eye(ndim)
                            attn.in_proj_weight.data[1 * ndim : 2 * ndim] = torch.eye(ndim)
                            attn.in_proj_weight.data[2 * ndim : 3 * ndim] = torch.eye(ndim)
                            attn.out_proj.weight.data[:] = torch.eye(ndim)

                self.attentions.append(attn)

                self.add_module("attention{}".format(i), attn)
            else:
                self.attentions.append(None)

    def forward(self, data):
        if self.attentions is None:
            self.attentions = [None] * len(self.layers)

        with catchtime("embed") as t:
            embedding = self.encoder(
                data.fragments.coordinates, data.fragments.region_indptr, data.minibatch.regions_oi_torch
            )
            # embedding = self.encoder(data.fragments.coordinates)
            if self.distance_encoder is not None:
                embedding = torch.cat([embedding, self.distance_encoder(data.fragments.coordinates)], dim=-1)

        with catchtime(f"layers") as t:
            for i, (layer, attention) in enumerate(zip(self.layers, self.attentions)):
                with catchtime(f"layer_{i}") as t:
                    if attention is not None:
                        if len(data.fragments.doublet_idx) > 0:
                            embedding_doublet = embedding[data.fragments.doublet_idx]
                            embedding_doublet_reshaped = embedding_doublet.reshape(
                                len(data.fragments.doublet_idx) // 2, 2, -1
                            )
                            output = attention(
                                embedding_doublet_reshaped,
                                embedding_doublet_reshaped,
                                embedding_doublet_reshaped,
                                need_weights=False,
                            )[0].reshape(-1, embedding_doublet.shape[-1])
                            embedding[data.fragments.doublet_idx] = embedding[data.fragments.doublet_idx] + output

                    embedding2 = embedding
                    # print(embedding2[:, 0])
                    for i, sublayer in enumerate(layer):
                        if i == 0:
                            embedding2 = sublayer(
                                embedding2, data.fragments.region_indptr, data.minibatch.regions_oi_torch
                            )
                        else:
                            embedding2 = sublayer(embedding2)
                    if (self.residual) and (i > 0):
                        embedding = embedding + embedding2
                    else:
                        embedding = embedding2

        return embedding


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

    pooler = None

    def __init__(self, n_embedding_dimensions, reduce="sum", pooler=None):
        if pooler is None:
            self.reduce = reduce
            self.pooler = None
        elif pooler == "attention":
            from . import pooling

            self.pooler = pooling.SelfAttentionPooling(n_embedding_dimensions)
        super().__init__()

    def forward(self, embedding, fragment_regionxcell_ix, cell_n, region_n):
        if self.pooler is not None:
            regionxcell_embedding = self.pooler(embedding)
        if self.reduce == "mean":
            regionxcell_embedding = torch_scatter.segment_mean_coo(
                embedding, fragment_regionxcell_ix, dim_size=cell_n * region_n
            )
        elif self.reduce == "sum":
            regionxcell_embedding = torch_scatter.segment_sum_coo(
                embedding, fragment_regionxcell_ix, dim_size=cell_n * region_n
            )
        else:
            raise ValueError()
        region_cell_embedding = regionxcell_embedding.reshape((region_n, cell_n, regionxcell_embedding.shape[-1]))
        return region_cell_embedding


class LibrarySizeEncoder(torch.nn.Module):
    def __init__(self, fragments, n_layers=1):
        super().__init__()

        library_size = np.bincount(fragments.mapping[:, 0], minlength=fragments.n_cells)
        self.register_buffer(
            "differential_library_size",
            torch.from_numpy((library_size - library_size.mean()) / library_size.std()).float(),
        )

        self.layers = []
        for i in range(n_layers):
            sublayers = []
            sublayers.append(torch.nn.Linear(1, 1))
            # sublayers.append(torch.nn.ReLU())
            nn = torch.nn.Sequential(*sublayers)
            self.add_module("nn{}".format(i), nn)
            self.layers.append(nn)

        self.n_embedding_dimensions = 1

    def forward(self, data):
        return self.layers[0](self.differential_library_size[data.minibatch.cells_oi].reshape(-1, 1))


class EmbeddingToExpression(torch.nn.Module):
    """
    Predicts region expression using a [cell, region, component] embedding in a region-specific manner
    """

    def __init__(
        self,
        fragments,
        n_input_embedding_dimensions,
        n_embedding_dimensions=5,
        initialization="default",
        n_layers=1,
        dropout_rate=0.0,
        nonlinear=True,
        residual=False,
        batchnorm=False,
        **kwargs,
    ):
        self.n_input_embedding_dimensions = n_input_embedding_dimensions
        self.n_embedding_dimensions = n_embedding_dimensions
        self.residual = residual

        super().__init__()

        # make layers
        self.layers = []

        for i in range(n_layers):
            sublayers = []
            sublayers.append(
                MultiLinear(
                    n_input_embedding_dimensions if i == 0 else n_embedding_dimensions,
                    self.n_embedding_dimensions,
                    fragments.n_regions,
                )
            )
            if batchnorm:
                sublayers.append(torch.nn.BatchNorm1d(n_embedding_dimensions))
            if nonlinear is True:
                sublayers.append(torch.nn.GELU())
            elif nonlinear == "relu":
                sublayers.append(torch.nn.ReLU())
            elif nonlinear == "silu":
                sublayers.append(torch.nn.SiLU())
            else:
                sublayers.append(nonlinear())

            if dropout_rate > 0:
                sublayers.append(torch.nn.Dropout(p=dropout_rate))
            nn = torch.nn.Sequential(*sublayers)
            self.add_module("nn{}".format(i), nn)
            self.layers.append(nn)

        final = MultiLinear(
            self.n_embedding_dimensions, 1, fragments.n_regions, bias=False, weight_constructor=torch.zeros
        )
        self.final = final

    def forward(self, region_cell_embedding, regions_oi):
        embedding = region_cell_embedding.reshape(-1, self.n_input_embedding_dimensions)

        region_indptr = torch.arange(
            0,
            region_cell_embedding.shape[1] * region_cell_embedding.shape[0] + 1,
            region_cell_embedding.shape[1],
            device=region_cell_embedding.device,
        )

        for i, layer in enumerate(self.layers):
            embedding2 = embedding
            for i, sublayer in enumerate(layer):
                if i == 0:
                    embedding2 = sublayer(embedding2, region_indptr, regions_oi)
                else:
                    embedding2 = sublayer(embedding2)
            if (self.residual) and (i < len(self.layers) - 1):
                embedding = embedding + embedding2
            else:
                embedding = embedding2
        embedding = self.final(embedding, region_indptr, regions_oi)

        return embedding.reshape(region_cell_embedding.shape[:-1]).T


class Model(FlowModel):
    """
    Predicting region expression from raw fragments using an additive model across fragments from the same cell

    Parameters:
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

    region_oi = Stored()

    def __init__(
        self,
        path=None,
        fragments: Fragments | None = None,
        transcriptome: Transcriptome | None = None,
        fold=None,
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
        pooler=None,
        distance_encoder=None,
        library_size_encoder=None,
        region_oi=None,
        encoder_kwargs=None,
        fragment_embedder_kwargs=None,
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

        if not self.o.region_oi.exists(self):
            self.region_oi = region_oi

        if not self.o.state.exists(self):
            if fragment_embedder_kwargs is None:
                fragment_embedder_kwargs = {}

            self.fragment_embedder = FragmentEmbedder(
                fragments.n_regions,
                n_frequencies=n_frequencies,
                nonlinear=nonlinear,
                n_layers=n_layers_fragment_embedder,
                n_embedding_dimensions=n_embedding_dimensions,
                dropout_rate=dropout_rate_fragment_embedder,
                residual=residual_fragment_embedder,
                batchnorm=batchnorm_fragment_embedder,
                fragments=self.fragments,
                encoder=encoder,
                distance_encoder=distance_encoder,
                encoder_kwargs=encoder_kwargs,
                **fragment_embedder_kwargs,
            )
            self.embedding_region_pooler = EmbeddingGenePooler(
                self.fragment_embedder.n_embedding_dimensions, reduce=reduce, pooler=pooler
            )

            n_input_embedding_dimensions = self.fragment_embedder.n_embedding_dimensions
            if library_size_encoder == "linear":
                self.library_size_encoder = LibrarySizeEncoder(fragments)
                n_input_embedding_dimensions += self.library_size_encoder.n_embedding_dimensions

            self.embedding_to_expression = EmbeddingToExpression(
                fragments=fragments,
                n_input_embedding_dimensions=n_input_embedding_dimensions,
                n_embedding_dimensions=self.fragment_embedder.n_embedding_dimensions,
                initialization=embedding_to_expression_initialization,
                n_layers=n_layers_embedding2expression,
                residual=residual_embedding2expression,
                dropout_rate=dropout_rate_embedding2expression,
                batchnorm=batchnorm_embedding2expression,
                nonlinear=nonlinear,
            )

    def forward(self, data):
        """
        Make a prediction given a data object
        """

        with catchtime("all") as t:
            with catchtime("fagment_embedding") as t:
                fragment_embedding = self.fragment_embedder(data)

            with catchtime("region_cell_embedding") as t:
                region_cell_embedding = self.embedding_region_pooler(
                    fragment_embedding,
                    data.fragments.local_regionxcell_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_regions,
                )
            if hasattr(self, "library_size_encoder"):
                region_cell_embedding = torch.cat(
                    [
                        region_cell_embedding,
                        self.library_size_encoder(data).unsqueeze(0).expand(*region_cell_embedding.shape[:-1], 1),
                    ],
                    dim=-1,
                )

            with catchtime("expression_prediction") as t:
                expression_predicted = self.embedding_to_expression(
                    region_cell_embedding, data.minibatch.regions_oi_torch
                )

        self.expression_predicted = expression_predicted
        return expression_predicted

    def forward_loss(self, data):
        """
        Make a prediction and calculate the loss given a data object
        """
        expression_predicted = self.forward(data)
        expression_true = data.transcriptome.value
        # return -paircor(expression_predicted, expression_true).sum() * 100
        return -paircor(expression_predicted, expression_true) * 100
        # return paircor_loss(expression_predicted, expression_true)

    # def forward_backward(self, data):
    #     expression_predicted = self.forward(data)
    #     expression_true = data.transcriptome.value
    #     loss = -paircor(expression_predicted, expression_true) * 100
    #     loss.backward()
    #     return loss

    def forward_backward(self, data):
        expression_predicted = self.forward(data)
        for i in range(data.minibatch.n_regions):
            # for i in [0]:
            loss = -paircor(expression_predicted[:, [i]], data.transcriptome.value[:, [i]])[0] * 100
            loss.backward(retain_graph=True)
        return loss

    def forward_region_loss(self, data):
        """
        Make a prediction and calculate the loss given a data object
        """
        expression_predicted = self.forward(data)
        expression_true = data.transcriptome.value
        return region_paircor_loss(expression_predicted, expression_true)

    def forward_multiple(self, data, fragments_oi, min_fragments=1):
        fragment_embedding = self.fragment_embedder(data)

        total_n_fragments = torch.bincount(
            data.fragments.local_regionxcell_ix,
            minlength=data.minibatch.n_regions * data.minibatch.n_cells,
        ).reshape((data.minibatch.n_cells, data.minibatch.n_regions))

        total_region_cell_embedding = self.embedding_region_pooler.forward(
            fragment_embedding,
            data.fragments.local_regionxcell_ix,
            data.minibatch.n_cells,
            data.minibatch.n_regions,
        )
        region_cell_embedding = total_region_cell_embedding

        if hasattr(self, "library_size_encoder"):
            region_cell_embedding = torch.cat(
                [region_cell_embedding, self.library_size_encoder(data).unsqueeze(-2)], dim=-1
            )

        total_expression_predicted = self.embedding_to_expression.forward(
            region_cell_embedding, data.minibatch.regions_oi_torch
        )

        for fragments_oi_ in fragments_oi:
            if (fragments_oi_ is not None) and ((~fragments_oi_).sum() > min_fragments):
                lost_fragments_oi = ~fragments_oi_
                lost_local_regionxcell_ix = data.fragments.local_regionxcell_ix[lost_fragments_oi]
                n_fragments = total_n_fragments - torch.bincount(
                    lost_local_regionxcell_ix,
                    minlength=data.minibatch.n_regions * data.minibatch.n_cells,
                ).reshape((data.minibatch.n_cells, data.minibatch.n_regions))
                region_cell_embedding = total_region_cell_embedding - self.embedding_region_pooler.forward(
                    fragment_embedding[lost_fragments_oi],
                    lost_local_regionxcell_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_regions,
                )

                if hasattr(self, "library_size_encoder"):
                    region_cell_embedding = torch.cat(
                        [region_cell_embedding, self.library_size_encoder(data).unsqueeze(-2)], dim=-1
                    )

                expression_predicted = self.embedding_to_expression.forward(region_cell_embedding)
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
        n_regions_step=5,
        n_cells_step=20000,
        weight_decay=1e-2,
        checkpoint_every_epoch=1,
        optimizer="adam",
        **kwargs,
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

        regions_oi = fragments.var.index

        region_cuts = [*np.arange(0, len(regions_oi), step=250), len(regions_oi)]
        region_bins = [regions_oi[a:b] for a, b in zip(region_cuts[:-1], region_cuts[1:])]

        for regions_oi in region_bins:
            region_ixs = fragments.var.index.get_indexer(regions_oi)

            minibatcher_train = Minibatcher(
                fold["cells_train"],
                region_ixs,
                n_regions_step=n_regions_step,
                n_cells_step=n_cells_step,
                permute_cells=False,
                permute_regions=False,
            )
            minibatcher_validation = Minibatcher(
                fold["cells_validation"],
                region_ixs,
                n_regions_step=n_regions_step * 2,
                n_cells_step=n_cells_step // 2,
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
                    regionxcell_batch_size=minibatcher_train.cellxregion_batch_size,
                    layer=self.layer,
                ),
                n_workers=20,
            )
            loaders_validation = LoaderPool(
                TranscriptomeFragments,
                dict(
                    transcriptome=transcriptome,
                    fragments=fragments,
                    regionxcell_batch_size=minibatcher_validation.cellxregion_batch_size,
                    layer=self.layer,
                ),
                n_workers=10,
            )

            def recurse_for_feature_parameters(module, visited=set(), params=set()):
                for submodule in module.children():
                    if submodule.__class__.__name__ in "FeatureParameter":
                        params.add(submodule)
                    else:
                        visited.add(submodule)
                        recurse_for_feature_parameters(submodule, visited=visited, params=params)
                return params

            parameters = list(recurse_for_feature_parameters(self))

            if optimizer == "sgd":
                optim = SGDPerFeature(
                    parameters,
                    len(region_ixs),
                    lr=lr,
                    weight_decay=weight_decay,
                )
            elif optimizer == "adam":
                optim = AdamPerFeature(
                    parameters,
                    len(region_ixs),
                    lr=lr,
                    weight_decay=weight_decay,
                )
            else:
                raise ValueError(optimizer)

            trainer = TrainerPerFeature(
                self,
                loaders_train,
                loaders_validation,
                minibatcher_train,
                minibatcher_validation,
                optim=optim,
                n_epochs=n_epochs,
                checkpoint_every_epoch=checkpoint_every_epoch,
                optimize_every_step=1,
                device=device,
                pbar=pbar,
                **kwargs,
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
            cell_ixs = fragments.obs.index.get_indexer(cells)
        if cells is None:
            cells = fragments.obs.index[cell_ixs]
        if region_ixs is None:
            if regions is None:
                regions = fragments.var.index
            region_ixs = fragments.var.index.get_indexer(regions)
        if regions is None:
            regions = fragments.var.index[region_ixs]

        if device is None:
            device = get_default_device()

        minibatches = Minibatcher(
            cell_ixs,
            region_ixs,
            n_regions_step=5,
            n_cells_step=20000,
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
                regionxcell_batch_size=minibatches.cellxregion_batch_size,
                layer=self.layer,
                region_oi=self.region_oi,
            ),
            n_workers=10,
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
                    data.fragments.local_regionxcell_ix,
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
        min_fragments=10,
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

        region_ixs = [fragments.var.index.get_loc(self.region_oi)]

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
                regionxcell_batch_size=minibatcher.regionxcell_batch_size,
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
                ) in enumerate(self.forward_multiple(data, fragments_oi, min_fragments=min_fragments)):
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
    models = LinkedDict()

    transcriptome = Linked()
    """The transcriptome"""

    fragments = Linked()
    """The fragments"""

    folds = Linked()
    """The folds"""

    model_params = Stored(default=dict)
    train_params = Stored(default=dict)

    regions_oi = Stored(default=lambda: None)

    @property
    def models_path(self):
        path = self.path / "models"
        path.mkdir(exist_ok=True)
        return path

    def train_models(self, device=None, pbar=True, **kwargs):
        fragments = self.fragments
        transcriptome = self.transcriptome
        folds = self.folds

        progress = itertools.product(enumerate(folds))
        progress = tqdm.tqdm(progress, total=len(folds)) if pbar else progress

        for ((fold_ix, fold),) in progress:
            model_name = f"{fold_ix}"
            model_folder = self.models_path / (model_name)
            force = False
            if model_name not in self.models:
                force = True
            elif not self.models[model_name].o.state.exists(self.models[model_name]):
                force = True

            if force:
                model = Model(
                    fragments=fragments,
                    transcriptome=transcriptome,
                    fold=fold,
                    path=model_folder,
                    **self.model_params,
                )
                model.train_model(device=device, pbar=True, **self.train_params)
                model.save_state()

                model = model.to("cpu")

                self.models[model_name] = model

    def __contains__(self, ix):
        return ix in self.models

    def __getitem__(self, ix):
        return self.models[ix]

    def __setitem__(self, ix, value):
        self.models[ix] = value

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        for ix in range(len(self)):
            yield self[ix]

    @property
    def design(self):
        design_dimensions = {
            "fold": range(len(self.folds)),
        }
        design = crossing(**design_dimensions)
        design.index = design["fold"].astype(str)
        return design

    def fitted(self, region, fold_ix):
        return f"{fold_ix}" in self.models

    def get_prediction(self, fold_ix, **kwargs):
        model = self[f"{fold_ix}"]
        return model.get_prediction(**kwargs)

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
import tqdm.auto as tqdm
import time
import os

import pickle

from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.models import FlowModel
from chromatinhd.flow import Flow, Stored, LinkedDict, Linked

from chromatinhd.loaders.minibatches import Minibatcher
from chromatinhd.loaders.transcriptome_fragments import (
    TranscriptomeFragments,
)
from chromatinhd.data.fragments import Fragments
from chromatinhd.data.transcriptome import Transcriptome
from chromatinhd.models.pred.trainer import SharedTrainer
from chromatinhd.loaders import LoaderPool
from chromatinhd.optim import SparseDenseAdam
from chromatinhd.utils import crossing
from torch.optim import Adam, AdamW, RAdam, Adamax

from chromatinhd import get_default_device

from .loss import (
    paircor,
    paircor_loss,
    region_paircor_loss,
    pairzmse_loss,
    region_pairzmse_loss,
)

from typing import Any

from .encoders import (
    SineEncoding,
    SineEncoding2,
    SineEncoding3,
    RadialEncoding,
    TophatEncoding,
    DirectEncoding,
    ExponentialEncoding,
    RadialBinaryEncoding,
    RadialBinaryEncoding2,
    RadialBinaryCenterEncoding,
    LinearBinaryEncoding,
    SplineBinaryEncoding,
    DirectDistanceEncoding,
    SplitDistanceEncoding,
    LinearDistanceEncoding,
    TophatBinaryEncoding,
    OneEncoding,
)


class FragmentEmbedder(torch.nn.Module):
    dropout_rate = 0.0
    attentions = None

    def __init__(
        self,
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

        # make positional encoding
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
                n_frequencies=n_frequencies,
                window=fragments.regions.window,
                **encoder_kwargs,
            )
        elif encoder == "radial_binary2":
            self.encoder = RadialBinaryEncoding2(window=fragments.regions.window, **encoder_kwargs)
        elif encoder == "radial_binary_center":
            self.encoder = RadialBinaryCenterEncoding(
                n_frequencies=n_frequencies,
                window=fragments.regions.window,
                **encoder_kwargs,
            )
        elif encoder == "linear_binary":
            self.encoder = LinearBinaryEncoding(
                n_frequencies=n_frequencies,
                window=fragments.regions.window,
                **encoder_kwargs,
            )
        elif encoder == "spline_binary":
            self.encoder = SplineBinaryEncoding(window=fragments.regions.window, **encoder_kwargs)
        elif encoder == "tophat_binary":
            self.encoder = TophatBinaryEncoding(
                n_frequencies=n_frequencies,
                window=fragments.regions.window,
                **encoder_kwargs,
            )
        elif encoder == "nothing":
            self.encoder = OneEncoding()
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
                torch.nn.Linear(
                    n_output_embedding_dimensions if i > 0 else n_input_embedding_dimensions,
                    n_output_embedding_dimensions,
                )
            )
            if (i > 0) and (batchnorm):
                sublayers.append(torch.nn.BatchNorm1d(n_output_embedding_dimensions))
            if (i > 0) and (layernorm):
                sublayers.append(torch.nn.LayerNorm((n_output_embedding_dimensions,)))
            if nonlinear is not False:
                if (nonlinear is True) or nonlinear == "gelu":
                    sublayers.append(torch.nn.GELU())
                elif nonlinear == "relu":
                    sublayers.append(torch.nn.ReLU())
                elif nonlinear == "silu":
                    sublayers.append(torch.nn.SiLU())
                elif nonlinear == "tanh":
                    sublayers.append(torch.nn.Tanh())
                elif nonlinear == "sigmoid":
                    sublayers.append(torch.nn.Sigmoid())
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

        embedding = self.encoder(data.fragments.coordinates)
        if self.distance_encoder is not None:
            embedding = torch.cat([embedding, self.distance_encoder(data.fragments.coordinates)], dim=-1)

        for i, (layer, attention) in enumerate(zip(self.layers, self.attentions)):
            if attention is not None:
                if len(data.fragments.doublet_idx) > 0:
                    embedding_doublet = embedding[data.fragments.doublet_idx]
                    embedding_doublet_reshaped = embedding_doublet.reshape(len(data.fragments.doublet_idx) // 2, 2, -1)
                    output = attention(
                        embedding_doublet_reshaped,
                        embedding_doublet_reshaped,
                        embedding_doublet_reshaped,
                        need_weights=False,
                    )[0].reshape(-1, embedding_doublet.shape[-1])
                    embedding[data.fragments.doublet_idx] = embedding[data.fragments.doublet_idx] + output
            embedding2 = layer(embedding)
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

    def forward(self, embedding, fragment_cellxregion_ix, cell_n, region_n):
        if self.pooler is not None:
            cellxregion_embedding = self.pooler(embedding)
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


class LibrarySizeEncoder(torch.nn.Module):
    """
    Encodes library size as a linear transformation
    """

    def __init__(self, fragments, n_layers=1, scale=1.0):
        super().__init__()

        self.scale = scale
        self.n_embedding_dimensions = 1

    def forward(self, data):
        return data.fragments.libsize.reshape(-1, 1) * self.scale


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
        layernorm=False,
        **kwargs,
    ):
        self.n_input_embedding_dimensions = n_input_embedding_dimensions
        self.n_embedding_dimensions = n_input_embedding_dimensions
        self.residual = residual

        super().__init__()

        # make layers
        self.layers = []

        for i in range(n_layers):
            sublayers = []
            sublayers.append(
                torch.nn.Linear(
                    n_input_embedding_dimensions if i == 0 else self.n_embedding_dimensions,
                    self.n_embedding_dimensions,
                )
            )

            if (nonlinear is True) or nonlinear == "gelu":
                sublayers.append(torch.nn.GELU())
            elif nonlinear == "relu":
                sublayers.append(torch.nn.ReLU())
            elif nonlinear == "silu":
                sublayers.append(torch.nn.SiLU())
            elif nonlinear == "sigmoid":
                sublayers.append(torch.nn.Sigmoid())
            elif nonlinear == "tanh":
                sublayers.append(torch.nn.Tanh())
            else:
                sublayers.append(nonlinear())
            if i > 0 and batchnorm:
                sublayers.append(
                    torch.nn.BatchNorm1d(
                        self.n_embedding_dimensions,
                        affine=False,
                        track_running_stats=False,
                    )
                )
            if i > 0 and layernorm:
                sublayers.append(torch.nn.LayerNorm((self.n_embedding_dimensions,)))

            if dropout_rate > 0:
                sublayers.append(torch.nn.Dropout(p=dropout_rate))
            nn = torch.nn.Sequential(*sublayers)
            self.add_module("nn{}".format(i), nn)
            self.layers.append(nn)

        nn = torch.nn.Linear(self.n_embedding_dimensions, 1, bias=False)
        self.final = nn
        nn.weight.data[:] = 0
        self.layers.append(nn)

    def forward(self, cell_region_embedding):
        embedding = cell_region_embedding.reshape(-1, self.n_input_embedding_dimensions)

        for i, layer in enumerate(self.layers):
            embedding2 = layer(embedding)
            if (self.residual) and (i < len(self.layers) - 1):
                embedding = embedding + embedding2
            else:
                embedding = embedding2

        return embedding.reshape(cell_region_embedding.shape[:-1])


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
    """The region of interest"""

    @classmethod
    def create(
        cls,
        fragments: Fragments,
        transcriptome: Transcriptome,
        fold=None,
        layer: str | None = None,
        path: str | os.PathLike = None,
        dummy: bool = False,
        n_frequencies: int = (1000, 500, 250, 125, 63, 31),
        reduce: str = "sum",
        nonlinear: bool = "silu",
        n_embedding_dimensions: int = 100,
        embedding_to_expression_initialization: str = "default",
        dropout_rate_fragment_embedder: float = 0.0,
        n_layers_fragment_embedder=1,
        residual_fragment_embedder=True,
        batchnorm_fragment_embedder=False,
        layernorm_fragment_embedder=False,
        n_layers_embedding2expression=5,
        dropout_rate_embedding2expression: float = 0.0,
        residual_embedding2expression=True,
        batchnorm_embedding2expression=False,
        layernorm_embedding2expression=True,
        overwrite=False,
        encoder=None,
        pooler=None,
        distance_encoder="direct",
        library_size_encoder="linear",
        library_size_encoder_kwargs=None,
        region_oi=None,
        encoder_kwargs=None,
        fragment_embedder_kwargs=None,
        **kwargs: Any,
    ) -> None:
        """
        Create the model

        Parameters:
            fragments:
                the fragments
            transcriptome:
                the transcriptome
            fold:
                the fold
            layer:
                which layer from the transcriptome to use for training and inference, will use the first layer if None
            path:
                the path to save the model
            dummy:
                whether to use a dummy model that just counts fragments.
            n_frequencies:
                the number of frequencies to use for the encoding
            reduce:
                the reduction to use for pooling fragments across regions and cells
            nonlinear:
                whether to use a non-linear activation function
            n_embedding_dimensions:
                the number of embedding dimensions
            dropout_rate:
                the dropout rate
        """
        self = super(Model, cls).create(
            path=path, fragments=fragments, transcriptome=transcriptome, fold=fold, reset=overwrite
        )

        self.fold = fold

        if layer is not None:
            self.layer = layer
        else:
            layer = list(self.transcriptome.layers.keys())[0]
        self.layer = layer

        if region_oi is None:
            region_oi = fragments.var.index[0]
        self.region_oi = region_oi

        if fragment_embedder_kwargs is None:
            fragment_embedder_kwargs = {}

        if dummy is True:
            self.fragment_embedder = FragmentEmbedderCounter()
        else:
            self.fragment_embedder = FragmentEmbedder(
                n_frequencies=n_frequencies,
                nonlinear=nonlinear,
                n_layers=n_layers_fragment_embedder,
                n_embedding_dimensions=n_embedding_dimensions,
                dropout_rate=dropout_rate_fragment_embedder,
                residual=residual_fragment_embedder,
                batchnorm=batchnorm_fragment_embedder,
                layernorm=layernorm_fragment_embedder,
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

        # library size encoder
        if library_size_encoder == "linear":
            library_size_encoder_kwargs = library_size_encoder_kwargs or {}
            self.library_size_encoder = LibrarySizeEncoder(fragments, **library_size_encoder_kwargs)
            n_input_embedding_dimensions += self.library_size_encoder.n_embedding_dimensions
        elif library_size_encoder is None:
            self.library_size_encoder = None
        else:
            raise ValueError(library_size_encoder + " is not a valid library size encoder")

        # embedding to expression
        self.embedding_to_expression = EmbeddingToExpression(
            fragments=fragments,
            n_input_embedding_dimensions=n_input_embedding_dimensions,
            n_embedding_dimensions=self.fragment_embedder.n_embedding_dimensions,
            initialization=embedding_to_expression_initialization,
            n_layers=n_layers_embedding2expression,
            residual=residual_embedding2expression,
            dropout_rate=dropout_rate_embedding2expression,
            batchnorm=batchnorm_embedding2expression,
            layernorm=layernorm_embedding2expression,
            nonlinear=nonlinear,
        )

        return self

    def forward(self, data):
        """
        Make a prediction given a data object
        """
        assert data.minibatch.n_regions == 1

        fragment_embedding = self.fragment_embedder(data)
        cell_region_embedding = self.embedding_region_pooler(
            fragment_embedding,
            data.fragments.local_cellxregion_ix,
            data.minibatch.n_cells,
            data.minibatch.n_regions,
        )
        if hasattr(self, "library_size_encoder") and (self.library_size_encoder is not None):
            library_size_encoding = self.library_size_encoder(data).unsqueeze(-2)
            cell_region_embedding = torch.cat([cell_region_embedding, library_size_encoding], dim=-1)
        expression_predicted = self.embedding_to_expression(cell_region_embedding)
        self.expression_predicted = expression_predicted

        return expression_predicted

    def forward_loss(self, data):
        """
        Make a prediction and calculate the loss
        """
        expression_predicted = self.forward(data)
        expression_true = data.transcriptome.value
        return paircor_loss(expression_predicted, expression_true)
        # return pairzmse_loss(expression_predicted, expression_true)

    def forward_region_loss(self, data):
        """
        Make a prediction and calculate the loss on a per region basis
        """
        expression_predicted = self.forward(data)
        expression_true = data.transcriptome.value
        return region_paircor_loss(expression_predicted, expression_true)

    def forward_multiple(self, data, fragments_oi, min_fragments=1):
        """
        Make multiple predictions based on different sets of fragments

        Parameters:
            data:
                the data object
            fragments_oi:
                an iterator of boolean arrays indicating which fragments to use
            min_fragments:
                the minimum number of fragments that have to remove before re-calculating the prediction
        """
        fragment_embedding = self.fragment_embedder(data)

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
        cell_region_embedding = total_cell_region_embedding

        if hasattr(self, "library_size_encoder"):
            cell_region_embedding = torch.cat(
                [cell_region_embedding, self.library_size_encoder(data).unsqueeze(-2)],
                dim=-1,
            )

        total_expression_predicted = self.embedding_to_expression.forward(cell_region_embedding)

        for fragments_oi_ in fragments_oi:
            if (fragments_oi_ is not None) and ((~fragments_oi_).sum().item() > min_fragments):
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

                if hasattr(self, "library_size_encoder"):
                    cell_region_embedding = torch.cat(
                        [
                            cell_region_embedding,
                            self.library_size_encoder(data).unsqueeze(-2),
                        ],
                        dim=-1,
                    )

                expression_predicted = self.embedding_to_expression.forward(cell_region_embedding)
            else:
                n_fragments = total_n_fragments
                expression_predicted = total_expression_predicted

            yield expression_predicted, n_fragments

    def forward_multiple2(self, data, fragments_oi, min_fragments=1):
        fragment_embedding = self.fragment_embedder(data)

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
        cell_region_embedding = total_cell_region_embedding

        if hasattr(self, "library_size_encoder"):
            cell_region_embedding = torch.cat(
                [cell_region_embedding, self.library_size_encoder(data).unsqueeze(-2)],
                dim=-1,
            )

        # total_expression_predicted = self.embedding_to_expression.forward(
        #     cell_region_embedding
        # )

        tot = 0.0

        cell_region_embeddings = []

        start = time.time()

        for fragments_oi_ in fragments_oi:
            if (fragments_oi_ is not None) and ((~fragments_oi_).sum().item() > min_fragments):
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

                if hasattr(self, "library_size_encoder"):
                    cell_region_embedding = torch.cat(
                        [
                            cell_region_embedding,
                            self.library_size_encoder(data).unsqueeze(-2),
                        ],
                        dim=-1,
                    )

                cell_region_embeddings.append(cell_region_embedding)

                # expression_predicted = self.embedding_to_expression.forward(cell_region_embedding)
                # end = time.time()
                # tot += end - start

            else:
                cell_region_embedding = total_cell_region_embedding
                if hasattr(self, "library_size_encoder"):
                    cell_region_embedding = torch.cat(
                        [
                            cell_region_embedding,
                            self.library_size_encoder(data).unsqueeze(-2),
                        ],
                        dim=-1,
                    )
                cell_region_embeddings.append(cell_region_embedding)
                # n_fragments = total_n_fragments
                # expression_predicted = total_expression_predicted
        cell_region_embeddings = torch.stack(cell_region_embeddings, dim=0)
        expression_predicted = self.embedding_to_expression.forward(cell_region_embeddings)

        for expression_predicted_, n_fragments in zip(expression_predicted, total_n_fragments):
            yield expression_predicted_, n_fragments

        tot = time.time() - start
        print(tot)

    def train_model(
        self,
        fold: list = None,
        fragments: Fragments = None,
        transcriptome: Transcriptome = None,
        device=None,
        lr=1e-4,
        n_epochs=1000,
        pbar=True,
        n_regions_step=1,
        n_cells_step=20000,
        weight_decay=1e-1,
        checkpoint_every_epoch=1,
        optimizer="adam",
        n_cells_train=None,
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

        # set up minibatchers and loaders
        if self.region_oi is not None:
            fragments.var["ix"] = np.arange(len(fragments.var))
            region_ixs = fragments.var["ix"].loc[[self.region_oi]].values
        else:
            region_ixs = range(fragments.n_regions)

        if n_cells_train is not None:
            cells_train = fold["cells_train"][:n_cells_train]
        else:
            cells_train = fold["cells_train"]

        minibatcher_train = Minibatcher(
            cells_train,
            region_ixs,
            n_regions_step=n_regions_step,
            n_cells_step=n_cells_step,
        )
        minibatcher_validation = Minibatcher(
            fold["cells_validation"],
            region_ixs,
            n_regions_step=10,
            n_cells_step=20000,
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
                region_oi=self.region_oi,
            ),
            n_workers=2,
        )
        loaders_validation = LoaderPool(
            TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxregion_batch_size=minibatcher_validation.cellxregion_batch_size,
                layer=self.layer,
                region_oi=self.region_oi,
            ),
            n_workers=2,
        )

        if optimizer == "adam":
            optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer == "radam":
            optimizer = RAdam(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optimizer == "lbfgs":
            optimizer = Adamax(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError()

        trainer = SharedTrainer(
            self,
            loaders_train,
            loaders_validation,
            minibatcher_train,
            minibatcher_validation,
            optimizer,
            n_epochs=n_epochs,
            checkpoint_every_epoch=checkpoint_every_epoch,
            optimize_every_step=1,
            device=device,
            pbar=pbar,
            **kwargs,
        )

        self.trace = trainer.trace

        trainer.train()

    def get_prediction(
        self,
        fragments=None,
        transcriptome=None,
        cells=None,
        cell_ixs=None,
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

        regions = [self.region_oi]
        region_ixs = [fragments.var.index.get_loc(self.region_oi)]

        if device is None:
            device = get_default_device()

        minibatches = Minibatcher(
            cell_ixs,
            region_ixs,
            n_regions_step=500,
            n_cells_step=1000,
            # n_cells_step=200,
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
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    region_mapping[data.minibatch.regions_oi],
                )
            ] = pred_mb.cpu().numpy()
            expected[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    region_mapping[data.minibatch.regions_oi],
                )
            ] = data.transcriptome.value.cpu().numpy()
            n_fragments[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    region_mapping[data.minibatch.regions_oi],
                )
            ] = (
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
                    coords={
                        fragments.obs.index.name: cells,
                        fragments.var.index.name: regions,
                    },
                ),
                "expected": xr.DataArray(
                    expected,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={
                        fragments.obs.index.name: cells,
                        fragments.var.index.name: regions,
                    },
                ),
                "n_fragments": xr.DataArray(
                    n_fragments,
                    dims=(fragments.obs.index.name, fragments.var.index.name),
                    coords={
                        fragments.obs.index.name: cells,
                        fragments.var.index.name: regions,
                    },
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
        min_fragments=5,
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

        loader = TranscriptomeFragments(
            transcriptome=transcriptome,
            fragments=fragments,
            cellxregion_batch_size=minibatcher.cellxregion_batch_size,
            layer=self.layer,
            region_oi=self.region_oi,
        )

        predicted = np.zeros((len(censorer), len(cell_ixs), len(region_ixs)), dtype=float)
        expected = np.zeros((len(cell_ixs), len(region_ixs)), dtype=float)
        n_fragments = np.zeros((len(censorer), len(cell_ixs), len(region_ixs)), dtype=int)

        predicted = []
        n_fragments = []

        cell_mapping = np.zeros(fragments.n_cells, dtype=np.int64)
        cell_mapping[cell_ixs] = np.arange(len(cell_ixs))
        region_mapping = np.zeros(fragments.n_regions, dtype=np.int64)
        region_mapping[region_ixs] = np.arange(len(region_ixs))

        self.eval()
        self.to(device)
        assert len(minibatcher) == 1
        for minibatch in minibatcher:
            data = loader.load(minibatch)
            data = data.to(device)
            fragments_oi = censorer(data)

            with torch.no_grad():
                for (
                    design_ix,
                    (
                        pred_mb,
                        n_fragments_oi_mb,
                    ),
                ) in enumerate(self.forward_multiple(data, fragments_oi, min_fragments=min_fragments)):
                    predicted.append(pred_mb)
                    n_fragments.append(n_fragments_oi_mb)
            expected = data.transcriptome.value.cpu().numpy()

        self.to("cpu")
        predicted = torch.stack(predicted, axis=0).cpu().numpy()
        n_fragments = torch.stack(n_fragments, axis=0).cpu().numpy()

        return predicted, expected, n_fragments

    def get_performance_censored(
        self,
        censorer,
        fragments=None,
        transcriptome=None,
        cells=None,
        cell_ixs=None,
        regions=None,
        region_ixs=None,
        device=None,
        min_fragments=5,
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

        loader = TranscriptomeFragments(
            transcriptome=transcriptome,
            fragments=fragments,
            cellxregion_batch_size=minibatcher.cellxregion_batch_size,
            layer=self.layer,
            region_oi=self.region_oi,
        )

        deltacors = []
        losts = []
        effects = []

        self.eval()
        self.to(device)
        for minibatch in minibatcher:
            data = loader.load(minibatch).to(device)
            fragments_oi = censorer(data)

            with torch.no_grad():
                for (
                    design_ix,
                    (
                        pred_mb,
                        n_fragments_oi_mb,
                    ),
                ) in enumerate(self.forward_multiple(data, fragments_oi, min_fragments=min_fragments)):
                    if design_ix == 0:
                        cor_baseline = paircor(pred_mb, data.transcriptome.value).cpu().numpy()
                        deltacor = 0
                        n_fragments_baseline = n_fragments_oi_mb.sum().cpu().numpy()
                        prediction_baseline = pred_mb.cpu().numpy()
                        lost = 0
                        effect = 0
                    else:
                        cor = paircor(pred_mb, data.transcriptome.value).cpu().numpy()
                        deltacor = cor - cor_baseline
                        lost = n_fragments_baseline - n_fragments_oi_mb.sum().cpu().numpy()
                        effect = (pred_mb.cpu().numpy() - prediction_baseline).mean()

                        deltacors.append(deltacor)
                        losts.append(lost)
                        effects.append(effect)

        self.to("cpu")

        deltacors = np.concatenate(deltacors)
        print(deltacors)

        return deltacors, losts, effects


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

    def train_models(
        self,
        device=None,
        pbar=True,
        transcriptome=None,
        fragments=None,
        folds=None,
        regions_oi=None,
        **kwargs,
    ):
        if "device" in self.train_params and device is None:
            device = self.train_params["device"]

        if fragments is None:
            fragments = self.fragments
        if transcriptome is None:
            transcriptome = self.transcriptome
        if folds is None:
            folds = self.folds

        if regions_oi is None:
            if self.regions_oi is None:
                regions_oi = fragments.var.index
            else:
                regions_oi = self.regions_oi

        progress = itertools.product(enumerate(regions_oi), enumerate(folds))
        if pbar:
            progress = tqdm.tqdm(progress, total=len(regions_oi) * len(folds))

        for (region_ix, region), (fold_ix, fold) in progress:
            model_name = f"{region}_{fold_ix}"
            model_folder = self.models_path / (model_name)
            force = False
            if model_name not in self.models:
                force = True
            elif not self.models[model_name].o.state.exists(self.models[model_name]):
                force = True

            if force:
                model = Model.create(
                    fragments=fragments,
                    transcriptome=transcriptome,
                    fold=fold,
                    region_oi=region,
                    path=model_folder,
                    **self.model_params,
                )
                model.train_model(
                    device=device,
                    pbar=False,
                    **{**{k: v for k, v in self.train_params.items() if k not in ["device"]}, **kwargs},
                )
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

    def get_region_cors(self, fragments, transcriptome, folds, device=None):
        regions_oi = fragments.var.index if self.regions_oi is None else self.regions_oi

        from itertools import product

        cors = []

        if device is None:
            device = get_default_device()
        for region_id, (fold_ix, fold) in product(regions_oi, enumerate(folds)):
            if region_id + "_" + str(fold_ix) in self:
                model = self[region_id + "_" + str(fold_ix)]
                prediction = model.get_prediction(fragments, transcriptome, cell_ixs=fold["cells_test"], device=device)

                cors.append(
                    {
                        fragments.var.index.name: region_id,
                        "cor": np.corrcoef(
                            prediction["predicted"].values[:, 0],
                            prediction["expected"].values[:, 0],
                        )[0, 1],
                        "cor_n_fragments": np.corrcoef(
                            prediction["n_fragments"].values[:, 0],
                            prediction["expected"].values[:, 0],
                        )[0, 1],
                    }
                )

        cors = pd.DataFrame(cors).set_index(fragments.var.index.name)
        cors["deltacor"] = cors["cor"] - cors["cor_n_fragments"]

        return cors

    @property
    def design(self):
        design_dimensions = {
            "fold": range(len(self.folds)),
            "gene": self.regions_oi,
        }
        design = crossing(**design_dimensions)
        design.index = design["gene"] + "_" + design["fold"].astype(str)
        return design

    def fitted(self, region, fold_ix):
        return f"{region}_{fold_ix}" in self.models

    def get_prediction(self, region, fold_ix, **kwargs):
        model = self[f"{region}_{fold_ix}"]
        return model.get_prediction(**kwargs)

    def trained(self, region):
        return all([f"{region}_{fold_ix}" in self.models for fold_ix in range(len(self.folds))])

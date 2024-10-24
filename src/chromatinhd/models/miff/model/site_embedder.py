"""
Embed motif sites
"""


import torch
# import torch_scatter
import math
import numpy as np

import pickle

from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.models import HybridModel
from chromatinhd.flow import Flow, Stored

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

from typing import Any


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


class SiteEmbedder(torch.nn.Module):
    dropout_rate = 0.0

    def __init__(
        self,
        n_frequencies=10,
        n_layers=1,
        dropout_rate=0.0,
        add_residual_count=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sine_encoding = SineEncoding(n_frequencies=n_frequencies, n_coordinates=1)

        self.n_embedding_dimensions = self.sine_encoding.n_embedding_dimensions

        layers = []
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
            layers.append(torch.nn.BatchNorm1d(self.n_embedding_dimensions))
            if layer_ix == 0:
                layers.append(torch.nn.ReLU())
            else:
                layers.append(torch.nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(torch.nn.Dropout(self.dropout_rate))
        layers.append(torch.nn.Linear(self.n_embedding_dimensions, self.n_embedding_dimensions, bias=False))

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, coordinates):
        embedding = self.sine_encoding(coordinates)
        embedding = self.nn(embedding)

        return embedding


class SiteEmbedderCounter(torch.nn.Sequential):
    """
    Dummy embedding of fragments in a single embedding dimension of ones
    """

    def __init__(self, *args, **kwargs):
        self.n_embedding_dimensions = 1
        super().__init__(*args, **kwargs)

    def forward(self, coordinates):
        return torch.ones((*coordinates.shape[:-1], 1), device=coordinates.device, dtype=torch.float)


class SiteEmbeddingGenePooler(torch.nn.Module):
    """
    Pools fragments across genes and cells
    """

    def __init__(self, reduce="sum"):
        self.reduce = reduce
        super().__init__()

    def forward(self, embedding, local_gene_ix, n_genes):
        if self.reduce == "sum":
            gene_embedding = torch_scatter.segment_sum_coo(embedding, local_gene_ix.to(torch.int64), dim_size=n_genes)
        else:
            raise ValueError()
        return gene_embedding


class GeneEmbedder(torch.nn.Module):
    """
    Embeds genes
    """

    def __init__(
        self,
        n_input_dimensions,
        n_output_dimensions,
        n_layers=1,
        dropout_rate=0.0,
    ):
        super().__init__()

        self.n_input_dimensions = n_input_dimensions
        self.n_output_dimensions = n_output_dimensions
        self.dropout_rate = dropout_rate

        layers = []
        for layer_ix in range(n_layers):
            if layer_ix == 0:
                layers.append(
                    torch.nn.Linear(
                        self.n_input_dimensions,
                        self.n_input_dimensions,
                    )
                )
            else:
                layers.append(torch.nn.Linear(self.n_input_dimensions, self.n_input_dimensions))
            layers.append(torch.nn.BatchNorm1d(self.n_input_dimensions))
            layers.append(torch.nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(torch.nn.Dropout(self.dropout_rate))
        layers.append(torch.nn.Linear(self.n_input_dimensions, self.n_output_dimensions, bias=False))

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, embedding):
        embedding = self.nn(embedding)

        return embedding

"""
Additive model for predicting gene expression from fragments
"""


import torch
import torch_scatter
import math
import numpy as np
import xarray as xr
import pandas as pd

import pickle

from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.models import HybridModel
from chromatinhd.flow import Linked, Flow, Stored

from chromatinhd.models.pred.loader.minibatches import Minibatcher
from chromatinhd.models.pred.loader.transcriptome_fragments import (
    TranscriptomeFragments,
)
from chromatinhd.models.pred.trainer import Trainer
from chromatinhd.loaders import LoaderPool2
from chromatinhd.optim import SparseDenseAdam


def paircor(x, y, dim=0, eps=0.1):
    divisor = (y.std(dim) * x.std(dim)) + eps
    cor = ((x - x.mean(dim, keepdims=True)) * (y - y.mean(dim, keepdims=True))).mean(
        dim
    ) / divisor
    return cor


def paircor_loss(x, y):
    return -paircor(x, y).mean() * 100


def gene_paircor_loss(x, y):
    return -paircor(x, y) * 100


class SineEncoding(torch.nn.Module):
    def __init__(self, n_frequencies):
        super().__init__()

        self.register_buffer(
            "frequencies",
            torch.tensor(
                [
                    [1 / 1000 ** (2 * i / n_frequencies)] * 2
                    for i in range(1, n_frequencies + 1)
                ]
            ).flatten(-2),
        )
        self.register_buffer(
            "shifts",
            torch.tensor(
                [[0, torch.pi / 2] for _ in range(1, n_frequencies + 1)]
            ).flatten(-2),
        )

        self.n_embedding_dimensions = n_frequencies * 2 * 2

    def forward(self, coordinates):
        embedding = torch.sin(
            (coordinates[..., None] * self.frequencies + self.shifts).flatten(-2)
        )
        return embedding


class FragmentEmbedder(torch.nn.Module):
    dropout_rate = 0.0

    def __init__(
        self,
        n_genes,
        n_frequencies=10,
        n_embedding_dimensions=5,
        nonlinear=True,
        dropout_rate=0.0,
        **kwargs
    ):

        self.n_embedding_dimensions = n_embedding_dimensions

        self.nonlinear = nonlinear
        self.dropout_rate = dropout_rate

        super().__init__(**kwargs)

        self.sine_encoding = SineEncoding(n_frequencies=n_frequencies)

        # default initialization same as a torch.nn.Linear
        self.bias1 = EmbeddingTensor(
            n_genes,
            (self.n_embedding_dimensions,),
            sparse=True,
        )
        self.bias1.data.zero_()

        self.weight1 = EmbeddingTensor(
            n_genes,
            (
                self.sine_encoding.n_embedding_dimensions,
                self.n_embedding_dimensions,
            ),
            sparse=True,
        )
        stdv = 1.0 / math.sqrt(self.weight1.shape[-1])  # / 100
        self.weight1.data.uniform_(-stdv, stdv)

    def forward(self, coordinates, gene_ix):
        embedding = self.sine_encoding(coordinates)
        embedding = torch.einsum(
            "ab,abc->ac", embedding, self.weight1(gene_ix)
        ) + self.bias1(gene_ix)
        # embedding = (embedding[..., None] * self.weight1[gene_ix]).sum(-2)

        # non-linear
        if self.nonlinear is True:
            embedding = torch.sigmoid(embedding)
        elif isinstance(self.nonlinear, str) and self.nonlinear == "relu":
            embedding = torch.relu(embedding)
        elif isinstance(self.nonlinear, str) and self.nonlinear == "elu":
            embedding = torch.nn.functional.elu(embedding)

        if self.dropout_rate > 0:
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

    def forward(self, coordinates, gene_ix):
        return torch.ones(
            (*coordinates.shape[:-1], 1), device=coordinates.device, dtype=torch.float
        )


class EmbeddingGenePooler(torch.nn.Module):
    """
    Pools fragments across genes and cells
    """

    def __init__(self, reduce="sum"):
        self.reduce = reduce
        super().__init__()

    def forward(self, embedding, fragment_cellxgene_ix, cell_n, gene_n):
        if self.reduce == "mean":
            cellxgene_embedding = torch_scatter.segment_mean_coo(
                embedding, fragment_cellxgene_ix, dim_size=cell_n * gene_n
            )
        elif self.reduce == "sum":
            cellxgene_embedding = torch_scatter.segment_sum_coo(
                embedding, fragment_cellxgene_ix, dim_size=cell_n * gene_n
            )
        else:
            raise ValueError()
        cell_gene_embedding = cellxgene_embedding.reshape(
            (cell_n, gene_n, cellxgene_embedding.shape[-1])
        )
        return cell_gene_embedding


class EmbeddingToExpression(torch.nn.Module):
    """
    Predicts gene expression using a [cell, gene, component] embedding in a gene-specific manner
    """

    def __init__(
        self, n_genes, n_embedding_dimensions=5, initialization="default", **kwargs
    ):
        self.n_genes = n_genes
        self.n_embedding_dimensions = n_embedding_dimensions

        super().__init__()

        # set bias to empirical mean
        self.bias1 = EmbeddingTensor(
            n_genes,
            tuple(),
            sparse=True,
        )
        self.bias1.data[:] = 0.0

        self.weight1 = EmbeddingTensor(
            n_genes,
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

    def forward(self, cell_gene_embedding, gene_ix):
        out = (cell_gene_embedding * self.weight1(gene_ix)).sum(-1) + self.bias1(
            gene_ix
        ).squeeze()
        return out

    def parameters_sparse(self):
        return [self.bias1.weight, self.weight1.weight]


class Model(torch.nn.Module, HybridModel):
    def __init__(
        self,
        n_genes,
        dummy=False,
        n_frequencies=50,
        reduce="sum",
        nonlinear=True,
        n_embedding_dimensions=10,
        dropout_rate=0.0,
        embedding_to_expression_initialization="default",
        **kwargs
    ):
        super().__init__()

        if dummy:
            self.fragment_embedder = FragmentEmbedderCounter()
        else:
            self.fragment_embedder = FragmentEmbedder(
                n_frequencies=n_frequencies,
                n_genes=n_genes,
                nonlinear=nonlinear,
                n_embedding_dimensions=n_embedding_dimensions,
                dropout_rate=dropout_rate,
            )
        self.embedding_gene_pooler = EmbeddingGenePooler(reduce=reduce)
        self.embedding_to_expression = EmbeddingToExpression(
            n_genes=n_genes,
            n_embedding_dimensions=self.fragment_embedder.n_embedding_dimensions,
            initialization=embedding_to_expression_initialization,
        )

    def forward(self, data):
        fragment_embedding = self.fragment_embedder(
            data.fragments.coordinates, data.fragments.genemapping
        )
        cell_gene_embedding = self.embedding_gene_pooler(
            fragment_embedding,
            data.fragments.local_cellxgene_ix,
            data.minibatch.n_cells,
            data.minibatch.n_genes,
        )
        expression_predicted = self.embedding_to_expression(
            cell_gene_embedding, data.minibatch.genes_oi_torch
        )
        return expression_predicted

    def forward_loss(self, data):
        expression_predicted = self.forward(data)
        expression_true = data.transcriptome.value
        return paircor_loss(expression_predicted, expression_true)

    def forward_gene_loss(self, data):
        expression_predicted = self.forward(data)
        expression_true = data.transcriptome.value
        return gene_paircor_loss(expression_predicted, expression_true)

    def forward_multiple(self, data, fragments_oi, min_fragments=1):
        fragment_embedding = self.fragment_embedder(
            data.fragments.coordinates, data.fragments.genemapping
        )

        total_n_fragments = torch.bincount(
            data.fragments.local_cellxgene_ix,
            minlength=data.minibatch.n_genes * data.minibatch.n_cells,
        ).reshape((data.minibatch.n_cells, data.minibatch.n_genes))

        total_cell_gene_embedding = self.embedding_gene_pooler.forward(
            fragment_embedding,
            data.fragments.local_cellxgene_ix,
            data.minibatch.n_cells,
            data.minibatch.n_genes,
        )

        total_expression_predicted = self.embedding_to_expression.forward(
            total_cell_gene_embedding, data.minibatch.genes_oi_torch
        )

        for fragments_oi_ in fragments_oi:
            if (fragments_oi_ is not None) and ((~fragments_oi_).sum() > min_fragments):
                lost_fragments_oi = ~fragments_oi_
                lost_local_cellxgene_ix = data.fragments.local_cellxgene_ix[
                    lost_fragments_oi
                ]
                n_fragments = total_n_fragments - torch.bincount(
                    lost_local_cellxgene_ix,
                    minlength=data.minibatch.n_genes * data.minibatch.n_cells,
                ).reshape((data.minibatch.n_cells, data.minibatch.n_genes))
                cell_gene_embedding = (
                    total_cell_gene_embedding
                    - self.embedding_gene_pooler.forward(
                        fragment_embedding[lost_fragments_oi],
                        lost_local_cellxgene_ix,
                        data.minibatch.n_cells,
                        data.minibatch.n_genes,
                    )
                )

                expression_predicted = self.embedding_to_expression.forward(
                    cell_gene_embedding, data.minibatch.genes_oi_torch
                )
            else:
                n_fragments = total_n_fragments
                expression_predicted = total_expression_predicted

            yield expression_predicted, n_fragments

    def train_model(self, fragments, transcriptome, fold, device="cuda"):
        # set up minibatchers and loaders
        minibatcher_train = Minibatcher(
            fold["cells_train"],
            range(fragments.n_genes),
            n_genes_step=500,
            n_cells_step=200,
        )
        minibatcher_validation = Minibatcher(
            fold["cells_validation"],
            range(fragments.n_genes),
            n_genes_step=10,
            n_cells_step=10000,
            permute_cells=False,
            permute_genes=False,
        )

        loaders_train = LoaderPool2(
            TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxgene_batch_size=minibatcher_train.cellxgene_batch_size,
            ),
            n_workers=10,
        )
        loaders_validation = LoaderPool2(
            TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxgene_batch_size=minibatcher_validation.cellxgene_batch_size,
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
                lr=1e-2,
                weight_decay=1e-5,
            ),
            n_epochs=30,
            checkpoint_every_epoch=1,
            optimize_every_step=1,
            device=device,
        )

        trainer.train()
        # trainer.trace.plot()

    def get_prediction(
        self,
        fragments,
        transcriptome,
        cells=None,
        cell_ixs=None,
        genes=None,
        gene_ixs=None,
        device="cuda",
        return_raw=False,
    ):
        """
        Returns the prediction of a dataset
        """
        if cell_ixs is None:
            if cells is None:
                cells = fragments.obs.index
            fragments.obs["ix"] = np.arange(len(fragments.obs))
            cell_ixs = fragments.obs.loc[cells]["ix"].values
        if cells is None:
            cells = fragments.obs.index[cell_ixs]

        if gene_ixs is None:
            if genes is None:
                genes = fragments.var.index
            fragments.var["ix"] = np.arange(len(fragments.var))
            gene_ixs = fragments.var.loc[genes]["ix"].values
        if genes is None:
            genes = fragments.var.index[gene_ixs]

        minibatches = Minibatcher(
            cell_ixs,
            gene_ixs,
            n_genes_step=500,
            n_cells_step=200,
            use_all_cells=True,
            use_all_genes=True,
            permute_cells=False,
            permute_genes=False,
        )
        loaders = LoaderPool2(
            TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxgene_batch_size=minibatches.cellxgene_batch_size,
            ),
            n_workers=5,
        )
        loaders.initialize(minibatches)

        predicted = np.zeros((len(cell_ixs), fragments.n_genes))
        expected = np.zeros((len(cell_ixs), fragments.n_genes))
        n_fragments = np.zeros((len(cell_ixs), fragments.n_genes))

        cell_mapping = np.zeros(fragments.n_cells, dtype=np.int64)
        cell_mapping[cell_ixs] = np.arange(len(cell_ixs))

        device = "cuda"
        self.eval()
        self = self.to(device)

        for data in loaders:
            data = data.to(device)
            with torch.no_grad():
                pred_mb = self.forward(data)
            predicted[
                np.ix_(cell_mapping[data.minibatch.cells_oi], data.minibatch.genes_oi)
            ] = pred_mb.cpu().numpy()
            expected[
                np.ix_(cell_mapping[data.minibatch.cells_oi], data.minibatch.genes_oi)
            ] = data.transcriptome.value.cpu().numpy()
            n_fragments[
                np.ix_(cell_mapping[data.minibatch.cells_oi], data.minibatch.genes_oi)
            ] = (
                torch.bincount(
                    data.fragments.local_cellxgene_ix,
                    minlength=len(data.minibatch.cells_oi)
                    * len(data.minibatch.genes_oi),
                )
                .reshape(len(data.minibatch.cells_oi), len(data.minibatch.genes_oi))
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
                    dims=("cell", "gene"),
                    coords={"cell": cells, "gene": fragments.var.index},
                ),
                "expected": xr.DataArray(
                    expected,
                    dims=("cell", "gene"),
                    coords={"cell": cells, "gene": fragments.var.index},
                ),
                "n_fragments": xr.DataArray(
                    n_fragments,
                    dims=("cell", "gene"),
                    coords={"cell": cells, "gene": fragments.var.index},
                ),
            }
        )
        return result

    def get_prediction_censored(
        self,
        fragments,
        transcriptome,
        censorer,
        cells=None,
        cell_ixs=None,
        genes=None,
        gene_ixs=None,
        device="cuda",
    ):
        """
        Returns the prediction of multiple censored dataset
        """
        if cell_ixs is None:
            if cells is None:
                cells = fragments.obs.index
            fragments.obs["ix"] = np.arange(len(fragments.obs))
            cell_ixs = fragments.obs.loc[cells]["ix"].values
        if cells is None:
            cells = fragments.obs.index[cell_ixs]

        if gene_ixs is None:
            if genes is None:
                genes = fragments.var.index
            fragments.var["ix"] = np.arange(len(fragments.var))
            gene_ixs = fragments.var.loc[genes]["ix"].values
        if genes is None:
            genes = fragments.var.index[gene_ixs]

        minibatcher = Minibatcher(
            cell_ixs,
            gene_ixs,
            n_genes_step=500,
            n_cells_step=5000,
            use_all_cells=True,
            use_all_genes=True,
            permute_cells=False,
            permute_genes=False,
        )
        loaders = LoaderPool2(
            TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxgene_batch_size=minibatcher.cellxgene_batch_size,
            ),
            n_workers=10,
        )
        loaders.initialize(minibatcher)

        predicted = np.zeros((len(censorer), len(cell_ixs), len(gene_ixs)), dtype=float)
        expected = np.zeros((len(cell_ixs), len(gene_ixs)), dtype=float)
        n_fragments = np.zeros((len(censorer), len(cell_ixs), len(gene_ixs)), dtype=int)

        cell_mapping = np.zeros(fragments.n_cells, dtype=np.int64)
        cell_mapping[cell_ixs] = np.arange(len(cell_ixs))
        gene_mapping = np.zeros(fragments.n_genes, dtype=np.int64)
        gene_mapping[gene_ixs] = np.arange(len(gene_ixs))

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
                        gene_mapping[data.minibatch.genes_oi],
                    )
                    predicted[ix] = pred_mb.cpu().numpy()
                    n_fragments[ix] = n_fragments_oi_mb.cpu().numpy()
            expected[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    gene_mapping[data.minibatch.genes_oi],
                )
            ] = data.transcriptome.value.cpu().numpy()

        self.to("cpu")

        return predicted, expected, n_fragments


class Models(Flow):
    n_models = Stored("n_models")

    @property
    def models_path(self):
        path = self.path / "models"
        path.mkdir(exist_ok=True)
        return path

    def train_models(self, fragments, transcriptome, folds, device="cuda"):
        self.n_models = len(folds)
        for fold_ix, fold in [(fold_ix, fold) for fold_ix, fold in enumerate(folds)]:
            desired_outputs = [self.models_path / ("model_" + str(fold_ix) + ".pkl")]
            force = False
            if not all([desired_output.exists() for desired_output in desired_outputs]):
                force = True

            if force:
                model = Model(
                    n_genes=fragments.n_genes,
                )
                model.train_model(fragments, transcriptome, fold, device=device)

                model = model.to("cpu")

                pickle.dump(
                    model,
                    open(self.models_path / ("model_" + str(fold_ix) + ".pkl"), "wb"),
                )

    def __getitem__(self, ix):
        return pickle.load(
            (self.models_path / ("model_" + str(ix) + ".pkl")).open("rb")
        )

    def __len__(self):
        return self.n_models

    def __iter__(self):
        for ix in range(len(self)):
            yield self[ix]

    def get_gene_cors(self, fragments, transcriptome, folds, device="cuda"):
        cor_predicted = np.zeros((len(fragments.var.index), len(folds)))
        cor_n_fragments = np.zeros((len(fragments.var.index), len(folds)))
        n_fragments = np.zeros((len(fragments.var.index), len(folds)))
        for model_ix, (model, fold) in enumerate(zip(self, folds)):
            prediction = model.get_prediction(
                fragments, transcriptome, cell_ixs=fold["cells_test"], device=device
            )

            cor_predicted[:, model_ix] = paircor(
                prediction["predicted"].values, prediction["expected"].values
            )
            cor_n_fragments[:, model_ix] = paircor(
                prediction["n_fragments"].values, prediction["expected"].values
            )

            n_fragments[:, model_ix] = prediction["n_fragments"].values.sum(0)
        cor_predicted = pd.Series(
            cor_predicted.mean(1), index=fragments.var.index, name="cor_predicted"
        )
        cor_n_fragments = pd.Series(
            cor_n_fragments.mean(1), index=fragments.var.index, name="cor_n_fragments"
        )
        n_fragments = pd.Series(
            n_fragments.mean(1), index=fragments.var.index, name="n_fragments"
        )
        result = pd.concat([cor_predicted, cor_n_fragments, n_fragments], axis=1)
        result["deltacor"] = result["cor_predicted"] - result["cor_n_fragments"]

        return result

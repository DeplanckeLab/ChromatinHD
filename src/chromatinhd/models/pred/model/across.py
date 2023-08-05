"""
Additive model for predicting gene expression from fragments
"""

import pickle

import numpy as np
import pandas as pd
import torch
import torch_scatter
import xarray as xr

from chromatinhd.flow import Flow, Stored
from chromatinhd.loaders import LoaderPool
from chromatinhd.models import HybridModel
from chromatinhd.models.pred.loader.minibatches import Minibatcher
from chromatinhd.models.pred.loader.transcriptome_fragments import (
    TranscriptomeFragments,
)
from chromatinhd.models.pred.trainer import Trainer2
from chromatinhd.optim import SparseDenseAdam

from chromatinhd import get_default_device

from .loss import gene_paircor_loss, paircor, paircor_loss


class SineEncoding(torch.nn.Module):
    def __init__(self, n_frequencies, add_length=False):
        super().__init__()

        self.register_buffer(
            "frequencies",
            torch.tensor([[1 / 1000 ** (2 * i / n_frequencies)] * 2 for i in range(1, n_frequencies + 1)]).flatten(-2),
        )
        self.register_buffer(
            "shifts",
            torch.tensor([[0, torch.pi / 2] for _ in range(1, n_frequencies + 1)]).flatten(-2),
        )

        self.n_embedding_dimensions = n_frequencies * 2 * 2

        if add_length:
            self.n_embedding_dimensions += n_frequencies * 2
        self.add_length = add_length

    def forward(self, coordinates):
        embedding = torch.sin((coordinates[..., None] * self.frequencies + self.shifts).flatten(-2))
        if self.add_length:
            embedding = torch.cat(
                [
                    embedding,
                    torch.sin(
                        (
                            (coordinates[..., 0, None] - coordinates[..., 1, None])[..., None] * self.frequencies
                            + self.shifts
                        ).flatten(-2)
                    ),
                ],
                -1,
            )
        return embedding


class FragmentEmbedder(torch.nn.Module):
    dropout_rate = 0.0

    def __init__(
        self,
        n_frequencies=10,
        n_embedding_dimensions=5,
        n_layers=1,
        nonlinear=True,
        dropout_rate=0.0,
        add_length=False,
        add_residual_count=False,
        **kwargs,
    ):
        self.n_embedding_dimensions = n_embedding_dimensions

        self.nonlinear = nonlinear
        self.dropout_rate = dropout_rate

        super().__init__(**kwargs)

        self.sine_encoding = SineEncoding(n_frequencies=n_frequencies, add_length=add_length)

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
            layers.append(torch.nn.Sigmoid())
            if self.dropout_rate > 0:
                layers.append(torch.nn.Dropout(self.dropout_rate))
        layers.append(torch.nn.Linear(self.n_embedding_dimensions, self.n_embedding_dimensions))

        self.nn = torch.nn.Sequential(*layers)

        if add_residual_count:
            self.n_embedding_dimensions += 1
        self.add_residual_count = add_residual_count

    def forward(self, coordinates):
        embedding = self.sine_encoding(coordinates)
        embedding = self.nn(embedding)

        if self.add_residual_count:
            embedding = torch.cat(
                [
                    embedding,
                    torch.ones(
                        (*coordinates.shape[:-1], 1),
                        device=coordinates.device,
                        dtype=torch.float,
                    ),
                ],
                -1,
            )

        return embedding


class FragmentEmbedderCounter(torch.nn.Module):
    """
    Dummy embedding of fragments in a single embedding dimension of ones
    """

    def __init__(self, *args, **kwargs):
        self.n_embedding_dimensions = 1
        super().__init__(*args, **kwargs)

    def forward(self, coordinates):
        return torch.ones((*coordinates.shape[:-1], 1), device=coordinates.device, dtype=torch.float)


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
        cell_gene_embedding = cellxgene_embedding.reshape((cell_n, gene_n, cellxgene_embedding.shape[-1]))
        return cell_gene_embedding


class EmbeddingToExpression(torch.nn.Module):
    """
    Predicts gene expression using a [cell, gene, component] embedding in a gene-specific manner
    """

    def __init__(self, n_embedding_dimensions=5, n_layers=1):
        self.n_embedding_dimensions = n_embedding_dimensions

        super().__init__()

        layers = []
        for layer_ix in range(n_layers):
            if layer_ix == 0:
                layers.append(torch.nn.Linear(self.n_embedding_dimensions, self.n_embedding_dimensions))
            else:
                layers.append(torch.nn.Linear(self.n_embedding_dimensions, self.n_embedding_dimensions))
            layers.append(torch.nn.Sigmoid())
        layers.append(torch.nn.Linear(self.n_embedding_dimensions, 1))
        self.nn = torch.nn.Sequential(*layers)

    def forward(self, cell_gene_embedding):
        out = self.nn(cell_gene_embedding).squeeze(-1)
        return out


class Model(torch.nn.Module, HybridModel):
    trace = None

    def __init__(
        self,
        dummy=False,
        n_frequencies=50,
        n_fragment_embedder_layers=1,
        n_embedding_to_expression_layers=1,
        reduce="sum",
        nonlinear=True,
        n_embedding_dimensions=10,
        dropout_rate=0.0,
        add_length=False,
        add_residual_count=False,
    ):
        super().__init__()

        if dummy:
            self.fragment_embedder = FragmentEmbedderCounter()
        else:
            self.fragment_embedder = FragmentEmbedder(
                n_frequencies=n_frequencies,
                nonlinear=nonlinear,
                n_embedding_dimensions=n_embedding_dimensions,
                dropout_rate=dropout_rate,
                n_layers=n_fragment_embedder_layers,
                add_length=add_length,
                add_residual_count=add_residual_count,
            )
        self.embedding_gene_pooler = EmbeddingGenePooler(reduce=reduce)
        self.embedding_to_expression = EmbeddingToExpression(
            n_embedding_dimensions=self.fragment_embedder.n_embedding_dimensions,
            n_layers=n_embedding_to_expression_layers if not dummy else 0,
        )

    def forward(self, data):
        fragment_embedding = self.fragment_embedder(data.fragments.coordinates)
        cell_gene_embedding = self.embedding_gene_pooler(
            fragment_embedding,
            data.fragments.local_cellxgene_ix,
            data.minibatch.n_cells,
            data.minibatch.n_genes,
        )
        expression_predicted = self.embedding_to_expression(cell_gene_embedding)
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
        fragment_embedding = self.fragment_embedder(data.fragments.coordinates, data.fragments.genemapping)

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

        total_expression_predicted = self.embedding_to_expression.forward(total_cell_gene_embedding)

        for fragments_oi_ in fragments_oi:
            if (fragments_oi_ is not None) and ((~fragments_oi_).sum() > min_fragments):
                lost_fragments_oi = ~fragments_oi_
                lost_local_cellxgene_ix = data.fragments.local_cellxgene_ix[lost_fragments_oi]
                n_fragments = total_n_fragments - torch.bincount(
                    lost_local_cellxgene_ix,
                    minlength=data.minibatch.n_genes * data.minibatch.n_cells,
                ).reshape((data.minibatch.n_cells, data.minibatch.n_genes))
                cell_gene_embedding = total_cell_gene_embedding - self.embedding_gene_pooler.forward(
                    fragment_embedding[lost_fragments_oi],
                    lost_local_cellxgene_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_genes,
                )

                expression_predicted = self.embedding_to_expression.forward(cell_gene_embedding)
            else:
                n_fragments = total_n_fragments
                expression_predicted = total_expression_predicted

            yield expression_predicted, n_fragments

    def train_model(self, fragments, transcriptome, fold, device=None):
        # set up minibatchers and loaders
        minibatcher_train = Minibatcher(
            fold["cells_train"],
            fold["genes_train"],
            n_genes_step=500,
            n_cells_step=500,
            permute_cells=True,
            permute_genes=True,
        )
        minibatcher_validation = Minibatcher(
            fold["cells_validation"],
            fold["genes_validation"],
            n_genes_step=500,
            n_cells_step=500,
            permute_cells=False,
            permute_genes=False,
        )

        if device is None:
            device = get_default_device()

        loaders_train = LoaderPool(
            TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxgene_batch_size=minibatcher_train.cellxgene_batch_size,
            ),
            n_workers=10,
        )
        loaders_validation = LoaderPool(
            TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxgene_batch_size=minibatcher_validation.cellxgene_batch_size,
            ),
            n_workers=5,
        )

        trainer = Trainer2(
            self,
            loaders_train,
            loaders_validation,
            minibatcher_train,
            minibatcher_validation,
            SparseDenseAdam(
                self.parameters_sparse(),
                self.parameters_dense(),
                lr=1e-3,
                weight_decay=1e-5,
            ),
            n_epochs=30,
            checkpoint_every_epoch=1,
            optimize_every_step=1,
            device=device,
        )

        trainer.train()
        self.trace = trainer.trace
        # trainer.trace.plot()

    def get_prediction(
        self,
        fragments,
        transcriptome,
        cells=None,
        cell_ixs=None,
        genes=None,
        gene_ixs=None,
        device=None,
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

        if device is None:
            device = get_default_device()

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
        loaders = LoaderPool(
            TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxgene_batch_size=minibatches.cellxgene_batch_size,
            ),
            n_workers=5,
        )
        loaders.initialize(minibatches)

        predicted = np.zeros((len(cell_ixs), len(gene_ixs)))
        expected = np.zeros((len(cell_ixs), len(gene_ixs)))
        n_fragments = np.zeros((len(cell_ixs), len(gene_ixs)))

        cell_mapping = np.zeros(fragments.n_cells, dtype=np.int64)
        cell_mapping[cell_ixs] = np.arange(len(cell_ixs))

        gene_mapping = np.zeros(fragments.n_genes, dtype=np.int64)
        gene_mapping[gene_ixs] = np.arange(len(gene_ixs))

        self.eval()
        self = self.to(device)

        for data in loaders:
            data = data.to(device)
            with torch.no_grad():
                pred_mb = self.forward(data)
            predicted[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    gene_mapping[data.minibatch.genes_oi],
                )
            ] = pred_mb.cpu().numpy()
            expected[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    gene_mapping[data.minibatch.genes_oi],
                )
            ] = data.transcriptome.value.cpu().numpy()
            n_fragments[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    gene_mapping[data.minibatch.genes_oi],
                )
            ] = (
                torch.bincount(
                    data.fragments.local_cellxgene_ix,
                    minlength=len(data.minibatch.cells_oi) * len(data.minibatch.genes_oi),
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
                    coords={"cell": cells, "gene": genes},
                ),
                "expected": xr.DataArray(
                    expected,
                    dims=("cell", "gene"),
                    coords={"cell": cells, "gene": genes},
                ),
                "n_fragments": xr.DataArray(
                    n_fragments,
                    dims=("cell", "gene"),
                    coords={"cell": cells, "gene": genes},
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
        device=None,
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

        if device is None:
            device = get_default_device()

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
        loaders = LoaderPool(
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
                model = Model()
                model.train_model(fragments, transcriptome, fold, device=device)

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

    def get_gene_cors(self, fragments, transcriptome, folds, device=None):
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

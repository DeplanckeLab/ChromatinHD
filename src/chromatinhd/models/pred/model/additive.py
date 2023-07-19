"""
- a positional encoding per fragment
- summarizes the encoding using a linear layer to a fragment embedding
- summation over cellxgene, to get a cellxgene embedding
- linear layer to fold change

Intuitively, for each gene, a fragment at a particular position has a positive or negative impact on expression
This effect is simply summed, without any interactions between positions
"""


import torch
import torch_scatter
import math
from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.models import HybridModel
from chromatinhd.flow import Linked, Flow


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
        self, n_genes, n_embedding_dimensions=5, initialization="ones", **kwargs
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
            # stdv = 1. / math.sqrt(self.weight1.size(-1))
            # self.weight1.data.uniform_(-stdv, stdv)
        elif initialization == "smaller":
            stdv = 1.0 / math.sqrt(self.weight1.size(-1)) / 100
            self.weight1.data.uniform_(-stdv, stdv)
        # stdv = 1. / math.sqrt(self.weight1.size(-1)) / 100
        # self.weight1.data.uniform_(-stdv, stdv)

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
        embedding_to_expression_initialization="ones",
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
            data.fragments.n_cells,
            data.fragments.n_genes,
        )
        expression_predicted = self.embedding_to_expression(
            cell_gene_embedding, data.fragments.genes_oi_torch
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

    def forward_multiple(self, data, fragments_oi, extract_total=False):
        fragment_embedding = self.fragment_embedder(data.coordinates, data.genemapping)

        for fragments_oi_ in fragments_oi:
            if fragments_oi_ is not None:
                fragment_embedding_ = fragment_embedding[fragments_oi_]
                local_cellxgene_ix = data.local_cellxgene_ix[fragments_oi_]
            else:
                fragment_embedding_ = fragment_embedding
                local_cellxgene_ix = data.local_cellxgene_ix

            cell_gene_embedding = self.embedding_gene_pooler(
                fragment_embedding_,
                local_cellxgene_ix,
                data.n_cells,
                data.n_genes,
            )
            expression_predicted = self.embedding_to_expression.forward(
                cell_gene_embedding, data.genes_oi_torch
            )

            if extract_total:
                n_fragments = torch.bincount(
                    data.local_cellxgene_ix,
                    minlength=data.n_genes * data.n_cells,
                ).reshape((data.n_cells, data.n_genes))
                yield expression_predicted, n_fragments
            else:
                if fragments_oi_ is None:
                    n_fragments_lost = 0
                else:
                    n_fragments_lost = torch.bincount(
                        data.local_cellxgene_ix[~fragments_oi_],
                        minlength=data.n_genes * data.n_cells,
                    ).reshape((data.n_cells, data.n_genes))
                yield expression_predicted, n_fragments_lost

    def train_model(self, fragments, transcriptome, fold, device="cuda"):
        import chromatinhd
        import chromatinhd.models.pred.loader
        import chromatinhd.models.pred.trainer
        import chromatinhd.loaders

        # set up minibatchers and loaders
        minibatcher_train = chromatinhd.models.pred.loader.minibatches.Minibatcher(
            fold["cells_train"],
            range(fragments.n_genes),
            n_genes_step=500,
            n_cells_step=200,
        )
        minibatcher_validation = chromatinhd.models.pred.loader.minibatches.Minibatcher(
            fold["cells_validation"],
            range(fragments.n_genes),
            n_genes_step=10,
            n_cells_step=10000,
            permute_cells=False,
            permute_genes=False,
        )

        loaders_train = chromatinhd.loaders.LoaderPool2(
            chromatinhd.models.pred.loader.transcriptome_fragments.TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxgene_batch_size=minibatcher_train.cellxgene_batch_size,
            ),
            n_workers=10,
        )
        loaders_validation = chromatinhd.loaders.LoaderPool2(
            chromatinhd.models.pred.loader.transcriptome_fragments.TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxgene_batch_size=minibatcher_validation.cellxgene_batch_size,
            ),
            n_workers=5,
        )

        trainer = chromatinhd.models.pred.trainer.Trainer(
            self,
            loaders_train,
            loaders_validation,
            minibatcher_train,
            minibatcher_validation,
            chromatinhd.optim.SparseDenseAdam(
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
        trainer.trace.plot()

    def get_prediction(self, fragments, transcriptome, cells=None, device="cuda"):
        import chromatinhd
        import chromatinhd.models.pred.loader
        import chromatinhd.models.pred.trainer
        import chromatinhd.loaders

        if cells is None:
            cells = np.arange(fragments.n_cells)

        minibatcher_test = chd.models.pred.loader.minibatches.Minibatcher(
            cells,
            range(fragments.n_genes),
            n_genes_step=500,
            n_cells_step=200,
            use_all_cells=True,
            use_all_genes=True,
            permute_cells=False,
            permute_genes=False,
        )
        loaders_test = chd.loaders.LoaderPool2(
            chd.models.pred.loader.transcriptome_fragments.TranscriptomeFragments,
            dict(
                transcriptome=transcriptome,
                fragments=fragments,
                cellxgene_batch_size=minibatcher_test.cellxgene_batch_size,
            ),
            n_workers=5,
        )
        loaders_test.initialize(minibatcher_test)

        predicted = np.zeros((len(cells), fragments.n_genes))
        expected = np.zeros((len(cells), fragments.n_genes))
        n_fragments = np.zeros((len(cells), fragments.n_genes))

        testcell_mapping = np.zeros(fragments.n_cells, dtype=np.int64)
        testcell_mapping[cells] = np.arange(len(cells))

        device = "cuda"
        model.eval()
        model = model.to(device)

        for data in loaders_test:
            data = data.to(device)
            with torch.no_grad():
                pred_mb = model.forward(data)
            predicted[
                np.ix_(
                    testcell_mapping[data.minibatch.cells_oi], data.minibatch.genes_oi
                )
            ] = pred_mb.cpu().numpy()
            expected[
                np.ix_(
                    testcell_mapping[data.minibatch.cells_oi], data.minibatch.genes_oi
                )
            ] = data.transcriptome.value.cpu().numpy()
            n_fragments[
                np.ix_(
                    testcell_mapping[data.minibatch.cells_oi], data.minibatch.genes_oi
                )
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

        model = model.to("cpu")


class Models(Flow):
    folds = Linked("folds")

    @property
    def models_path(self):
        return self.path / "models"

    def train(self, fragments, transcriptome, folds):
        for fold_ix, fold in [(fold_ix, fold) for fold_ix, fold in enumerate(folds)][
            fold_slice
        ]:
            desired_outputs = [self.models_path / (str(fold_ix) + ".pkl")]
            force = subdesign["force"].iloc[0]
            if not all([desired_output.exists() for desired_output in desired_outputs]):
                force = True

            if force:
                general_model_parameters = {
                    "mean_gene_expression": transcriptome_X_dense.mean(0),
                    "n_genes": fragments.n_genes,
                }

                general_loader_parameters = {
                    "fragments": fragments,
                    "cellxgene_batch_size": n_cells_step * n_genes_step,
                }

                fold = get_folds_training(fragments, [copy.copy(fold)])[0]
                loaders = chd.loaders.LoaderPool(
                    method_info["loader_cls"],
                    method_info["loader_parameters"],
                    shuffle_on_iter=True,
                    n_workers=10,
                )
                loaders_validation = chd.loaders.LoaderPool(
                    method_info["loader_cls"],
                    method_info["loader_parameters"],
                    n_workers=5,
                )
                loaders_validation.shuffle_on_iter = False

                # model
                model = method_info["model_cls"](**method_info["model_parameters"])

                # optimization
                optimize_every_step = 1
                lr = 1e-2
                optimizer = chd.optim.SparseDenseAdam(
                    model.parameters_sparse(),
                    model.parameters_dense(),
                    lr=lr,
                    weight_decay=1e-5,
                )
                n_epochs = (
                    30 if "n_epoch" not in method_info else method_info["n_epoch"]
                )
                checkpoint_every_epoch = 1

                # train
                from chromatinhd.models.positional.trainer import Trainer

                def paircor(x, y, dim=0, eps=0.1):
                    divisor = (y.std(dim) * x.std(dim)) + eps
                    cor = (
                        (x - x.mean(dim, keepdims=True))
                        * (y - y.mean(dim, keepdims=True))
                    ).mean(dim) / divisor
                    return cor

                loss = lambda x, y: -paircor(x, y).mean() * 100

                if outcome_source == "counts":
                    outcome = transcriptome.X.dense()
                else:
                    outcome = torch.from_numpy(transcriptome.adata.layers["magic"])

                trainer = Trainer(
                    model,
                    loaders,
                    loaders_validation,
                    optim=optimizer,
                    outcome=outcome,
                    loss=loss,
                    checkpoint_every_epoch=checkpoint_every_epoch,
                    optimize_every_step=optimize_every_step,
                    n_epochs=n_epochs,
                    device=device,
                )
                trainer.train(
                    fold["minibatches_train_sets"],
                    fold["minibatches_validation_trace"],
                )

                model = model.to("cpu")
                pickle.dump(
                    model,
                    open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "wb"),
                )

                torch.cuda.empty_cache()
                import gc

                gc.collect()
                torch.cuda.empty_cache()

                import matplotlib.pyplot as plt

                fig, ax = plt.subplots()
                plotdata_validation = (
                    pd.DataFrame(trainer.trace.validation_steps)
                    .groupby("checkpoint")
                    .mean()
                    .reset_index()
                )
                plotdata_train = (
                    pd.DataFrame(trainer.trace.train_steps)
                    .groupby("checkpoint")
                    .mean()
                    .reset_index()
                )
                ax.plot(
                    plotdata_validation["checkpoint"],
                    plotdata_validation["loss"],
                    label="validation",
                )
                # ax.plot(plotdata_train["checkpoint"], plotdata_train["loss"], label = "train")
                ax.legend()
                fig.savefig(str(prediction.path / ("trace_" + str(fold_ix) + ".png")))
                plt.close()

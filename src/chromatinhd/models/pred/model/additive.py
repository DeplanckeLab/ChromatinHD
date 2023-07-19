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

class SineEncoding(torch.nn.Module):
    def __init__(self, n_frequencies):
        super().__init__()

        self.register_buffer(
            "frequencies",
            # torch.tensor([[] * 2 for i in range(1, n_frequencies + 1)]).flatten(-2)
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
        self,
        n_genes,
        mean_gene_expression,
        n_embedding_dimensions=5,
        initialization="ones",
        **kwargs
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
        self.bias1.data = mean_gene_expression.clone().detach().to("cpu")[:, None]

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
        mean_gene_expression,
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
            mean_gene_expression=mean_gene_expression,
            initialization=embedding_to_expression_initialization,
        )

    def forward(self, data):
        fragment_embedding = self.fragment_embedder(data.coordinates, data.genemapping)
        cell_gene_embedding = self.embedding_gene_pooler(
            fragment_embedding, data.local_cellxgene_ix, data.n_cells, data.n_genes
        )
        expression_predicted = self.embedding_to_expression(
            cell_gene_embedding, data.genes_oi_torch
        )
        return expression_predicted

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


class Models():
    folds = 
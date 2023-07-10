"""
- a positional encoding per fragment
- summarizes the encoding using a linear layer to 3 dimensions
- self-attention
- summation over cellxgene
- summation over the component
- gene expression by adding an intercept

Intuitively, for each gene, a fragment at a particular position has a positive or negative impact on expression
This effect is simply summed, without any interactions between positions
"""


import torch
import torch_scatter

from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.models import HybridModel


class FragmentEmbedder(torch.nn.Sequential):
    """
    Dummy embedding of fragments in a single embedding dimension of ones
    """

    def __init__(self):
        self.n_embedding_dimensions = 1
        super().__init__()

    def forward(self, coordinates, gene_ix):
        return torch.ones(
            (*coordinates.shape[:-1], 1), device=coordinates.device, dtype=torch.float
        )


class EmbeddingGenePooler(torch.nn.Module):
    """
    Pools fragments across genes and cells
    """

    def __init__(self, reduce="mean"):
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
        self, n_genes, mean_gene_expression, n_embedding_dimensions=100, **kwargs
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
        self.weight1.data[:] = 1.0

    def forward(self, cell_gene_embedding, gene_ix):
        return (cell_gene_embedding * self.weight1(gene_ix) * 10).sum(-1) + self.bias1(
            gene_ix
        ).squeeze()

    def parameters_sparse(self):
        return [self.bias1.weight, self.weight1.weight]


class Model(torch.nn.Module, HybridModel):
    def __init__(self, n_genes, mean_gene_expression, reduce="sum", window=None):
        super().__init__()

        self.fragment_embedder = FragmentEmbedder()
        self.embedding_gene_pooler = EmbeddingGenePooler(reduce=reduce)
        self.embedding_to_expression = EmbeddingToExpression(
            n_genes=n_genes,
            n_embedding_dimensions=1,
            mean_gene_expression=mean_gene_expression,
        )

        self.window = window

    def forward(self, data):
        if self.window is not None:
            selection = ~(
                (data.coordinates[:, 0] >= self.window[1])
                | (data.coordinates[:, 1] <= self.window[0])
            )
        else:
            selection = slice(None, None)
        coordinates = data.coordinates[selection]
        genemapping = data.genemapping[selection]
        local_cellxgene_ix = data.local_cellxgene_ix[selection]
        fragment_embedding = self.fragment_embedder(coordinates, genemapping)
        cell_gene_embedding = self.embedding_gene_pooler(
            fragment_embedding, local_cellxgene_ix, data.n_cells, data.n_genes
        )
        expression_predicted = self.embedding_to_expression(
            cell_gene_embedding, data.genes_oi_torch
        )
        return expression_predicted

    def forward_multiple(self, data, fragments_oi, extract_total=False):
        fragment_embedding = self.fragment_embedder(data.coordinates, data.genemapping)

        for fragments_oi_ in fragments_oi:
            cell_gene_embedding = self.embedding_gene_pooler(
                fragment_embedding[fragments_oi_],
                data.local_cellxgene_ix[fragments_oi_],
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
                n_fragments_lost = torch.bincount(
                    data.local_cellxgene_ix[~fragments_oi_],
                    minlength=data.n_genes * data.n_cells,
                ).reshape((data.n_cells, data.n_genes))
                yield expression_predicted, n_fragments_lost

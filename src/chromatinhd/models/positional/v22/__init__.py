import torch
import torch_scatter
import math
from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.models import HybridModel


class LinearEmbedded(torch.nn.Module):
    def __init__(self, n_input_features, n_output_features, n_genes):
        super().__init__()
        self.bias1 = EmbeddingTensor(
            n_genes,
            tuple([n_output_features]),
            sparse=True,
        )
        self.bias1.data[:] = 0.0
        self.weight1 = EmbeddingTensor(
            n_genes,
            tuple([n_input_features, n_output_features]),
            sparse=True,
        )
        self.weight1.data[:] = 0.0

    def forward(self, x, gene_ix):
        return torch.einsum("abc,bcd->abd", x, self.weight1(gene_ix)) + self.bias1(
            gene_ix
        )

    def parameters_sparse(self):
        return [self.bias1.weight, self.weight1.weight]


class EmbeddingToExpression(torch.nn.Module):
    """
    Predicts gene expression using a [cell, gene, component] embedding in a gene-specific manner
    """

    def __init__(self, n_genes, n_embedding_dimensions=5, **kwargs):
        self.n_genes = n_genes
        self.n_embedding_dimensions = n_embedding_dimensions

        super().__init__()

        # set bias to empirical mean
        n_intermediate_dimensions = 5

        self.linear1 = LinearEmbedded(
            n_embedding_dimensions, n_intermediate_dimensions, n_genes
        )

        self.linear2 = LinearEmbedded(n_intermediate_dimensions, 1, n_genes)

    def forward(self, cell_gene_embedding, gene_ix):
        out = self.linear1(cell_gene_embedding, gene_ix)
        out = torch.sigmoid(out)
        out = self.linear2(out, gene_ix).squeeze(-1)
        return out

    def parameters_sparse(self):
        return self.linear1.parameters_sparse() + self.linear2.parameters_sparse()


class Model(torch.nn.Module, HybridModel):
    def __init__(
        self,
        base_model,
    ):
        super().__init__()

        self.fragment_embedder = base_model.fragment_embedder
        self.embedding_gene_pooler = base_model.embedding_gene_pooler
        self.embedding_to_expression = base_model.embedding_to_expression
        self.embedding_to_expression2 = EmbeddingToExpression(
            base_model.embedding_to_expression.n_genes,
            base_model.embedding_to_expression.n_embedding_dimensions,
        )

    def forward(self, data):
        fragment_embedding = self.fragment_embedder(data.coordinates, data.genemapping)
        cell_gene_embedding = self.embedding_gene_pooler(
            fragment_embedding, data.local_cellxgene_ix, data.n_cells, data.n_genes
        )
        expression_predicted = self.embedding_to_expression(
            cell_gene_embedding, data.genes_oi_torch
        ) + self.embedding_to_expression2(cell_gene_embedding, data.genes_oi_torch)
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
            ) + self.embedding_to_expression2.forward(
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

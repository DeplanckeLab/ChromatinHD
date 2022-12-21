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
import math

class FixableParameter(torch.nn.Parameter):
    _data = None
    _already_fixed = None
    _prev = None
    def fix(self, fix):
        if (self._data is None) and (self._already_fixed is None):
            self._data = self.data.clone()
            self._already_fixed = fix
        else:
            novel = fix & ~self._already_fixed
            self._data[novel] = self._prev[novel].clone()
            # self._data[novel] = self.data[novel].clone()
            self._already_fixed = fix | self._already_fixed
            self.data[self._already_fixed] = self._data[self._already_fixed].clone()

    def replace(self):
        self._prev = self.data.clone()
        if (self._data is not None) and (self._already_fixed is not None):
            self.data[self._already_fixed] = self._data[self._already_fixed].clone()


class FragmentEmbedder(torch.nn.Sequential):
    """
    Dummy embedding of fragments in a single embedding dimension of ones
    """
    def __init__(self):
        self.n_embedding_dimensions = 1
        super().__init__()
        
    def forward(self, coordinates, gene_ix):
        return torch.ones((*coordinates.shape[:-1], 1), device = coordinates.device, dtype = torch.float)
    
class EmbeddingGenePooler(torch.nn.Module):
    """
    Pools fragments across genes and cells
    """
    def __init__(self, reduce = "mean"):
        self.reduce = reduce
        super().__init__()
    
    def forward(self, embedding, fragment_cellxgene_ix, cell_n, gene_n):
        if self.reduce == "mean":
            cellxgene_embedding = torch_scatter.segment_mean_coo(embedding, fragment_cellxgene_ix, dim_size = cell_n * gene_n)
        elif self.reduce == "sum":
            cellxgene_embedding = torch_scatter.segment_sum_coo(embedding, fragment_cellxgene_ix, dim_size = cell_n * gene_n)
        else:
            raise ValueError()
        cell_gene_embedding = cellxgene_embedding.reshape((cell_n, gene_n, cellxgene_embedding.shape[-1]))
        return cell_gene_embedding
    
class EmbeddingToExpression(torch.nn.Module):
    """
    Predicts gene expression using a [cell, gene, component] embedding in a gene-specific manner
    """
    def __init__(self, n_genes, mean_gene_expression, n_embedding_dimensions = 100, **kwargs):
        self.n_genes = n_genes
        self.n_embedding_dimensions = n_embedding_dimensions
        
        super().__init__()
        
        # set bias to empirical mean
        self.bias1 = FixableParameter(mean_gene_expression.clone().detach().to("cpu"), requires_grad = True)

        self.weight1 = FixableParameter(torch.ones((n_genes, n_embedding_dimensions), requires_grad = True) / 10)
        
    def forward(self, cell_gene_embedding, gene_ix):
        return (cell_gene_embedding * self.weight1[gene_ix] * 10).sum(-1) + self.bias1[gene_ix]
    
class Model(torch.nn.Module):
    def __init__(
        self,
        loader,
        n_genes,
        mean_gene_expression,
        reduce = "sum"
    ):
        super().__init__()
        
        self.fragment_embedder = FragmentEmbedder()
        self.embedding_gene_pooler = EmbeddingGenePooler(reduce = reduce)
        self.embedding_to_expression = EmbeddingToExpression(
            n_genes = n_genes,
            n_embedding_dimensions = 1,
            mean_gene_expression = mean_gene_expression
        )

    def forward(
        self,
        data
    ):
        fragment_embedding = self.fragment_embedder(data.coordinates, data.genemapping)
        cell_gene_embedding = self.embedding_gene_pooler(fragment_embedding, data.local_cellxgene_ix, data.n_cells, data.n_genes)
        expression_predicted = self.embedding_to_expression(cell_gene_embedding, data.genes_oi)
        return expression_predicted
        
    def get_parameters(self):
        return self.parameters()
        # return self.embedding_gene_pooler.parameters()
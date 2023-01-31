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

class FragmentEmbedder(torch.nn.Sequential):
    """
    Embeds individual fragments    
    """
    def __init__(self, window, n_hidden_dimensions = 100, n_embedding_dimensions = 100, **kwargs):
        self.scale = window[1] - window[0]
        self.shift = window[0] + self.scale/2

        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_embedding_dimensions = n_embedding_dimensions
        args = [
            torch.nn.Linear(2, n_hidden_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_dimensions, n_hidden_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_dimensions, n_hidden_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_dimensions, n_embedding_dimensions)
        ]

        super().__init__(*args, **kwargs)
        
    def forward(self, coordinates):
        return super().forward((coordinates - self.shift) / self.scale)
    
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
        self.bias1 = torch.nn.Parameter(mean_gene_expression.clone().detach().to("cpu"), requires_grad = True)

        self.weight1 = torch.nn.Parameter(torch.ones((n_genes, n_embedding_dimensions), requires_grad = True))
        stdv = 1. / math.sqrt(self.weight1.size(1)) / 100
        self.weight1.data.uniform_(-stdv, stdv)
        
    def forward(self, cell_gene_embedding, gene_ix):
        return (cell_gene_embedding * self.weight1[gene_ix]).sum(-1) + self.bias1[gene_ix]
    
class Model(torch.nn.Module):
    def __init__(
        self,
        loader,
        n_genes,
        mean_gene_expression,
        window,
        reduce = "mean",
        n_embedding_dimensions = 100
    ):
        super().__init__()
        
        self.fragment_embedder = FragmentEmbedder(
            window = window,
            n_embedding_dimensions=n_embedding_dimensions
        )
        self.embedding_gene_pooler = EmbeddingGenePooler(reduce = reduce)
        self.embedding_to_expression = EmbeddingToExpression(
            n_genes = n_genes,
            n_embedding_dimensions = self.fragment_embedder.n_embedding_dimensions,
            mean_gene_expression = mean_gene_expression
        )

    def forward(
        self,
        data
    ):
        fragment_embedding = self.fragment_embedder(data.coordinates)
        cell_gene_embedding = self.embedding_gene_pooler(fragment_embedding, data.local_cellxgene_ix, data.n_cells, data.n_genes)
        expression_predicted = self.embedding_to_expression(cell_gene_embedding, data.genes_oi)
        return expression_predicted
        

    def get_parameters(self):
        return self.parameters()
        # return self.embedding_gene_pooler.parameters()
"""
"""


import torch
import torch_scatter
import math
import itertools
    
class EmbeddingGenePooler(torch.nn.Module):
    """
    Pools fragments across genes and cells
    """
    def __init__(self, n_components, debug = False):
        self.debug = debug
        super().__init__()

        self.n_components = n_components

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(n_components, n_components),
            torch.nn.ReLU(),
            torch.nn.Linear(n_components, n_components),
            torch.nn.ReLU(),
            torch.nn.Linear(n_components, 1)
        )
        self.nn[-1].weight.data.zero_()
    
    def forward(self, embedding, fragment_cellxgene_ix, cell_n, gene_n):
        if self.debug:
            assert (torch.diff(fragment_cellxgene_ix) >= 0).all(), "fragment_cellxgene_ix should be sorted"
        embedding = self.nn(embedding)
        cellxgene_embedding = torch_scatter.segment_sum_coo(embedding, fragment_cellxgene_ix, dim_size = cell_n * gene_n)
        cell_gene_embedding = cellxgene_embedding.reshape((cell_n, gene_n, cellxgene_embedding.shape[-1]))
        return cell_gene_embedding
    
class EmbeddingToExpression(torch.nn.Module):
    """
    Predicts gene expression using a [cell, gene, component] embedding in a global non-gene-specific manner (apart from mean_gene_expression)
    """
    def __init__(self, mean_gene_expression, n_components = 100, dummy = False, **kwargs):
        self.n_components = n_components
        
        super().__init__()

        self.dummy = dummy
        
        self.bias1 = torch.nn.Parameter(mean_gene_expression.clone().detach().to("cpu"), requires_grad = True)
        # self.weight1 = torch.nn.Parameter(torch.ones(mean_gene_expression.shape), requires_grad = True)
        
    def forward(self, cell_gene_embedding, gene_ix):
        return cell_gene_embedding.squeeze(-1) + self.bias1[gene_ix]
        # return self.weight1[gene_ix] * cell_gene_embedding.squeeze(-1) + self.bias1[gene_ix]
        # return (torch.ones((cell_gene_embedding.shape[0], 1), device = cell_gene_embedding.device) * self.bias1[gene_ix])
        
    
class FragmentEmbeddingToExpression(torch.nn.Module):
    """
    Predicts gene expression from individual fragments
    """
    def __init__(
        self,
        n_genes,
        mean_gene_expression,
        n_components,
        debug = False, 
        **kwargs
    ):
        super().__init__()
        
        self.embedding_gene_pooler = EmbeddingGenePooler(
            n_components = n_components,
            debug = debug
        )
        self.embedding_to_expression = EmbeddingToExpression(
            n_genes = n_genes,
            n_components = n_components,
            mean_gene_expression = mean_gene_expression
        )
        
    def forward(
        self,
        data
    ):
        cell_gene_embedding = self.embedding_gene_pooler(data.motifcounts, data.local_cellxgene_ix, data.n_cells, data.n_genes)
        expression_predicted = self.embedding_to_expression(cell_gene_embedding, data.genes_oi)
        return expression_predicted

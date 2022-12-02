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
    def __init__(self, n_embedding_dimensions, debug = False):
        self.debug = debug
        super().__init__()

        # self.batchnorm = torch.nn.BatchNorm1d(n_embedding_dimensions)
    
    def forward(self, embedding, fragment_cellxgene_ix, cell_n, gene_n):
        if self.debug:
            assert (torch.diff(fragment_cellxgene_ix) >= 0).all(), "fragment_cellxgene_ix should be sorted"
        # embedding = self.batchnorm(embedding)
        cellxgene_embedding = torch_scatter.segment_sum_coo(embedding, fragment_cellxgene_ix, dim_size = cell_n * gene_n)
        cell_gene_embedding = cellxgene_embedding.reshape((cell_n, gene_n, cellxgene_embedding.shape[-1]))
        return cell_gene_embedding
    
class EmbeddingToExpression(torch.nn.Module):
    """
    Predicts gene expression using a [cell, gene, component] embedding in a global non-gene-specific manner (apart from mean_gene_expression)
    """
    def __init__(self, mean_gene_expression, n_embedding_dimensions = 100, **kwargs):
        self.n_embedding_dimensions = n_embedding_dimensions
        
        super().__init__()
        
        # set bias to empirical mean
        self.bias_normal = mean_gene_expression.clone().detach().to("cpu")
        # self.bias1 = torch.nn.Parameter(mean_gene_expression.clone().detach().to("cpu"), requires_grad = True)
        self.bias1 = mean_gene_expression.clone().detach().to("cpu")

        # default initialization same as a torch.nn.Linear
        self.weight1 = torch.nn.Parameter(torch.ones((n_embedding_dimensions, ), requires_grad = True))
        self.weight1.data.zero_()
        
    def forward(self, cell_gene_embedding, gene_ix):
        return (cell_gene_embedding * self.weight1).sum(-1) + self.bias1[gene_ix]
        # return (torch.ones((cell_gene_embedding.shape[0], 1), device = cell_gene_embedding.device) * self.bias1[gene_ix])
    
class FragmentEmbeddingToExpression(torch.nn.Module):
    """
    Predicts gene expression from individual fragments
    """
    def __init__(
        self,
        n_genes,
        mean_gene_expression,
        n_embedding_dimensions,
        debug = False, 
        **kwargs
    ):
        super().__init__()
        
        self.embedding_gene_pooler = EmbeddingGenePooler(
            n_embedding_dimensions = n_embedding_dimensions,
            debug = debug
        )
        self.embedding_to_expression = EmbeddingToExpression(
            n_genes = n_genes,
            n_embedding_dimensions = n_embedding_dimensions,
            mean_gene_expression = mean_gene_expression
        )
        
    def forward(
        self,
        fragment_embedding,
        fragment_cellxgene_ix,
        cell_n,
        gene_n,
        gene_ix,
    ):
        cell_gene_embedding = self.embedding_gene_pooler(fragment_embedding, fragment_cellxgene_ix, cell_n, gene_n)
        expression_predicted = self.embedding_to_expression(cell_gene_embedding, gene_ix)
        return expression_predicted

    def forward2(
        self,
        split,
        coordinates,
        mapping,
        fragments_oi = None
    ):
        if fragments_oi is not None:
            return self.forward(
                split.motifscores[fragments_oi],
                split.fragment_cellxgene_ix[fragments_oi],
                split.cell_n,
                split.gene_n,
                split.gene_ix,
            )
        else:
            return self.forward(
                split.motifscores,
                split.fragment_cellxgene_ix,
                split.cell_n,
                split.gene_n,
                split.gene_ix,
            )


    def get_parameters(self):
        return itertools.chain(
            self.embedding_gene_pooler.parameters(),
            self.embedding_to_expression.parameters()
        )
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
    
    def forward(self, embedding, fragment_cellxgene_ix, cell_n, gene_n):
        if self.debug:
            assert (torch.diff(fragment_cellxgene_ix) >= 0).all(), "fragment_cellxgene_ix should be sorted"
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
        
        # set bias to empirical mean
        self.bias_normal = mean_gene_expression.clone().detach().to("cpu")
        self.bias1 = torch.nn.Parameter(mean_gene_expression.clone().detach().to("cpu"), requires_grad = True)

        self.directnn = torch.nn.Linear(n_components, 1, bias = False)
        self.directnn.weight.data /= 100

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(n_components, n_components, bias = False),
            torch.nn.ReLU(),
            torch.nn.Linear(n_components, 1, bias = False),
        )
        self.nn[2].weight.data /= 100
        
    def forward(self, cell_gene_embedding, gene_ix):
        if self.dummy:
            return (torch.ones((cell_gene_embedding.shape[0], 1), device = cell_gene_embedding.device) * self.bias1[gene_ix])
        return self.directnn(cell_gene_embedding).sum(-1) + self.nn(cell_gene_embedding).sum(-1) + self.bias1[gene_ix]
        

    @property
    def linear_weight(self):
        return self.directnn.weight
        
    
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
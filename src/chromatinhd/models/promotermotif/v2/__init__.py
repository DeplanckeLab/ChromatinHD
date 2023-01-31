"""
"""


import torch
import torch_scatter
    
class FragmentNumberGenePooler(torch.nn.Module):
    """
    Pools fragments across genes and cells
    """
    def __init__(self, n_components):
        super().__init__()

        self.n_components = n_components

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )
        self.nn[-1].weight.data.zero_()
    
    def forward(self, cellxgene_ix, weights, n_cells, n_genes):
        n_fragments = torch.bincount(cellxgene_ix, minlength = n_cells * n_genes).reshape((n_cells, n_genes, cellxgene_embedding.shape[-1]))
        cellxgene_embedding = self.nn(n_fragments.unsqueeze(-1)).squeeze(-1)
        cellxgene_embedding = cellxgene_embedding.reshape((n_cells, n_genes, cellxgene_embedding.shape[-1]))
        return cellxgene_embedding

class EmbeddingToExpression(torch.nn.Module):
    """
    Predicts gene expression using a [cell, gene, component] embedding in a global non-gene-specific manner (apart from mean_gene_expression)
    """
    def __init__(self, mean_gene_expression, n_components = 100, dummy = False, **kwargs):
        self.n_components = n_components
        
        super().__init__()

        self.dummy = dummy
        
        self.bias1 = torch.nn.Parameter(mean_gene_expression.clone().detach().to("cpu"), requires_grad = True)
        
    def forward(self, cell_gene_embedding, gene_ix):
        return cell_gene_embedding.squeeze(-1) + self.bias1[gene_ix]
    
class FragmentEmbeddingToExpression(torch.nn.Module):
    """
    Predicts gene expression from individual fragments
    """
    distance_weighting = True
    def __init__(
        self,
        n_genes,
        mean_gene_expression,
        n_components,
        **kwargs
    ):
        super().__init__()
        
        self.embedding_gene_pooler = FragmentNumberGenePooler(
        )
        self.embedding_to_expression = EmbeddingToExpression(
            n_genes = n_genes,
            n_components = 1,
            mean_gene_expression = mean_gene_expression
        )
        
    def forward(
        self,
        data
    ):
        cell_gene_embedding = self.embedding_gene_pooler(
            embedding = data.motifcounts,
            cellxgene_ix = data.local_cellxgene_ix,
            n_cells = data.n_cells,
            n_genes = data.n_genes,
        )
        expression_predicted = self.embedding_to_expression(cell_gene_embedding, data.genes_oi)
        return expression_predicted

    def get_parameters(self):
        return self.parameters()
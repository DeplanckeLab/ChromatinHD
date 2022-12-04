"""
"""


import torch
import torch_scatter
import math
import itertools

class FragmentSineEncoder(torch.nn.Module):
    def __init__(self, n_frequencies = 5):
        super().__init__()

        self.register_buffer(
            "frequencies",
            torch.tensor([[1 / 1000**(2 * i/n_frequencies)] * 2 for i in range(1, n_frequencies + 1)]).flatten(-2)
        )
        self.register_buffer(
            "shifts",
            torch.tensor([[0, torch.pi/2] for _ in range(1, n_frequencies + 1)]).flatten(-2)
        )

        self.n_features = n_frequencies * 2 * 2

    def forward(self, coordinates):
        encoding = torch.sin((coordinates[..., None] * self.frequencies + self.shifts).flatten(-2))
        return encoding

class FragmentWeighter(torch.nn.Module):
    def __init__(self, n_frequencies = 5):
        super().__init__()

        self.encoder = FragmentSineEncoder(n_frequencies=n_frequencies)

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(self.encoder.n_features, 1)
        )

    def forward(self, coordinates):
        encoding = self.encoder(coordinates)
        weight = self.nn(encoding).squeeze(-1)
        return weight
    
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
            torch.nn.Linear(n_components, 1)
        )
        self.nn[-1].weight.data.zero_()
    
    def forward(self, embedding, cellxgene_ix, weights, n_cells, n_genes):
        if self.debug:
            assert (torch.diff(cellxgene_ix) >= 0).all(), "cellxgene_ix should be sorted"
        if weights is not None:
            embedding = embedding * weights.unsqueeze(-1)
        embedding = self.nn(embedding)
        cellxgene_embedding = torch_scatter.segment_sum_coo(embedding, cellxgene_ix, dim_size = n_cells * n_genes)
        cell_gene_embedding = cellxgene_embedding.reshape((n_cells, n_genes, cellxgene_embedding.shape[-1]))
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
        
    def forward(self, cell_gene_embedding, gene_ix):
        return cell_gene_embedding.squeeze(-1) + self.bias1[gene_ix]
        # return (torch.ones((cell_gene_embedding.shape[0], 1), device = cell_gene_embedding.device) * self.bias1[gene_ix])
    
class FragmentEmbeddingToExpression(torch.nn.Module):
    """
    Predicts gene expression from individual fragments
    """
    weighting = True
    def __init__(
        self,
        n_genes,
        mean_gene_expression,
        n_components,
        weighting = True,
        **kwargs
    ):
        super().__init__()

        self.weighting = weighting

        self.fragment_weighter = FragmentWeighter()
        
        self.embedding_gene_pooler = EmbeddingGenePooler(
            n_components = n_components
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
        if self.weighting:
            fragment_weights = self.fragment_weighter(data.coordinates)
        else:
            fragment_weights = None
        cell_gene_embedding = self.embedding_gene_pooler(
            embedding = data.motifcounts,
            weights = fragment_weights,
            cellxgene_ix = data.local_cellxgene_ix,
            n_cells = data.n_cells,
            n_genes = data.n_genes,
        )
        expression_predicted = self.embedding_to_expression(cell_gene_embedding, data.genes_oi)
        return expression_predicted

    def get_parameters(self):
        return self.parameters()
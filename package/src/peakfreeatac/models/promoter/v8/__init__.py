"""
- a positional encoding per fragment
- summarizes the encoding using a linear layer to 1 dimension
- summation over cellxgene
- but summarizes this encoding using gene-specific parameters in 1 dimension before aggregating for each cellxgene

Intuitively, for each gene, a fragment at a particular position has a positive or negative impact on expression
This effect is simply summed, without any interactions between positions
"""


import torch
import torch_scatter
import math

class FragmentEmbedder(torch.nn.Module):
    def __init__(self, n_genes, n_frequencies = 20, **kwargs):
        self.n_positional_dimensions = n_frequencies * 2 * 2
        self.n_embedding_dimensions = 1
        
        super().__init__(**kwargs)

        self.register_buffer(
            "frequencies",
            torch.tensor([[1 / 100**(2 * i/n_frequencies)] * 2 for i in range(1, n_frequencies + 1)]).flatten(-2)
            # torch.tensor([[i * 2 * torch.pi / 6000] * 2 for i in range(1, n_frequencies + 1)]).flatten(-2)
        )
        self.register_buffer(
            "shifts",
            torch.tensor([[0, torch.pi/2] for i in range(1, n_frequencies + 1)]).flatten(-2)
        )

        # default initialization same as a torch.nn.Linear
        self.weight1 = torch.nn.Parameter(torch.empty((n_genes, self.n_positional_dimensions, self.n_embedding_dimensions), requires_grad = True))
        stdv = 1. / math.sqrt(self.weight1.size(-1)) / 100
        self.weight1.data.uniform_(-stdv, stdv)
    
    def forward(self, coordinates, gene_ix):
        embedding = torch.sin((coordinates[..., None] * self.frequencies + self.shifts).flatten(-2))
        embedding = torch.einsum('ab,abc->a', embedding, self.weight1[gene_ix])

        return embedding
    
class EmbeddingGenePooler(torch.nn.Module):
    """
    Pools fragments across genes and cells
    """
    def __init__(self, debug = False):
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
    Predicts gene expression using a [cell, gene, component] embedding in a gene-specific manner
    """
    def __init__(self, n_genes, mean_gene_expression, n_embedding_dimensions = 100, **kwargs):
        self.n_genes = n_genes
        self.n_embedding_dimensions = n_embedding_dimensions
        
        super().__init__()
        
        # set bias to empirical mean
        self.bias1 = torch.nn.Parameter(mean_gene_expression.clone().detach().to("cpu"), requires_grad = True)
        
    def forward(self, cell_gene_embedding, gene_ix):
        return cell_gene_embedding.mean(-1) + self.bias1[gene_ix]
    
class FragmentsToExpression(torch.nn.Module):
    """
    Predicts gene expression from individual fragments
    """
    def __init__(
        self,
        n_genes,
        mean_gene_expression,
        n_frequencies = 10,
        debug = False, 
        **kwargs
    ):
        super().__init__()
        
        self.fragment_embedder = FragmentEmbedder(
            n_frequencies = n_frequencies,
            n_genes = n_genes
        )
        self.embedding_gene_pooler = EmbeddingGenePooler(debug = debug)
        self.embedding_to_expression = EmbeddingToExpression(
            n_genes = n_genes,
            n_embedding_dimensions = self.fragment_embedder.n_embedding_dimensions,
            mean_gene_expression = mean_gene_expression
        )
        
    def forward(
        self,
        fragment_coordinates,
        fragment_cellxgene_ix,
        fragment_gene_ix,
        cell_n,
        gene_n,
        gene_ix
    ):
        fragment_embedding = self.fragment_embedder(fragment_coordinates, fragment_gene_ix)
        cell_gene_embedding = self.embedding_gene_pooler(fragment_embedding, fragment_cellxgene_ix, cell_n, gene_n)
        expression_predicted = self.embedding_to_expression(cell_gene_embedding, gene_ix)
        return expression_predicted

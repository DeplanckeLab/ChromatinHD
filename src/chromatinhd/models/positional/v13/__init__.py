import torch
import torch_scatter
import math
import numpy as np
import dataclasses
import itertools

class FragmentEmbedder(torch.nn.Sequential):
    """
    Dummy embedding of fragments in a single embedding dimension of ones
    """
    def __init__(self, *args, **kwargs):
        self.n_embedding_dimensions = 1
        super().__init__(*args, **kwargs)
        
    def forward(self, coordinates):
        return torch.ones((*coordinates.shape[:-1], 1), device = coordinates.device, dtype = torch.float)
    
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
        
        # default initialization same as a torch.nn.Linear
        self.weight1 = torch.nn.Parameter(torch.empty((n_genes, n_embedding_dimensions), requires_grad = True))
        # stdv = 1. / math.sqrt(self.weight1.size(1)) / 100
        # self.weight1.data.uniform_(-stdv, stdv)
        self.weight1.data.zero_()
        
        # set bias to empirical mean
        self.bias1 = torch.nn.Parameter(mean_gene_expression.clone().detach().to("cpu"), requires_grad = True)
        
    def forward(self, cell_gene_embedding, gene_ix):
        return (cell_gene_embedding * self.weight1[gene_ix]).sum(-1) + self.bias1[gene_ix]
    
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
        
        self.fragment_embedder = FragmentEmbedder()
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
        fragment_embedding = self.fragment_embedder(fragment_coordinates)
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
                coordinates[split.fragments_selected][fragments_oi],
                split.fragment_cellxgene_ix[fragments_oi],
                mapping[split.fragments_selected, 1][fragments_oi],
                split.cell_n,
                split.gene_n,
                split.gene_ix,
            )
        else:
            return self.forward(
                coordinates[split.fragments_selected],
                split.fragment_cellxgene_ix,
                mapping[split.fragments_selected, 1],
                split.cell_n,
                split.gene_n,
                split.gene_ix
            )


    def get_parameters(self):
        return itertools.chain(
            self.fragment_embedder.parameters(),
            self.embedding_gene_pooler.parameters(),
            self.embedding_to_expression.parameters()
        )
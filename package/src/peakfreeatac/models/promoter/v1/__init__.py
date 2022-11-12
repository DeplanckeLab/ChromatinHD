import torch
import torch_scatter
import math
import numpy as np
import dataclasses

class FragmentEmbedder(torch.nn.Sequential):
    """
    Embeds individual fragments    
    """
    def __init__(self, n_virtual_dimensions = 100, n_embedding_dimensions = 100, **kwargs):
        self.n_virtual_dimensions = n_virtual_dimensions
        self.n_embedding_dimensions = n_embedding_dimensions
        args = [
            torch.nn.Linear(2, n_virtual_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(n_virtual_dimensions, n_virtual_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(n_virtual_dimensions, n_virtual_dimensions),
            torch.nn.ReLU(),
            torch.nn.Linear(n_virtual_dimensions, n_embedding_dimensions)
        ]
        super().__init__(*args, **kwargs)
        
    def forward(self, coordinates):
        return super().forward(coordinates.float()/1000)
    
class FragmentEmbedderCounter(torch.nn.Sequential):
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
    
    def forward(self, embedding, fragment_cellxgene_idx, cell_n, gene_n):
        if self.debug:
            assert (torch.diff(fragment_cellxgene_idx) >= 0).all(), "fragment_cellxgene_idx should be sorted"
        cellxgene_embedding = torch_scatter.segment_mean_coo(embedding, fragment_cellxgene_idx, dim_size = cell_n * gene_n)
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
        stdv = 1. / math.sqrt(self.weight1.size(1)) / 100
        self.weight1.data.uniform_(-stdv, stdv)
        
        # set bias to empirical mean
        self.bias1 = torch.nn.Parameter(mean_gene_expression.clone().detach().to("cpu"), requires_grad = True)
        
    def forward(self, cell_gene_embedding, gene_idx):
        return (cell_gene_embedding * self.weight1[gene_idx]).sum(-1) + self.bias1[gene_idx]
    
class EmbeddingToExpressionBias(EmbeddingToExpression):
    """
    Dummy method for predicting the gene expression using a [cell, gene, component] embedding, by only including the bias
    """
    def forward(self, cell_gene_embedding, gene_idx):
        return (torch.ones((cell_gene_embedding.shape[0], 1), device = cell_gene_embedding.device) * self.bias1[gene_idx])
    
class FragmentsToExpression(torch.nn.Module):
    """
    Predicts gene expression from individual fragments
    """
    def __init__(
        self,
        n_genes,
        mean_gene_expression,
        n_embedding_dimensions = 100,
        n_virtual_dimensions = 100,
        debug = False, 
        **kwargs
    ):
        super().__init__()
        
        self.fragment_embedder = FragmentEmbedder(
            n_virtual_dimensions = n_virtual_dimensions,
            n_embedding_dimensions = n_embedding_dimensions
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
        fragment_cellxgene_idx,
        cell_n,
        gene_n,
        gene_idx
    ):
        fragment_embedding = self.fragment_embedder(fragment_coordinates)
        cell_gene_embedding = self.embedding_gene_pooler(fragment_embedding, fragment_cellxgene_idx, cell_n, gene_n)
        expression_predicted = self.embedding_to_expression(cell_gene_embedding, gene_idx)
        return expression_predicted

from typing import Optional


@dataclasses.dataclass
class Split():
    cell_idx:slice
    gene_idx:slice
    phase:int

    def __init__(self, cell_idx, gene_idx, phase="train"):
        assert isinstance(cell_idx, slice)
        assert isinstance(gene_idx, slice)
        self.cell_idx = cell_idx
        self.gene_idx = gene_idx

        self.phase = phase

    def populate(self, fragments):
        self.cell_start = self.cell_idx.start
        self.cell_stop = self.cell_idx.stop
        self.gene_start = self.gene_idx.start
        self.gene_stop = self.gene_idx.stop

        assert self.gene_stop <= fragments.n_genes
        assert self.cell_stop <= fragments.n_cells

        self.fragments_selected = torch.where(
            (fragments.mapping[:, 0] >= self.cell_start) &
            (fragments.mapping[:, 0] < self.cell_stop) &
            (fragments.mapping[:, 1] >= self.gene_start) &
            (fragments.mapping[:, 1] < self.gene_stop)
        )[0]
        
        self.cell_n = self.cell_stop - self.cell_start
        self.gene_n = self.gene_stop - self.gene_start

        self.fragments_coordinates = fragments.coordinates[self.fragments_selected]
        self.fragments_mappings = fragments.mapping[self.fragments_selected]

        # we should adapt this if the minibatch cells/genes would ever be non-contiguous
        self.local_cell_idx = self.fragments_mappings[:, 0] - self.cell_start
        self.local_gene_idx = self.fragments_mappings[:, 1] - self.gene_start

    @property
    def cell_idxs(self):
        """
        The cell indices within the whole dataset as a numpy array
        """
        return np.arange(self.cell_start, self.cell_stop)

    @property
    def gene_idxs(self):
        """
        The gene indices within the whole dataset as a numpy array
        """
        return np.arange(self.gene_start, self.gene_stop)
    
    @property
    def fragment_cellxgene_idx(self):
        """
        The local index of cellxgene, i.e. starting from 0 and going up to n_cells * n_genes - 1
        
        """
        return self.local_cell_idx * self.gene_n + self.local_gene_idx
    
    def to(self, device):
        self.fragments_selected = self.fragments_selected.to(device)
        self.fragments_coordinates = self.fragments_coordinates.to(device)
        self.fragments_mappings = self.fragments_mappings.to(device)
        self.local_cell_idx = self.local_cell_idx.to(device)
        self.local_gene_idx = self.local_gene_idx.to(device)
        return self
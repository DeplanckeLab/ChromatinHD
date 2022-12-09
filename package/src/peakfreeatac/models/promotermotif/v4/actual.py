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
    length_weighter = None
    aggregation = "sum"
    def __init__(self, n_components, dummy_motifs = False, normalization = 100, n_layers = 0, weight_lengths = False, aggregation = "sum"):
        super().__init__()

        self.n_components = n_components

        self.dummy_motifs = dummy_motifs

        self.motif_bias = torch.nn.Parameter(torch.zeros(n_components), requires_grad = True)

        self.normalization = normalization

        self.aggregation = aggregation

        if n_layers > 0:
            if n_layers == 1:
                n_hidden_dimension = 10
                self.nn1 = torch.nn.Sequential(
                    torch.nn.Linear(n_components, n_hidden_dimension),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_dimension, 1)
                )
            elif n_layers == 2:
                n_hidden_dimension = 10
                self.nn1 = torch.nn.Sequential(
                    torch.nn.Linear(n_components, n_hidden_dimension),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_dimension, n_hidden_dimension),
                    torch.nn.ReLU(),
                    torch.nn.Linear(n_hidden_dimension, 1)
                )
        else:
            self.nn1 = torch.nn.Sequential(
                torch.nn.Linear(n_components, 1)
            )

        self.batchnorm = torch.nn.BatchNorm1d(self.n_components)

        self.weight_lengths = weight_lengths
        if weight_lengths is True:
            self.length_weighter = torch.nn.Sequential(
                torch.nn.Linear(1, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 1),
            )
        elif isinstance(weight_lengths, str) and weight_lengths == "feature":
            self.length_weighter = torch.nn.Sequential(
                torch.nn.BatchNorm1d(1),
                torch.nn.Linear(1, 10),
                torch.nn.ReLU(),
                torch.nn.Linear(10, self.n_components),
            )
        elif weight_lengths is False:
            pass
        else:
            raise ValueError("weight_lengths")
    
    def forward(self, embedding, cellxgene_ix, n_cells, n_genes, lengths = None):
        # dummy baseline, set all motifs to 0
        if self.dummy_motifs is True:
            embedding[:] = 0.
        elif (self.dummy_motifs is not False) and isinstance(self.dummy_motifs, int):
            embedding[:] = self.dummy_motifs

        embedding = (self.batchnorm(embedding))

        embedding = embedding + self.motif_bias

        if self.length_weighter is not None:
            embedding = embedding * self.length_weighter(lengths.unsqueeze(-1))

        # transform motif counts to a gene expression effect specific to each fragment based on its motif counts
        fragment_effect = self.nn1(embedding)
        
        # combine effects across fragments for each cellxgene
        if self.aggregation == "mean":
            cellxgene_effect = torch_scatter.segment_sum_coo(fragment_effect, cellxgene_ix, dim_size = n_cells * n_genes)
        else:
            cellxgene_effect = torch_scatter.segment_sum_coo(fragment_effect, cellxgene_ix, dim_size = n_cells * n_genes)
        cell_gene_effect = cellxgene_effect.reshape((n_cells, n_genes))
        return cell_gene_effect
    
class Model(torch.nn.Module):
    def __init__(
        self,
        baseline,
        loader,
        dummy_motifs = False,
        normalization = 100,
        n_layers = 0,
        weight_lengths = False,
        aggregation = "sum",
        **kwargs
    ):
        super().__init__()

        self.baseline = baseline

        n_components = loader.n_features
        
        self.embedding_gene_pooler = EmbeddingGenePooler(
            n_components = n_components,
            dummy_motifs = dummy_motifs,
            normalization = normalization,
            n_layers = n_layers,
            weight_lengths = weight_lengths,
            aggregation=aggregation
        )
        
    def forward(
        self,
        data
    ):
        expression = self.baseline(data)
        expression = expression + self.embedding_gene_pooler(
            embedding = data.motifcounts,
            cellxgene_ix = data.local_cellxgene_ix,
            n_cells = data.n_cells,
            n_genes = data.n_genes,
            lengths = (data.coordinates[:, 1] - data.coordinates[:, 0]).to(torch.float)
        )
        return expression

    def get_parameters(self):
        # return self.parameters()
        return self.embedding_gene_pooler.parameters()
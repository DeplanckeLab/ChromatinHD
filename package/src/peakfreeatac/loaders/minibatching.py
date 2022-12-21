import numpy as np
import itertools
import dataclasses

@dataclasses.dataclass
class Minibatch():
    cells_oi:np.ndarray
    genes_oi:np.ndarray
    cellxgene_oi:np.ndarray
    phase:str = "train"

    def items(self):
        return {
            "cells_oi":self.cells_oi,
            "genes_oi":self.genes_oi
        }

    def filter_genes(self, genes):
        genes_oi = genes[self.genes_oi],
        
        return Minibatch(

        )


def cell_gene_to_cellxgene(cells_oi, genes_oi, n_genes):
    return (cells_oi[:, None] * n_genes + genes_oi).flatten()

def create_bins_ordered(cells, genes, n_genes_total, n_genes_step = 300, n_cells_step = 1000, use_all = False, rg = None, **kwargs):
    """
    Creates bins of cellxgene
    A number of cell and gene bins are created first, and all combinations of these bins make up the 
    """
    if rg is None:
        rg = np.random.RandomState()
    cells = rg.permutation(cells)
    genes = rg.permutation(genes)

    gene_cuts = [*np.arange(0, len(genes), step = n_genes_step)]
    if use_all:
        gene_cuts.append(len(genes))
    gene_bins = [genes[a:b] for a, b in zip(gene_cuts[:-1], gene_cuts[1:])]

    cell_cuts = [*np.arange(0, len(cells), step = n_cells_step)]
    if use_all:
        cell_cuts.append(len(cells))
    cell_bins = [cells[a:b] for a, b in zip(cell_cuts[:-1], cell_cuts[1:])]

    bins = []
    for cells_oi, genes_oi in itertools.product(cell_bins, gene_bins):
        bins.append(Minibatch(cells_oi = cells_oi, genes_oi = genes_oi, cellxgene_oi = cell_gene_to_cellxgene(cells_oi, genes_oi, n_genes_total), **kwargs))
    return bins

def create_bins_random(cells, genes, n_genes_total, n_genes_step = 300, n_cells_step = 1000, use_all = False, rg = None, **kwargs):
    """
    Creates bins of cellxgene
    Within each cell bin, the genes are put in a new set of random bins
    """
    if rg is None:
        rg = np.random.RandomState()
    cells = rg.permutation(cells)

    cell_cuts = [*np.arange(0, len(cells), step = n_cells_step)]
    if use_all:
        cell_cuts.append(len(cells))
    cell_bins = [cells[a:b] for a, b in zip(cell_cuts[:-1], cell_cuts[1:])]

    bins = []
    for cells_oi in cell_bins:
        genes = rg.permutation(genes)
        gene_cuts = [*np.arange(0, len(genes), step = n_genes_step)]
        gene_bins = [genes[a:b] for a, b in zip(gene_cuts[:-1], gene_cuts[1:])]
        
        for genes_oi in gene_bins:
            bins.append(Minibatch(cells_oi = cells_oi, genes_oi = genes_oi, cellxgene_oi = cell_gene_to_cellxgene(cells_oi, genes_oi, n_genes_total), **kwargs))
    return bins
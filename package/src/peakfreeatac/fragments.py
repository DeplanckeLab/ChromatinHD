import numpy as np
import pandas as pd
import pickle

import pathlib

from peakfreeatac.flow import Flow

import dataclasses
import functools
import torch

class Fragments(Flow):
    _coordinates = None
    @property
    def coordinates(self):
        if self._coordinates is None:
            self._coordinates = pickle.load((self.path / "coordinates.pkl").open("rb"))
        return self._coordinates
    @coordinates.setter
    def coordinates(self, value):
        pickle.dump(value, (self.path / "coordinates.pkl").open("wb"))
        self._coordinates = value

    _mapping = None
    @property
    def mapping(self):
        if self._mapping is None:
            self._mapping = pickle.load((self.path / "mapping.pkl").open("rb"))
        return self._mapping
    @mapping.setter
    def mapping(self, value):
        pickle.dump(value, (self.path / "mapping.pkl").open("wb"))
        self._mapping = value

    @property
    def var(self):
        return pd.read_table(self.path / "var.tsv", index_col = 0)
    @var.setter
    def var(self, value):
        value.index.name = "gene"
        value.to_csv(self.path / "var.tsv", sep = "\t")

    @property
    def obs(self):
        return pd.read_table(self.path / "obs.tsv", index_col = 0)
    @obs.setter
    def obs(self, value):
        value.index.name = "cell"
        value.to_csv(self.path / "obs.tsv", sep = "\t")

    _n_genes = None
    @property
    def n_genes(self):
        if self._n_genes is None:
            self._n_genes = self.var.shape[0]
        return self._n_genes
    _n_cells = None
    @property
    def n_cells(self):
        if self._n_cells is None:
            self._n_cells = self.obs.shape[0]
        return self._n_cells



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
    
    @functools.cached_property
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
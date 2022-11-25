import numpy as np
import pandas as pd
import pickle

import pathlib

from peakfreeatac.flow import Flow

import dataclasses
import functools
import itertools
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

    _cell_fragment_mapping = None
    @property
    def cell_fragment_mapping(self):
        if self._cell_fragment_mapping is None:
            self._cell_fragment_mapping = pickle.load((self.path / "cell_fragment_mapping.pkl").open("rb"))
        return self._cell_fragment_mapping
    @cell_fragment_mapping.setter
    def cell_fragment_mapping(self, value):
        pickle.dump(value, (self.path / "cell_fragment_mapping.pkl").open("wb"))
        self._cell_fragment_mapping = value

    def create_cell_fragment_mapping(self):
        cell_fragment_mapping = [[] for i in range(self.n_cells)]
        cur_cell_ix = -1
        for fragment_ix, cell_ix in enumerate(self.mapping[:, 0]):
            if cell_ix > cur_cell_ix:
                cur_cell_ix = cell_ix.item()
            cell_fragment_mapping[cur_cell_ix].append(fragment_ix)
        self.cell_fragment_mapping = cell_fragment_mapping



@dataclasses.dataclass
class Split():
    cell_ix:torch.Tensor
    gene_ix:slice
    phase:int

    def __init__(self, cell_ix, gene_ix, phase="train"):
        assert isinstance(cell_ix, torch.Tensor)
        assert isinstance(gene_ix, slice)
        self.cell_ix = cell_ix
        self.gene_ix = gene_ix

        self.phase = phase

    def populate(self, fragments):
        self.gene_start = self.gene_ix.start
        self.gene_stop = self.gene_ix.stop

        assert self.gene_stop <= fragments.n_genes

        fragments_selected = torch.cat([torch.tensor(fragments.cell_fragment_mapping[cell_ix], dtype = int) for cell_ix in self.cell_ix])
        fragments_selected = fragments_selected[(fragments.mapping[fragments_selected, 1] >= self.gene_start) & (fragments.mapping[fragments_selected, 1] < self.gene_stop)]
        self.fragments_selected = fragments_selected
        
        self.cell_n = len(self.cell_ix)
        self.gene_n = self.gene_stop - self.gene_start

        local_cell_ix_mapper = torch.zeros(fragments.n_cells, dtype = int)
        local_cell_ix_mapper[self.cell_ix] = torch.arange(self.cell_n)
        self.local_cell_ix = local_cell_ix_mapper[fragments.mapping[self.fragments_selected, 0]]
        self.local_gene_ix = fragments.mapping[self.fragments_selected, 1] - self.gene_start

    @property
    def cell_ixs(self):
        """
        The cell indices within the whole dataset as a numpy array
        """
        return self.cell_ix.detach().cpu().numpy()

    @property
    def gene_ixs(self):
        """
        The gene indices within the whole dataset as a numpy array
        """
        return np.arange(self.gene_start, self.gene_stop)
    
    _fragment_cellxgene_ix = None
    @property
    def fragment_cellxgene_ix(self):
        """
        The local index of cellxgene, i.e. starting from 0 and going up to n_cells * n_genes - 1
        """
        if self._fragment_cellxgene_ix is None:
            self._fragment_cellxgene_ix = self.local_cell_ix * self.gene_n + self.local_gene_ix
            
        return self._fragment_cellxgene_ix
    
    def to(self, device):
        self.fragments_selected = self.fragments_selected.to(device)
        self.local_cell_ix = self.local_cell_ix.to(device)
        self.local_gene_ix = self.local_gene_ix.to(device)

        if self._fragment_cellxgene_ix is not None:
            self._fragment_cellxgene_ix = self._fragment_cellxgene_ix.to(device)
        return self


@dataclasses.dataclass
class Fold():
    _splits = None

    cells_train = None
    cells_validation = None
    def __init__(self, cells_train, cells_validation, n_cell_step, n_genes, n_gene_step):
        self._splits = []

        self.cells_train = cells_train
        self.cells_validation = cells_validation

        gene_cuts = list(np.arange(n_genes, step = n_gene_step)) + [n_genes]
        gene_bins = [slice(a, b) for a, b in zip(gene_cuts[:-1], gene_cuts[1:])]
        cell_cuts_train = [*np.arange(0, len(cells_train), step = n_cell_step)] + [len(cells_train)]
        cell_bins_train = [cells_train[a:b] for a, b in zip(cell_cuts_train[:-1], cell_cuts_train[1:])]

        bins_train = list(itertools.product(cell_bins_train, gene_bins))
        for cells_split, genes_split in bins_train:
            self._splits.append(Split(cells_split, genes_split, phase = "train"))

        cell_cuts_validation = [*np.arange(0, len(cells_validation), step = n_cell_step)] + [len(cells_validation)]
        cell_bins_validation = [cells_validation[a:b] for a, b in zip(cell_cuts_validation[:-1], cell_cuts_validation[1:])]
        bins_validation = list(itertools.product(cell_bins_validation, gene_bins))
        for cells_split, genes_split in bins_validation:
            self._splits.append(Split(cells_split, genes_split, phase = "validation"))


    def __getitem__(self, k):
        return self._splits[k]

    def __setitem__(self, k, v):
        self._splits[k] = v

    def to(self, device):
        self._splits = [split.to(device) for split in self._splits]
        return self

    def populate(self, fragments):
        for split in self._splits:
            split.populate(fragments)


@dataclasses.dataclass
class Folds():
    _folds = []
    def __init__(self, n_cells, n_genes, n_cell_step, n_gene_step, n_folds):
        # mapping_x = fragments.mapping[:, 0] * fragments.n_genes + fragments.mapping[:, 1]

        # first define the different folds
        torch.manual_seed(1)
        cell_ixs = torch.arange(n_cells)[torch.randperm(n_cells)]

        n_train_cells = int(n_cells / n_folds)
        cuts = [i * n_train_cells for i in range(int(n_cells/n_train_cells))] + [9999999999999999999]
        cell_sets = [cell_ixs[i:j] for i, j in zip(cuts[:-1], cuts[1:])]


        # create splits for each fold
        self._folds = []
        for fold_ix in range(n_folds):
            fold_cells_train = torch.cat([cell_ixs for set_ix, cell_ixs in enumerate(cell_sets) if set_ix != fold_ix])
            fold_cells_validation = cell_sets[fold_ix]

            fold = Fold(fold_cells_train, fold_cells_validation, n_cell_step, n_genes, n_gene_step)

            self._folds.append(fold)

    def __getitem__(self, k):
        return self._folds[k]

    def __setitem__(self, k, v):
        self._folds[k] = v

    def populate(self, fragments):
        for fold in self._folds:
            fold.populate(fragments)

    def to(self, device):
        self._folds = [fold.to(device) for fold in self._folds]
        return self

    def __len__(self):
        return len(self._folds)
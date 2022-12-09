import numpy as np
import pandas as pd
import pickle

import pathlib

from peakfreeatac.flow import Flow

import dataclasses
import functools
import itertools
import torch
import tqdm.auto as tqdm

class Fragments(Flow):
    _coordinates = None
    @property
    def coordinates(self):
        if self._coordinates is None:
            self._coordinates = pickle.load((self.path / "coordinates.pkl").open("rb"))
        return self._coordinates
    @coordinates.setter
    def coordinates(self, value):
        value = value.to(torch.int64).contiguous()
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
        value = value.to(torch.int64).contiguous()
        pickle.dump(value, (self.path / "mapping.pkl").open("wb"))
        self._mapping = value

    _cellxgene_indptr = None
    @property
    def cellxgene_indptr(self):
        if self._cellxgene_indptr is None:
            self._cellxgene_indptr = pickle.load((self.path / "cellxgene_indptr.pkl").open("rb"))
        return self._cellxgene_indptr
    @cellxgene_indptr.setter
    def cellxgene_indptr(self, value):
        value = value.to(torch.int64).contiguous()
        pickle.dump(value, (self.path / "cellxgene_indptr.pkl").open("wb"))
        self._cellxgene_indptr = value

    _genemapping = None
    @property
    def genemapping(self):
        if self._genemapping is not None:
            self._genemapping = self.mapping[:, 1].contiguous()
        return self._genemapping

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

    def estimate_fragment_per_cellxgene(self):
        return int(self.coordinates.shape[0] / self.n_cells / self.n_genes * 2)


class Split():
    cell_ix:torch.Tensor
    gene_ix:slice
    phase:int

    count_mapper:dict = None

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

        # select fragments
        fragments_selected = torch.where(torch.isin(fragments.mapping[:, 0], self.cell_ix))[0]
        fragments_selected = fragments_selected[(fragments.mapping[fragments_selected, 1] >= self.gene_start) & (fragments.mapping[fragments_selected, 1] < self.gene_stop)]
        self.fragments_selected = fragments_selected
        
        # number of cells/genes
        self.cell_n = len(self.cell_ix)
        self.gene_n = self.gene_stop - self.gene_start

        # local gene/cell ix
        local_cell_ix_mapper = torch.zeros(fragments.n_cells, dtype = int)
        local_cell_ix_mapper[self.cell_ix] = torch.arange(self.cell_n)
        self.local_cell_ix = local_cell_ix_mapper[fragments.mapping[self.fragments_selected, 0]]
        self.local_gene_ix = fragments.mapping[self.fragments_selected, 1] - self.gene_start

        # count mapper
        count_mapper = {}
        padded_fragment_cellxgene_ix = torch.cat([torch.tensor([-1]), self.fragment_cellxgene_ix, torch.tensor([99999999999999999])])
        n = torch.arange(len(self.fragment_cellxgene_ix) + 1)[(padded_fragment_cellxgene_ix.diff() > 0)].diff()
        n_interleaved = torch.repeat_interleave(n, n)
        
        count_mapper[2] = n_interleaved == 2
        
        self.count_mapper = count_mapper

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

        if self.count_mapper is not None:
            for k, v in self.count_mapper.items():
                self.count_mapper[k] = v.to(device)
        return self

class SplitDouble(Split):
    def __init__(self, cell_ix, gene_ix, phase="train"):
        assert isinstance(cell_ix, torch.Tensor)
        assert isinstance(gene_ix, torch.Tensor)
        self.cell_ix = torch.sort(cell_ix)[0]
        self.gene_ix = torch.sort(gene_ix)[0]

        self.phase = phase

    def populate(self, fragments):
        # select fragments
        fragments_selected = torch.where(torch.isin(fragments.mapping[:, 0], self.cell_ix) & torch.isin(fragments.mapping[:, 1], self.gene_ix))[0]
        self.fragments_selected = fragments_selected
        
        # number of cells/genes
        self.cell_n = len(self.cell_ix)
        self.gene_n = len(self.gene_ix)

        # local gene/cell ix
        local_cell_ix_mapper = torch.zeros(fragments.n_cells, dtype = int)
        local_gene_ix_mapper = torch.zeros(fragments.n_genes, dtype = int)
        local_cell_ix_mapper[self.cell_ix] = torch.arange(self.cell_n)
        local_gene_ix_mapper[self.gene_ix] = torch.arange(self.gene_n)
        self.local_cell_ix = local_cell_ix_mapper[fragments.mapping[self.fragments_selected, 0]]
        self.local_gene_ix = local_gene_ix_mapper[fragments.mapping[self.fragments_selected, 1]]

        # count mapper
        # count_mapper = {}
        # padded_fragment_cellxgene_ix = torch.cat([torch.tensor([-1]), self.fragment_cellxgene_ix, torch.tensor([99999999999999999])])
        # n = torch.arange(len(self.fragment_cellxgene_ix) + 1)[(padded_fragment_cellxgene_ix.diff() > 0)].diff()
        # n_interleaved = torch.repeat_interleave(n, n)
        
        # count_mapper[2] = n_interleaved == 2
        
        # self.count_mapper = count_mapper

    @property
    def gene_ixs(self):
        """
        The gene indices within the whole dataset as a numpy array
        """
        return self.gene_ix.detach().cpu().numpy()

    def to(self, device):
        self.gene_ix = self.gene_ix.to(device)
        super().to(device)
        return self

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

    def __len__(self):
        return self._splits.__len__()

    def to(self, device):
        self._splits = [split.to(device) for split in self._splits]
        return self

    def populate(self, fragments):
        for split in tqdm.tqdm(self._splits, leave = False):
            split.populate(fragments)

    def plot(self):
        # go over each split and gather which cells and genes are included
        cell_sets = {}
        gene_sets = {}
        cellxgene_data = []
        for split in self:
            cell_ix = split.cell_ix.numpy()
            gene_ix = split.gene_ix.numpy()
            if cell_ix[0] in cell_sets:
                cell_set_ix = cell_sets[cell_ix[0]]
            else:
                cell_set_ix = cell_sets[cell_ix[0]] = len(cell_sets)
            if gene_ix[0] in gene_sets:
                gene_set_ix = gene_sets[gene_ix[0]]
            else:
                gene_set_ix = gene_sets[gene_ix[0]] = len(gene_sets)
            cellxgene_data.append({"cell_set_ix":cell_set_ix, "gene_set_ix":gene_set_ix, "n_cells":len(cell_ix), "n_genes":len(gene_ix), "phase":split.phase, "n_fragments":split.fragments_selected.sum().item()})
        cellxgene_data = pd.DataFrame(cellxgene_data)

        # aggregate over cell sets and gene sets
        cell_sets_data = cellxgene_data.groupby("cell_set_ix")["n_cells"].first().to_frame()
        cell_sets_data["cum_start"] = np.hstack([[0], np.cumsum(cell_sets_data["n_cells"])[:-1]])
        cell_sets_data["cum_end"] = np.cumsum(cell_sets_data["n_cells"])
        gene_sets_data = cellxgene_data.groupby("gene_set_ix")["n_genes"].first().to_frame()
        gene_sets_data["cum_start"] = np.hstack([[0], np.cumsum(gene_sets_data["n_genes"])[:-1]])
        gene_sets_data["cum_end"] = np.cumsum(gene_sets_data["n_genes"])

        # aggregate over phases
        import seaborn as sns
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        phases = pd.DataFrame({"phase":cellxgene_data["phase"].unique()}).set_index("phase")
        phases["color"] = sns.color_palette(n_colors = phases.shape[0])

        # plot each split
        fig, ax = plt.subplots()
        ax.set_xlim(0, gene_sets_data["cum_end"].max())
        ax.set_ylim(0, cell_sets_data["cum_end"].max())
        ax.set_xlabel("genes")
        ax.set_ylabel("cells")
        for _, row in cellxgene_data.iterrows():
            x, w = gene_sets_data.loc[row["gene_set_ix"], ["cum_start", "n_genes"]]
            y, h = cell_sets_data.loc[row["cell_set_ix"], ["cum_start", "n_cells"]]
            rect = mpl.patches.Rectangle((x, y), w, h, fc = phases.loc[row["phase"], "color"])
            ax.add_patch(rect)

            ax.text(x + w/2, y + h/2, "{x:.1e}".format(x = row["n_fragments"]), ha = "center", va = "center")

        return fig

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

class FoldsDouble():
    _folds = []
    def __init__(self, n_cells, n_genes, n_cell_step, n_folds, perc_train = None):
        # mapping_x = fragments.mapping[:, 0] * fragments.n_genes + fragments.mapping[:, 1]

        # first define the different folds
        torch.manual_seed(1)
        cell_ixs = torch.arange(n_cells)[torch.randperm(n_cells)]
        gene_ixs = torch.arange(n_genes)[torch.randperm(n_genes)]

        if perc_train is None:
            perc_train = 1/n_folds

        assert perc_train <= 1/n_folds
        assert perc_train > 0

        n_cut_cells = int(n_cells * (1-perc_train))
        cuts = [i * n_cut_cells for i in range(int(n_cells/n_cut_cells))] + [9999999999999999999]
        cell_sets = [cell_ixs[i:j] for i, j in zip(cuts[:-1], cuts[1:])]

        n_cut_genes = int(n_genes * (1-perc_train))
        cuts = [i * n_cut_genes for i in range(int(n_genes/n_cut_genes))] + [9999999999999999999]
        gene_sets = [gene_ixs[i:j] for i, j in zip(cuts[:-1], cuts[1:])]

        # create splits for each fold
        self._folds = []
        for fold_ix in range(n_folds):
            fold_cells_train = torch.cat([cell_ixs for set_ix, cell_ixs in enumerate(cell_sets) if set_ix != fold_ix])
            fold_cells_validation = cell_sets[fold_ix]

            fold_genes_train = torch.cat([gene_ixs for set_ix, gene_ixs in enumerate(gene_sets) if set_ix != fold_ix])
            fold_genes_validation = gene_sets[fold_ix]

            fold = FoldDouble(fold_cells_train, fold_cells_validation, fold_genes_train, fold_genes_validation, n_cell_step)

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

class FoldDouble(Fold):
    _splits = None

    cells_train = None
    cells_validation = None
    def __init__(self, cells_train, cells_validation, genes_train, genes_validation, n_cell_step):
        self._splits = []

        self.cells_train = cells_train
        self.cells_validation = cells_validation
        self.genes_train = genes_train
        self.genes_validation = genes_validation

        cell_cuts_train = [*np.arange(0, len(cells_train), step = n_cell_step)] + [len(cells_train)]
        cell_bins_train = [cells_train[a:b] for a, b in zip(cell_cuts_train[:-1], cell_cuts_train[1:])]

        cell_cuts_validation = [*np.arange(0, len(cells_validation), step = n_cell_step)] + [len(cells_validation)]
        cell_bins_validation = [cells_validation[a:b] for a, b in zip(cell_cuts_validation[:-1], cell_cuts_validation[1:])]

        bins_train = list(itertools.product(cell_bins_train, [genes_train]))
        for cells_split, genes_split in bins_train:
            self._splits.append(SplitDouble(cells_split, genes_split, phase = "train"))

        bins_validation = list(itertools.product(cell_bins_validation, [genes_validation]))
        for cells_split, genes_split in bins_validation:
            self._splits.append(SplitDouble(cells_split, genes_split, phase = "validation"))

        bins_validation = list(itertools.product(cell_bins_validation, [genes_train]))
        for cells_split, genes_split in bins_validation:
            self._splits.append(SplitDouble(cells_split, genes_split, phase = "validation_cells"))

        bins_validation = list(itertools.product(cell_bins_train, [genes_validation]))
        for cells_split, genes_split in bins_validation:
            self._splits.append(SplitDouble(cells_split, genes_split, phase = "validation_genes"))


    def __getitem__(self, k):
        return self._splits[k]

    def __setitem__(self, k, v):
        self._splits[k] = v

    def to(self, device):
        self._splits = [split.to(device) for split in self._splits]
        return self

    def populate(self, fragments):
        for split in tqdm.tqdm(self._splits, leave = False):
            split.populate(fragments)

import numpy as np
import pandas as pd
import pickle

from chromatinhd.flow import Flow

import torch
import math


class Fragments(Flow):
    _coordinates = None

    @property
    def coordinates(self):
        if self._coordinates is None:
            self._coordinates = pickle.load((self.path / "coordinates.pkl").open("rb"))
        if not self._coordinates.dtype is torch.int64:
            self._coordinates = self._coordinates.to(torch.int64)
        if not self._coordinates.is_contiguous():
            self._coordinates = self._coordinates.contiguous()
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
        if not self._mapping.dtype is torch.int64:
            self._mapping = self._mapping.to(torch.int64)
        if not self._mapping.is_contiguous():
            self._mapping = self._mapping.contiguous()
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
            self._cellxgene_indptr = pickle.load(
                (self.path / "cellxgene_indptr.pkl").open("rb")
            )
        return self._cellxgene_indptr

    @cellxgene_indptr.setter
    def cellxgene_indptr(self, value):
        value = value.to(torch.int64).contiguous()
        pickle.dump(value, (self.path / "cellxgene_indptr.pkl").open("wb"))
        self._cellxgene_indptr = value

    def create_cellxgene_indptr(self):
        import torch_sparse

        cellxgene = self.mapping[:, 0] * self.n_genes + self.mapping[:, 1]

        if not (cellxgene.diff() >= 0).all():
            raise ValueError(
                "Fragments should be ordered by cell then gene (ascending)"
            )

        n_cellxgene = self.n_genes * self.n_cells
        cellxgene_indptr = torch.ops.torch_sparse.ind2ptr(cellxgene, n_cellxgene)
        assert self.coordinates.shape[0] == cellxgene_indptr[-1]
        if not (cellxgene_indptr.diff() >= 0).all():
            raise ValueError(
                "Fragments should be ordered by cell then gene (ascending)"
            )
        self.cellxgene_indptr = cellxgene_indptr

    _genemapping = None

    @property
    def genemapping(self):
        if self._genemapping is None:
            self._genemapping = self.mapping[:, 1].contiguous()
        return self._genemapping

    _cellmapping = None

    @property
    def cellmapping(self):
        if self._cellmapping is None:
            self._cellmapping = self.mapping[:, 0].contiguous()
        return self._cellmapping

    _var = None

    @property
    def var(self):
        if self._var is None:
            self._var = pd.read_table(self.path / "var.tsv", index_col=0)
        return self._var

    @var.setter
    def var(self, value):
        value.index.name = "gene"
        value.to_csv(self.path / "var.tsv", sep="\t")
        self._var = value

    _obs = None

    @property
    def obs(self):
        if self._obs is None:
            self._obs = pd.read_table(self.path / "obs.tsv", index_col=0)
        return self._obs

    @obs.setter
    def obs(self, value):
        value.index.name = "gene"
        value.to_csv(self.path / "obs.tsv", sep="\t")
        self._obs = value

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

    @property
    def local_cellxgene_ix(self):
        return self.cellmapping * self.n_genes + self.genemapping

    def estimate_fragment_per_cellxgene(self):
        return math.ceil(self.coordinates.shape[0] / self.n_cells / self.n_genes * 2)

    def create_cut_data(self):
        cut_coordinates = self.coordinates.flatten()
        cut_coordinates = (cut_coordinates - self.window[0]) / (
            self.window[1] - self.window[0]
        )
        keep_cuts = (cut_coordinates >= 0) & (cut_coordinates <= 1)
        cut_coordinates = cut_coordinates[keep_cuts]

        self.cut_coordinates = cut_coordinates

        self.cut_local_gene_ix = self.genemapping.expand(2, -1).T.flatten()[keep_cuts]
        self.cut_local_cell_ix = self.cellmapping.expand(2, -1).T.flatten()[keep_cuts]

    @property
    def genes_oi_torch(self):
        return torch.from_numpy(self.genes_oi).to(self.coordinates.device)

    @property
    def cells_oi_torch(self):
        return torch.from_numpy(self.genes_oi).to(self.coordinates.device)

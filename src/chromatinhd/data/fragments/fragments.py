import numpy as np
import pandas as pd
import pickle

from chromatinhd.flow import Flow, StoredTorchInt32, Stored, StoredTorchInt64, TSV

import torch
import math


class Fragments(Flow):
    coordinates = StoredTorchInt64("coordinates")
    mapping = StoredTorchInt64("mapping")
    cellxgene_indptr = StoredTorchInt64("cellxgene_indptr")

    def create_cellxgene_indptr(self):
        cellxgene = self.mapping[:, 0] * self.n_genes + self.mapping[:, 1]

        if not (cellxgene.diff() >= 0).all():
            raise ValueError(
                "Fragments should be ordered by cell then gene (ascending)"
            )

        n_cellxgene = self.n_genes * self.n_cells
        cellxgene_indptr = torch.nn.functional.pad(
            torch.cumsum(torch.bincount(cellxgene, minlength=n_cellxgene), 0), (1, 0)
        )
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

    var = TSV("var")
    obs = TSV("obs")

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


class ChunkedFragments(Flow):
    chunk_size = Stored("chunk_size")
    chunkcoords = StoredTorchInt64("chunkcoords")
    chunkcoords_indptr = StoredTorchInt32("chunkcoords_indptr")
    clusters = StoredTorchInt32("clusters")
    relcoords = StoredTorchInt32("relcoords")

    clusters = Stored("clusters")
    clusters_info = Stored("clusters_info")
    chromosomes = Stored("chromosomes")

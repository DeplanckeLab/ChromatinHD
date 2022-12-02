import sys
import numpy as np
import pyximport; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))
import peakfreeatac.loaders.extraction.fragments
import peakfreeatac.loaders.extraction.motifs

import torch
from multiprocessing import shared_memory

import dataclasses

from .fragments import Fragments

class Result():
    @property
    def n_cells(self):
        return len(self.cells_oi)

    @property
    def n_genes(self):
        return len(self.genes_oi)

@dataclasses.dataclass
class FullResult(Result):
    motifcounts:np.ndarray
    local_cellxgene_ix:np.ndarray
    cells_oi:np.ndarray
    genes_oi:np.ndarray

class Full(Fragments):
    def __init__(self, fragments, motifscores, cellxgene_batch_size, window, cutwindow):
        super().__init__(fragments, cellxgene_batch_size, window)
        
        # store auxilliary information
        self.cutwindow = cutwindow
        self.cutwindow_width = cutwindow[1] - cutwindow[0]
        
        # create buffers for motifs
        n_motifs = motifscores.shape[1]
        n_motifs_per_fragment = 1000 # 400 motifs
        motif_buffer_size = self.fragment_buffer_size * n_motifs_per_fragment
        
        self.motifscores_indptr = motifscores.indptr.astype(np.int64)
        self.motifscores_indices = motifscores.indices.astype(np.int64)
        self.motifscores_data = motifscores.data.astype(np.float64)

        self.out_fragment_indptr = torch.from_numpy(np.zeros(motif_buffer_size, dtype = int))#.pin_memory()
        self.out_motif_ix = torch.from_numpy(np.zeros(motif_buffer_size, dtype = int))#.pin_memory()
        self.out_score = torch.from_numpy(np.zeros(motif_buffer_size, dtype = np.float64))#.pin_memory()
        self.out_distance = torch.from_numpy(np.zeros(motif_buffer_size, dtype = int))#.pin_memory()
        self.out_motifcounts = torch.from_numpy(np.ascontiguousarray(np.zeros((self.fragment_buffer_size, n_motifs), dtype = int)))#.pin_memory()
        
    def load(self, cellxgene_oi, **kwargs):
        super().load(cellxgene_oi)

        n_fragments = self.out_coordinates.shape[0]
        
        self.out_motifcounts.data.zero_() # this is a big slow down (~20% of function cpu time) but unsure how to fix
        
        n_motifs = peakfreeatac.loaders.extraction.motifs.extract_all(
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.motifscores_indptr,
            self.motifscores_indices,
            self.motifscores_data,
            *self.window,
            self.window_width,
            *self.cutwindow,
            self.out_fragment_indptr.numpy(),
            self.out_motif_ix.numpy(),
            self.out_score.numpy(),
            self.out_distance.numpy(),
            self.out_motifcounts.numpy()
        )
        self.out_fragment_indptr.resize_(n_fragments+1)
        self.out_motif_ix.resize_(n_motifs)
        self.out_score.resize_(n_motifs)
        self.out_distance.resize_(n_motifs)
        self.out_motifcounts.resize_((n_fragments, self.out_motifcounts.shape[1]))
        
        return FullResult(self.out_motifcounts, self.out_local_cellxgene_ix, **kwargs)

@dataclasses.dataclass
class MotifcountsResult(Result):
    motifcounts:np.ndarray
    local_cellxgene_ix:np.ndarray
    cells_oi:np.ndarray
    genes_oi:np.ndarray
    n_fragments:int

class Motifcounts(Fragments):
    def __init__(self, fragments, motifscores, cellxgene_batch_size, window, cutwindow):
        super().__init__(fragments, cellxgene_batch_size, window)
        
        # store auxilliary information
        self.cutwindow = cutwindow
        self.cutwindow_width = cutwindow[1] - cutwindow[0]
        
        # create buffers for motifs
        self.n_motifs = motifscores.shape[1]
        
        self.motifscores_indptr = motifscores.indptr.astype(np.int64).copy()
        self.motifscores_indices = motifscores.indices.astype(np.int64).copy()
        self.motifscores_data = motifscores.data.astype(np.float64).copy()

        # self.out_motifcounts = torch.from_numpy(np.ascontiguousarray(np.zeros((self.fragment_buffer_size, n_motifs), dtype = int)))#.pin_memory()
        self.out_motifcounts = np.ascontiguousarray(np.zeros((self.fragment_buffer_size, self.n_motifs), dtype = int))
        
    def load(self, cellxgene_oi, **kwargs):
        super().load(cellxgene_oi)

        n_fragments = self.out_coordinates.shape[0]

        out_motifcounts = np.zeros((self.fragment_buffer_size, self.n_motifs), dtype = np.int64)
        
        n_motifs = peakfreeatac.loaders.extraction.motifs.extract_motifcounts(
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.motifscores_indptr,
            self.motifscores_indices,
            self.motifscores_data,
            *self.window,
            self.window_width,
            *self.cutwindow,
            out_motifcounts
        )
        out_motifcounts.resize((n_fragments, self.n_motifs))
        
        return MotifcountsResult(motifcounts = out_motifcounts, local_cellxgene_ix = self.out_local_cellxgene_ix.numpy(), n_fragments = n_fragments, **kwargs)

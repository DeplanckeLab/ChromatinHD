import sys
import numpy as np
import pyximport; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))
import peakfreeatac.loaders.extraction.fragments
import peakfreeatac.loaders.extraction.motifs

import torch

from .fragments import Fragments

class Full(Fragments):
    def __init__(self, fragments, motifscores, cellxgene_batch_size, window, cutwindow):
        super().__init__(fragments, cellxgene_batch_size, window)
        self.cellxgene_batch_size = cellxgene_batch_size
        
        # store auxilliary information
        self.cutwindow = cutwindow
        self.cutwindow_width = cutwindow[1] - cutwindow[0]
        
        # store fragment data
        self.cellxgene_indptr = fragments.cellxgene_indptr.numpy().astype(np.int64)
        self.coordinates = fragments.coordinates.numpy().astype(np.int64)
        self.genemapping = fragments.mapping[:, 1].numpy().astype(np.int64)
        
        # create buffers for coordinates
        n_fragment_per_cellxgene = 2
        fragment_buffer_size = cellxgene_batch_size * n_fragment_per_cellxgene

        self.out_coordinates = torch.from_numpy(np.zeros((fragment_buffer_size, 2), dtype = np.int64))#.pin_memory()
        self.out_genemapping = torch.from_numpy(np.zeros(fragment_buffer_size, dtype = np.int64))#.pin_memory()
        self.out_local_cellxgene_ix = torch.from_numpy(np.zeros(fragment_buffer_size, dtype = np.int64))#.pin_memory()
        
        # create buffers for motifs
        n_motifs = motifscores.shape[1]
        n_motifs_per_fragment = 1000 # 400 motifs
        motif_buffer_size = fragment_buffer_size * n_motifs_per_fragment
        
        self.motifscores_indptr = motifscores.indptr.astype(np.int64)
        self.motifscores_indices = motifscores.indices.astype(np.int64)
        self.motifscores_data = motifscores.data.astype(np.float64)

        self.out_fragment_indptr = torch.from_numpy(np.zeros(motif_buffer_size, dtype = int))#.pin_memory()
        self.out_motif_ix = torch.from_numpy(np.zeros(motif_buffer_size, dtype = int))#.pin_memory()
        self.out_score = torch.from_numpy(np.zeros(motif_buffer_size, dtype = np.float64))#.pin_memory()
        self.out_distance = torch.from_numpy(np.zeros(motif_buffer_size, dtype = int))#.pin_memory()
        self.out_motifcounts = torch.from_numpy(np.ascontiguousarray(np.zeros((fragment_buffer_size, n_motifs), dtype = int)))#.pin_memory()
        
    def load(self, cellxgene_oi):
        super().load(cellxgene_oi)

        n_fragments = self.out_coordinates.shape[0]
        
        self.out_motifcounts.data.zero_() # this is a big slow down (~20% of function cpu time) but unsure how to fix
        
        n_motifs = peakfreeatac.loaders.motifs.extract_full(
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
        
        return self.out_motifcounts, self.out_local_cellxgene_ix

class Motifcounts(Fragments):
    def __init__(self, fragments, motifscores, batch_size, window, cutwindow):
        self.batch_size = batch_size
        
        # store auxilliary information
        self.window = window
        self.cutwindow = cutwindow
        self.window_width = window[1] - window[0]
        self.cutwindow_width = cutwindow[1] - cutwindow[0]
        
        # store fragment data
        self.cellxgene_indptr = fragments.cellxgene_indptr.numpy().astype(np.int64)
        self.coordinates = fragments.coordinates.numpy().astype(np.int64)
        self.genemapping = fragments.mapping[:, 1].numpy().astype(np.int64)
        
        # create buffers for coordinates
        n_fragment_per_cellxgene = 2
        fragment_buffer_size = batch_size * n_fragment_per_cellxgene

        self.out_coordinates = torch.from_numpy(np.zeros((fragment_buffer_size, 2), dtype = np.int64))#.pin_memory()
        self.out_genemapping = torch.from_numpy(np.zeros(fragment_buffer_size, dtype = np.int64))#.pin_memory()
        self.out_local_cellxgene_ix = torch.from_numpy(np.zeros(fragment_buffer_size, dtype = np.int64))#.pin_memory()
        
        # create buffers for motifs
        n_motifs = motifscores.shape[1]
        n_motifs_per_fragment = 1000 # 400 motifs
        motif_buffer_size = fragment_buffer_size * n_motifs_per_fragment
        
        self.motifscores_indptr = motifscores.indptr.astype(np.int64)
        self.motifscores_indices = motifscores.indices.astype(np.int64)
        self.motifscores_data = motifscores.data.astype(np.float64)

        self.out_motifcounts = torch.from_numpy(np.ascontiguousarray(np.zeros((fragment_buffer_size, n_motifs), dtype = int)))#.pin_memory()
        
    def load(self, cellxgene_oi):
        super().load(cellxgene_oi)

        n_fragments = self.out_coordinates.shape[0]
        
        self.out_motifcounts.data.zero_() # this is a big slow down (~20% of function) but unsure how to fix
        
        n_motifs = peakfreeatac.loaders.motifs.extract_motifcounts(
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.motifscores_indptr,
            self.motifscores_indices,
            self.motifscores_data,
            *self.window,
            self.window_width,
            *self.cutwindow,
            self.out_motifcounts.numpy()
        )
        self.out_motifcounts.resize_((n_fragments, self.out_motifcounts.shape[1]))
        
        return self.out_motifcounts, self.out_local_cellxgene_ix
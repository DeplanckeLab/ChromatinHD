import numpy as np
import torch
import pyximport; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))
import peakfreeatac.loaders.extraction.fragments

class Fragments():
    cellxgene_batch_size:int
    
    def __init__(self, fragments, cellxgene_batch_size, window):
        self.cellxgene_batch_size = cellxgene_batch_size
        
        # store auxilliary information
        self.window = window
        self.window_width = window[1] - window[0]
        
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
        
    def load(self, cellxgene_oi):
        assert len(cellxgene_oi) <= self.cellxgene_batch_size
        n_fragments = peakfreeatac.loaders.fragments.extract_fragments(
            cellxgene_oi,
            self.cellxgene_indptr,
            self.coordinates,
            self.genemapping,
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.out_local_cellxgene_ix.numpy()
        )
        self.out_coordinates.resize_((n_fragments, 2))
        self.out_genemapping.resize_((n_fragments))
        self.out_local_cellxgene_ix.resize_((n_fragments))
        
        return self.out_coordinates, self.out_local_cellxgene_ix
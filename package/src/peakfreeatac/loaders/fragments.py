import torch
import numpy as np
import pyximport; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))
import peakfreeatac.loaders.extraction.fragments
import dataclasses

@dataclasses.dataclass
class Result():
    coordinates:torch.Tensor
    local_cellxgene_ix:torch.Tensor
    genemapping:torch.Tensor
    n_fragments:int
    cells_oi:int
    genes_oi:int
    @property
    def n_cells(self):
        return len(self.cells_oi)

    @property
    def n_genes(self):
        return len(self.genes_oi)

    def to(self, device):
        for field_name, field in self.__dataclass_fields__.items():
            if field.type is torch.Tensor:
                self.__setattr__(field_name, self.__getattribute__(field_name).to(device))
        return self

class FragmentsResult(Result):
    pass

class Fragments():
    cellxgene_batch_size:int
    
    def __init__(self, fragments, cellxgene_batch_size, window, n_fragment_per_cellxgene = None):
        self.cellxgene_batch_size = cellxgene_batch_size
        
        # store auxilliary information
        self.window = window
        self.window_width = window[1] - window[0]
        
        # store fragment data
        self.cellxgene_indptr = fragments.cellxgene_indptr.numpy()
        self.coordinates = fragments.coordinates.numpy()
        self.genemapping = fragments.genemapping.numpy()
        
        # create buffers for coordinates
        if n_fragment_per_cellxgene is None:
            fragment_buffer_size = fragments.estimate_fragment_per_cellxgene() * cellxgene_batch_size
        self.fragment_buffer_size = fragment_buffer_size
        
        self.out_coordinates = torch.from_numpy(np.zeros((self.fragment_buffer_size, 2), dtype = np.int64))#.pin_memory()
        self.out_genemapping = torch.from_numpy(np.zeros(self.fragment_buffer_size, dtype = np.int64))#.pin_memory()
        self.out_local_cellxgene_ix = torch.from_numpy(np.zeros(self.fragment_buffer_size, dtype = np.int64))#.pin_memory()
        
    def load(self, minibatch, fragments_oi = None):
        # optional filtering based on fragments_oi
        coordinates = self.coordinates
        genemapping = self.genemapping
        cellxgene_indptr = self.cellxgene_indptr

        # filtering if fragments_oi != None
        if fragments_oi is not None:
            assert (torch.is_tensor(fragments_oi))
            coordinates = coordinates[fragments_oi]
            genemapping = genemapping[fragments_oi]

            # filtering has to be done on indices
            cellxgene_indptr = torch.ops.torch_sparse.ind2ptr(
                torch.ops.torch_sparse.ptr2ind(torch.from_numpy(cellxgene_indptr), cellxgene_indptr[-1]
                )[fragments_oi], 
                len(cellxgene_indptr)
            ).numpy()

        assert len(minibatch.cellxgene_oi) <= self.cellxgene_batch_size
        n_fragments = peakfreeatac.loaders.extraction.fragments.extract_fragments(
            minibatch.cellxgene_oi,
            cellxgene_indptr,
            coordinates,
            genemapping,
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.out_local_cellxgene_ix.numpy()
        )
        if n_fragments > self.fragment_buffer_size:
            raise ValueError("n_fragments is too large for the current buffer size")

        if n_fragments == 0:
            n_fragments = 1
        self.out_coordinates.resize_((n_fragments, 2))
        self.out_genemapping.resize_((n_fragments))
        self.out_local_cellxgene_ix.resize_((n_fragments))
        
        return FragmentsResult(
            local_cellxgene_ix = self.out_local_cellxgene_ix,
            coordinates = self.out_coordinates,
            n_fragments = n_fragments,
            genemapping = self.out_genemapping,
            **minibatch.items()
        )
import torch
import numpy as np

# try to load the shared library
# typically, this will be installed as a python extension
try:
    from . import fragments_helpers  # pylint: disable=C0413,E0611
# however, during developement, we want to load the cython source directly
except ImportError:
    import pyximport

    pyximport.install(
        reload_support=True,
        language_level=3,
        setup_args=dict(include_dirs=[np.get_include()]),
        build_in_temp=False,
    )
    from . import fragments_helpers  # pylint: disable=C0413,E0611

import dataclasses

import chromatinhd.data.fragments


@dataclasses.dataclass
class Result:
    coordinates: torch.Tensor
    local_cellxgene_ix: torch.Tensor
    n_fragments: int
    genemapping: torch.Tensor
    cells_oi: np.ndarray = None
    genes_oi: np.ndarray = None
    window: np.ndarray = None
    n_total_genes: int = None
    localcellxgene_ix: torch.Tensor = None

    @property
    def n_cells(self):
        return len(self.cells_oi)

    @property
    def n_genes(self):
        return len(self.genes_oi)

    def to(self, device):
        for field_name, field in self.__dataclass_fields__.items():
            if field.type is torch.Tensor:
                if self.__getattribute__(field_name) is not None:
                    self.__setattr__(field_name, self.__getattribute__(field_name).to(device))
        return self

    @property
    def local_gene_ix(self):
        return self.local_cellxgene_ix % self.n_genes

    @property
    def local_cell_ix(self):
        return torch.div(self.local_cellxgene_ix, self.n_genes, rounding_mode="floor")

    def filter_fragments(self, fragments_oi):
        assert len(fragments_oi) == self.n_fragments
        return Result(
            coordinates=self.coordinates[fragments_oi],
            local_cellxgene_ix=self.local_cellxgene_ix[fragments_oi],
            genemapping=self.genemapping[fragments_oi],
            n_fragments=fragments_oi.sum(),
            cells_oi=self.cells_oi,
            genes_oi=self.genes_oi,
            window=self.window,
            n_total_genes=self.n_total_genes,
        )


def cell_gene_to_cellxgene(cells_oi, genes_oi, n_genes):
    return (cells_oi[:, None] * n_genes + genes_oi).flatten()


class Fragments:
    cellxgene_batch_size: int

    preloaded = False

    out_coordinates: torch.Tensor
    out_genemapping: torch.Tensor
    out_local_cellxgene_ix: torch.Tensor

    n_genes: int

    def __init__(
        self,
        fragments: chromatinhd.data.fragments.Fragments,
        cellxgene_batch_size: int,
        n_fragment_per_cellxgene: int = None,
    ):
        self.cellxgene_batch_size = cellxgene_batch_size

        # store auxilliary information
        window = fragments.regions.window
        self.window = window
        self.window_width = window[1] - window[0]

        # store fragment data
        self.cellxgene_indptr = fragments.cellxgene_indptr.numpy()
        self.coordinates = fragments.coordinates.numpy()
        self.genemapping = fragments.genemapping.numpy()

        # create buffers for coordinates
        if n_fragment_per_cellxgene is None:
            n_fragment_per_cellxgene = fragments.estimate_fragment_per_cellxgene()
        fragment_buffer_size = n_fragment_per_cellxgene * cellxgene_batch_size
        self.fragment_buffer_size = fragment_buffer_size

        self.n_genes = fragments.n_genes

    def preload(self):
        self.out_coordinates = torch.from_numpy(
            np.zeros((self.fragment_buffer_size, 2), dtype=np.int64)
        )  # .pin_memory()
        self.out_genemapping = torch.from_numpy(np.zeros(self.fragment_buffer_size, dtype=np.int64))  # .pin_memory()
        self.out_local_cellxgene_ix = torch.from_numpy(
            np.zeros(self.fragment_buffer_size, dtype=np.int64)
        )  # .pin_memory()

        self.preloaded = True

    def load(self, minibatch):
        if not self.preloaded:
            self.preload()

        minibatch.cellxgene_oi = cell_gene_to_cellxgene(minibatch.cells_oi, minibatch.genes_oi, self.n_genes)

        assert len(minibatch.cellxgene_oi) <= self.cellxgene_batch_size, (
            len(minibatch.cellxgene_oi),
            self.cellxgene_batch_size,
        )
        n_fragments = fragments_helpers.extract_fragments(
            minibatch.cellxgene_oi,
            self.cellxgene_indptr,
            self.coordinates,
            self.genemapping,
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.out_local_cellxgene_ix.numpy(),
        )
        if n_fragments > self.fragment_buffer_size:
            raise ValueError("n_fragments is too large for the current buffer size")

        if n_fragments == 0:
            n_fragments = 1
        self.out_coordinates.resize_((n_fragments, 2))
        self.out_genemapping.resize_((n_fragments))
        self.out_local_cellxgene_ix.resize_((n_fragments))

        local_cell_ix = torch.div(self.out_local_cellxgene_ix, self.n_genes, rounding_mode="floor")
        localcellxgene_ix = local_cell_ix * self.n_genes + self.out_genemapping

        return Result(
            coordinates=self.out_coordinates,
            local_cellxgene_ix=self.out_local_cellxgene_ix,
            localcellxgene_ix=localcellxgene_ix,
            n_fragments=n_fragments,
            genemapping=self.out_genemapping,
            window=self.window,
            n_total_genes=self.n_genes,
            cells_oi=minibatch.cells_oi,
            genes_oi=minibatch.genes_oi,
        )

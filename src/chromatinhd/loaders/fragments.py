import torch
import numpy as np
import pyximport

pyximport.install(
    reload_support=True,
    language_level=3,
    setup_args=dict(include_dirs=[np.get_include()]),
)
import chromatinhd.data.fragments
import chromatinhd.loaders.extraction.fragments
import dataclasses
from functools import cached_property


@dataclasses.dataclass
class Result:
    coordinates: torch.Tensor
    local_cellxgene_ix: torch.Tensor
    genemapping: torch.Tensor
    n_fragments: int
    cells_oi: np.ndarray
    genes_oi: np.ndarray
    window: np.ndarray
    n_total_genes: int

    @property
    def n_cells(self):
        return len(self.cells_oi)

    @property
    def n_genes(self):
        return len(self.genes_oi)

    def to(self, device):
        for field_name, field in self.__dataclass_fields__.items():
            if field.type is torch.Tensor:
                self.__setattr__(
                    field_name, self.__getattribute__(field_name).to(device)
                )
        return self

    @property
    def local_gene_ix(self):
        return self.local_cellxgene_ix % self.n_genes

    @property
    def local_cell_ix(self):
        return torch.div(self.local_cellxgene_ix, self.n_genes, rounding_mode="floor")

    def create_cut_data(self):
        cut_coordinates = self.coordinates.flatten()
        cut_coordinates = (cut_coordinates - self.window[0]) / (
            self.window[1] - self.window[0]
        )
        keep_cuts = (cut_coordinates >= 0) & (cut_coordinates <= 1)
        cut_coordinates = cut_coordinates[keep_cuts]

        self._cut_coordinates = cut_coordinates

        self.cut_local_gene_ix = self.local_gene_ix.expand(2, -1).T.flatten()[keep_cuts]
        self.cut_local_cell_ix = self.local_cell_ix.expand(2, -1).T.flatten()[keep_cuts]
        self.cut_local_cellxgene_ix = self.local_cellxgene_ix.expand(2, -1).T.flatten()[
            keep_cuts
        ]
        self.cut_localcellxgene_ix = (
            self.cut_local_cell_ix * self.n_total_genes
            + self.genemapping.expand(2, -1).T.flatten()[keep_cuts]
        )

    _cut_coordinates = None

    @property
    def cut_coordinates(self):
        if self._cut_coordinates is None:
            self.create_cut_data()
        return self._cut_coordinates

    @property
    def genes_oi_torch(self):
        return torch.from_numpy(self.genes_oi).to(self.coordinates.device)

    @property
    def cells_oi_torch(self):
        return torch.from_numpy(self.cells_oi).to(self.coordinates.device)

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


class FragmentsResult(Result):
    pass


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
        window = fragments.window
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
        self.out_genemapping = torch.from_numpy(
            np.zeros(self.fragment_buffer_size, dtype=np.int64)
        )  # .pin_memory()
        self.out_local_cellxgene_ix = torch.from_numpy(
            np.zeros(self.fragment_buffer_size, dtype=np.int64)
        )  # .pin_memory()

        self.preloaded = True

    def load(self, minibatch):
        if not self.preloaded:
            self.preload()

        coordinates = self.coordinates
        genemapping = self.genemapping
        cellxgene_indptr = self.cellxgene_indptr

        minibatch.cellxgene_oi = cell_gene_to_cellxgene(
            minibatch.cells_oi, minibatch.genes_oi, self.n_genes
        )

        assert len(minibatch.cellxgene_oi) <= self.cellxgene_batch_size, (
            len(minibatch.cellxgene_oi),
            self.cellxgene_batch_size,
        )
        n_fragments = chromatinhd.loaders.extraction.fragments.extract_fragments(
            minibatch.cellxgene_oi,
            cellxgene_indptr,
            coordinates,
            genemapping,
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

        return FragmentsResult(
            local_cellxgene_ix=self.out_local_cellxgene_ix,
            coordinates=self.out_coordinates,
            n_fragments=n_fragments,
            genemapping=self.out_genemapping,
            window=self.window,
            n_total_genes=self.n_genes,
            **minibatch.items(),
        )


@dataclasses.dataclass
class FragmentsCountingResult(FragmentsResult):
    n: list[torch.Tensor]


class FragmentsCounting:
    cellxgene_batch_size: int

    preloaded = False

    out_coordinates: torch.Tensor
    out_genemapping: torch.Tensor
    out_local_cellxgene_ix: torch.Tensor

    n: tuple = (2,)

    def __init__(
        self,
        fragments,
        cellxgene_batch_size,
        n_fragment_per_cellxgene=None,
        n=(2,),
    ):
        self.cellxgene_batch_size = cellxgene_batch_size

        # store auxilliary information
        window = fragments.window
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

        self.n = n

        self.n_genes = fragments.n_genes

    def preload(self):
        self.out_coordinates = torch.from_numpy(
            np.zeros((self.fragment_buffer_size, 2), dtype=np.int64)
        )  # .pin_memory()
        self.out_genemapping = torch.from_numpy(
            np.zeros(self.fragment_buffer_size, dtype=np.int64)
        )  # .pin_memory()
        self.out_local_cellxgene_ix = torch.from_numpy(
            np.zeros(self.fragment_buffer_size, dtype=np.int64)
        )  # .pin_memory()

        self.out_n = [
            torch.from_numpy(np.zeros(self.fragment_buffer_size, dtype=np.int64))
            for i in self.n
        ]

        self.preloaded = True

    def load(self, minibatch):
        if not self.preloaded:
            self.preload()

        # optional filtering based on fragments_oi
        coordinates = self.coordinates
        genemapping = self.genemapping
        cellxgene_indptr = self.cellxgene_indptr

        minibatch.cellxgene_oi = cell_gene_to_cellxgene(
            minibatch.cells_oi, minibatch.genes_oi, self.n_genes
        )

        assert len(minibatch.cellxgene_oi) <= self.cellxgene_batch_size
        if self.n == (2,):
            (
                n_fragments,
                *n_n,
            ) = chromatinhd.loaders.extraction.fragments.extract_fragments_counting(
                minibatch.cellxgene_oi,
                cellxgene_indptr,
                coordinates,
                genemapping,
                self.out_coordinates.numpy(),
                self.out_genemapping.numpy(),
                self.out_local_cellxgene_ix.numpy(),
                *[out_n.numpy() for out_n in self.out_n],
            )
        else:
            (
                n_fragments,
                *n_n,
            ) = chromatinhd.loaders.extraction.fragments.extract_fragments_counting3(
                minibatch.cellxgene_oi,
                cellxgene_indptr,
                coordinates,
                genemapping,
                self.out_coordinates.numpy(),
                self.out_genemapping.numpy(),
                self.out_local_cellxgene_ix.numpy(),
                *[out_n.numpy() for out_n in self.out_n],
            )
        if n_fragments > self.fragment_buffer_size:
            raise ValueError("n_fragments is too large for the current buffer size")

        if n_fragments == 0:
            n_fragments = 1
        self.out_coordinates.resize_((n_fragments, 2))
        self.out_genemapping.resize_((n_fragments))
        self.out_local_cellxgene_ix.resize_((n_fragments))

        self.out_n = [out_n.resize_((i)) for out_n, i in zip(self.out_n, n_n)]

        return FragmentsCountingResult(
            local_cellxgene_ix=self.out_local_cellxgene_ix,
            coordinates=self.out_coordinates,
            n_fragments=n_fragments,
            genemapping=self.out_genemapping,
            n=self.out_n,
            window=self.window,
            n_total_genes=self.n_genes,
            **minibatch.items(),
        )

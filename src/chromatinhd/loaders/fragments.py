from __future__ import annotations
import torch
import numpy as np
from typing import Union

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
    local_cellxregion_ix: torch.Tensor
    n_fragments: int
    regionmapping: torch.Tensor
    cells_oi: np.ndarray = None
    regions_oi: np.ndarray = None
    window: np.ndarray = None
    n_total_regions: int = None
    localcellxregion_ix: torch.Tensor = None

    @property
    def n_cells(self):
        return len(self.cells_oi)

    @property
    def n_regions(self):
        return len(self.regions_oi)

    def to(self, device):
        for field_name, field in self.__dataclass_fields__.items():
            if field.type is torch.Tensor:
                if self.__getattribute__(field_name) is not None:
                    self.__setattr__(field_name, self.__getattribute__(field_name).to(device))
        return self

    @property
    def local_region_ix(self):
        return self.local_cellxregion_ix % self.n_regions

    @property
    def local_cell_ix(self):
        return torch.div(self.local_cellxregion_ix, self.n_regions, rounding_mode="floor")

    def filter_fragments(self, fragments_oi):
        assert len(fragments_oi) == self.n_fragments
        return Result(
            coordinates=self.coordinates[fragments_oi],
            local_cellxregion_ix=self.local_cellxregion_ix[fragments_oi],
            regionmapping=self.regionmapping[fragments_oi],
            n_fragments=fragments_oi.sum(),
            cells_oi=self.cells_oi,
            regions_oi=self.regions_oi,
            window=self.window,
            n_total_regions=self.n_total_regions,
        )


def cell_region_to_cellxregion(cells_oi, regions_oi, n_regions):
    return (cells_oi[:, None] * n_regions + regions_oi).flatten()


# class Fragments:
#     cellxregion_batch_size: int

#     preloaded = False

#     out_coordinates: torch.Tensor
#     out_regionmapping: torch.Tensor
#     out_local_cellxregion_ix: torch.Tensor

#     n_regions: int

#     def __init__(
#         self,
#         fragments: chromatinhd.data.fragments.Fragments,
#         cellxregion_batch_size: int,
#         n_fragment_per_cellxregion: int = None,
#         fully_contained=False,
#     ):
#         self.cellxregion_batch_size = cellxregion_batch_size

#         # store auxilliary information
#         window = fragments.regions.window
#         self.window = window
#         self.window_width = window[1] - window[0]

#         # store fragment data
#         self.cellxregion_indptr = fragments.cellxregion_indptr.numpy()
#         self.coordinates = fragments.coordinates.numpy()
#         self.regionmapping = fragments.regionmapping.numpy()

#         # create buffers for coordinates
#         if n_fragment_per_cellxregion is None:
#             n_fragment_per_cellxregion = fragments.estimate_fragment_per_cellxregion()
#         fragment_buffer_size = n_fragment_per_cellxregion * cellxregion_batch_size
#         self.fragment_buffer_size = fragment_buffer_size

#         self.n_regions = fragments.n_regions

#         self.fully_contained = fully_contained

#     def preload(self):
#         self.out_coordinates = torch.from_numpy(
#             np.zeros((self.fragment_buffer_size, 2), dtype=np.int64)
#         )  # .pin_memory()
#         self.out_regionmapping = torch.from_numpy(np.zeros(self.fragment_buffer_size, dtype=np.int64))  # .pin_memory()
#         self.out_local_cellxregion_ix = torch.from_numpy(
#             np.zeros(self.fragment_buffer_size, dtype=np.int64)
#         )  # .pin_memory()

#         self.preloaded = True

#     def load(self, minibatch):
#         if not self.preloaded:
#             self.preload()

#         minibatch.cellxregion_oi = cell_region_to_cellxregion(minibatch.cells_oi, minibatch.regions_oi, self.n_regions)

#         if (len(minibatch.cells_oi) * len(minibatch.regions_oi)) > self.cellxregion_batch_size:
#             raise ValueError(
#                 "Too many cell x region requested, increase cellxregion_batch_size at loader initialization"
#             )

#         n_fragments = fragments_helpers.extract_fragments(
#             minibatch.cellxregion_oi,
#             self.cellxregion_indptr,
#             self.coordinates,
#             self.regionmapping,
#             self.out_coordinates.numpy(),
#             self.out_regionmapping.numpy(),
#             self.out_local_cellxregion_ix.numpy(),
#         )
#         if n_fragments > self.fragment_buffer_size:
#             raise ValueError("n_fragments is too large for the current buffer size")

#         if n_fragments == 0:
#             n_fragments = 1
#         self.out_coordinates.resize_((n_fragments, 2))
#         self.out_regionmapping.resize_((n_fragments))
#         self.out_local_cellxregion_ix.resize_((n_fragments))

#         coordinates = self.out_coordinates
#         local_cellxregion_ix = self.out_local_cellxregion_ix
#         regionmapping = self.out_regionmapping

#         if self.fully_contained:
#             selection = (coordinates[:, 0] >= self.window[0]) & (coordinates[:, 1] < self.window[1])
#             coordinates = coordinates[selection]
#             local_cellxregion_ix = local_cellxregion_ix[selection]
#             regionmapping = regionmapping[selection]
#             n_fragments = selection.sum()

#         local_cell_ix = torch.div(local_cellxregion_ix, self.n_regions, rounding_mode="floor")
#         localcellxregion_ix = local_cell_ix * self.n_regions + regionmapping

#         return Result(
#             coordinates=coordinates,
#             local_cellxregion_ix=local_cellxregion_ix,
#             localcellxregion_ix=localcellxregion_ix,
#             n_fragments=n_fragments,
#             regionmapping=regionmapping,
#             window=self.window,
#             n_total_regions=self.n_regions,
#             cells_oi=minibatch.cells_oi,
#             regions_oi=minibatch.regions_oi,
#         )


class Fragments:
    """
    Basic loader for fragments.

    Example:
        >>> loader = Fragments(fragments, cellxregion_batch_size=1000)
        >>> minibatch = Minibatch(cells_oi=np.arange(100), regions_oi=np.arange(100))
        >>> data = loader.load(minibatch)
        >>> data.coordinates
    """

    cellxregion_batch_size: int

    preloaded = False

    out_coordinates: torch.Tensor
    out_regionmapping: torch.Tensor
    out_local_cellxregion_ix: torch.Tensor

    n_regions: int
    is_view: bool

    def __init__(
        self,
        fragments: Union[chromatinhd.data.fragments.Fragments, chromatinhd.data.fragments.FragmentsView],
        cellxregion_batch_size: int,
        n_fragment_per_cellxregion: int = None,
    ):
        self.cellxregion_batch_size = cellxregion_batch_size

        # store auxilliary information
        window = fragments.regions.window
        self.window = window

        # create buffers for coordinates
        if n_fragment_per_cellxregion is None:
            n_fragment_per_cellxregion = fragments.estimate_fragment_per_cellxregion()
        fragment_buffer_size = n_fragment_per_cellxregion * cellxregion_batch_size
        self.fragment_buffer_size = fragment_buffer_size

        self.n_regions = fragments.n_regions

        # set up readers and determine if we are dealing with a view or not
        if isinstance(fragments, chromatinhd.data.fragments.Fragments):
            self.regionxcell_indptr_reader = fragments.regionxcell_indptr.open_reader(
                {"context": {"data_copy_concurrency": {"limit": 1}}}
            )
            self.coordinates_reader = fragments.coordinates.open_reader(
                {
                    "context": {
                        "data_copy_concurrency": {"limit": 1},
                    }
                }
            )
            self.is_view = False

        elif isinstance(fragments, chromatinhd.data.fragments.FragmentsView):
            self.regionxcell_fragmentixs_indptr_reader = fragments.regionxcell_fragmentixs_indptr.open_reader(
                {"context": {"data_copy_concurrency": {"limit": 1}}}
            )
            self.coordinates_reader = fragments.parent.coordinates.open_reader(
                {
                    "context": {
                        "data_copy_concurrency": {"limit": 1},
                    }
                }
            )
            self.fragmentixs_reader = fragments.fragmentixs.open_reader(
                {"context": {"data_copy_concurrency": {"limit": 1}}}
            )
            self.region_centers = fragments.regions.coordinates["tss"].values
            self.region_strands = fragments.regions.coordinates["strand"].values
            self.is_view = True
        else:
            raise ValueError("fragments must be either a Fragments or FragmentsView object")

        self.n_cells = fragments.n_cells

    def preload(self):
        self.out_fragmentixs = np.zeros((self.fragment_buffer_size,), dtype=np.int64)
        self.out_local_cellxregion_ix = np.zeros((self.fragment_buffer_size,), dtype=np.int64)

        self.preloaded = True

    def load(self, minibatch):
        if not self.preloaded:
            self.preload()

        if (len(minibatch.cells_oi) * len(minibatch.regions_oi)) > self.cellxregion_batch_size:
            raise ValueError(
                "Too many cell x region requested, increase cellxregion_batch_size at loader initialization"
            )

        if self.is_view:
            # load the fragment indices using pointers to the regionxcell fragment indices
            regionxcell_ixs = (minibatch.regions_oi * self.n_cells + minibatch.cells_oi[:, None]).flatten()
            n_fragments = fragments_helpers.multiple_arange(
                self.regionxcell_fragmentixs_indptr_reader[regionxcell_ixs].read().result(),
                self.regionxcell_fragmentixs_indptr_reader[regionxcell_ixs + 1].read().result(),
                self.out_fragmentixs,
                self.out_local_cellxregion_ix,
            )

            fragmentixs = np.resize(self.out_fragmentixs, n_fragments)
            local_cellxregion_ix = np.resize(self.out_local_cellxregion_ix, n_fragments)

            # load the actual fragment data
            fragmentixs = self.fragmentixs_reader[fragmentixs].read().result()
            regionmapping = minibatch.regions_oi[local_cellxregion_ix % minibatch.n_regions]
            coordinates = (
                self.coordinates_reader[fragmentixs].read().result()
            )  # this is typically the slowest part by far

            # center coordinates around region centers, flip based on strandedness
            coordinates = (self.region_centers[regionmapping][:, None] - coordinates) * self.region_strands[
                regionmapping
            ][:, None]
        else:
            regionxcell_ixs = (minibatch.regions_oi * self.n_cells + minibatch.cells_oi[:, None]).flatten()
            n_fragments = fragments_helpers.multiple_arange(
                self.regionxcell_indptr_reader[regionxcell_ixs].read().result(),
                self.regionxcell_indptr_reader[regionxcell_ixs + 1].read().result(),
                self.out_fragmentixs,
                self.out_local_cellxregion_ix,
            )
            coordinates = (
                self.coordinates_reader[self.out_fragmentixs].read().result()
            )  # this is typically the slowest part by far
            local_cellxregion_ix = np.resize(self.out_local_cellxregion_ix, n_fragments)
            regionmapping = minibatch.regions_oi[local_cellxregion_ix % minibatch.n_regions]

        return Result(
            coordinates=coordinates,
            local_cellxregion_ix=local_cellxregion_ix,
            n_fragments=n_fragments,
            regionmapping=regionmapping,
            window=self.window,
            n_total_regions=self.n_regions,
            cells_oi=minibatch.cells_oi,
            regions_oi=minibatch.regions_oi,
        )

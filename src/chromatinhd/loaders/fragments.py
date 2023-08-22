from __future__ import annotations

from typing import Union

import numpy as np
import torch

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
from chromatinhd.data.minibatches import Minibatch


@dataclasses.dataclass
class FragmentsResult:
    coordinates: torch.Tensor
    "Coordinates of the left and right cut site for each fragment"

    local_cellxregion_ix: torch.Tensor
    "Local cell x region index"

    n_fragments: int
    "Number of fragments"

    regionmapping: torch.Tensor
    "Mapping from local cell x region index to region index"

    cells_oi: np.ndarray = None
    "Cells of interest"

    regions_oi: np.ndarray = None
    "Regions of interest"

    window: np.ndarray = None
    "Window of the region"

    n_total_regions: int = None
    "Total number of regions"

    localcellxregion_ix: torch.Tensor = None
    "Local cell x region index, in the same order as cells_oi and regions_oi"

    @property
    def n_cells(self):
        return len(self.cells_oi)

    @property
    def n_regions(self):
        return len(self.regions_oi)

    def to(self, device):
        self.coordinates = self.coordinates.to(device)
        self.local_cellxregion_ix = self.local_cellxregion_ix.to(device)
        self.regionmapping = self.regionmapping.to(device)
        return self

    @property
    def local_region_ix(self):
        return self.local_cellxregion_ix % self.n_regions

    @property
    def local_cell_ix(self):
        return torch.div(self.local_cellxregion_ix, self.n_regions, rounding_mode="floor")

    def filter_fragments(self, fragments_oi):
        assert len(fragments_oi) == self.n_fragments
        return FragmentsResult(
            coordinates=self.coordinates[fragments_oi],
            local_cellxregion_ix=self.local_cellxregion_ix[fragments_oi],
            regionmapping=self.regionmapping[fragments_oi],
            n_fragments=fragments_oi.sum(),
            cells_oi=self.cells_oi,
            regions_oi=self.regions_oi,
            window=self.window,
            n_total_regions=self.n_total_regions,
        )


class Fragments:
    """
    Basic loader for fragments.

    Example:
        ```
        loader = Fragments(fragments, cellxregion_batch_size=1000)
        minibatch = Minibatch(cells_oi=np.arange(100), regions_oi=np.arange(100))
        data = loader.load(minibatch)
        data.coordinates
        ```
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
        """
        Parameters:
            fragments: Fragments object
            cellxregion_batch_size: maximum number of cell x region combinations that will be loaded
            n_fragment_per_cellxregion: estimated number of the number of fragments per cell x region combination, used for preallocation
        """
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

    def load(self, minibatch: Minibatch) -> FragmentsResult:
        """
        Load a minibatch of fragments.

        Parameters:
            minibatch: Minibatch object

        Returns:
            The loaded fragments
        """
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
            fragmentixs = np.resize(self.out_fragmentixs, n_fragments)
            coordinates = (
                self.coordinates_reader[fragmentixs].read().result()
            )  # this is typically the slowest part by far
            local_cellxregion_ix = np.resize(self.out_local_cellxregion_ix, n_fragments)
            regionmapping = minibatch.regions_oi[local_cellxregion_ix % minibatch.n_regions]

        return FragmentsResult(
            coordinates=torch.from_numpy(coordinates),
            local_cellxregion_ix=torch.from_numpy(local_cellxregion_ix),
            n_fragments=n_fragments,
            regionmapping=torch.from_numpy(regionmapping),
            window=self.window,
            n_total_regions=self.n_regions,
            cells_oi=minibatch.cells_oi,
            regions_oi=minibatch.regions_oi,
        )


@dataclasses.dataclass
class CutsResult:
    coordinates: torch.Tensor
    local_cellxregion_ix: torch.Tensor
    localcellxregion_ix: torch.Tensor
    n_regions: int
    n_fragments: int
    n_cuts: int

    @property
    def local_region_ix(self):
        return self.local_cellxregion_ix % self.n_regions

    @property
    def local_cell_ix(self):
        return torch.div(self.local_cellxregion_ix, self.n_regions, rounding_mode="floor")

    def to(self, device):
        self.coordinates = self.coordinates.to(device)
        self.local_cellxregion_ix = self.local_cellxregion_ix.to(device)
        self.localcellxregion_ix = self.localcellxregion_ix.to(device)
        return self


class Cuts(Fragments):
    def load(self, minibatch: Minibatch) -> CutsResult:
        """
        Load a minibatch of cuts.

        Parameters:
            minibatch: Minibatch object

        Returns:
            The loaded cut sites
        """
        result = super().load(minibatch)

        cut_coordinates = result.coordinates.flatten()
        local_cellxregion_ix = result.local_cellxregion_ix.expand(2, -1).T.flatten()
        local_cell_ix = torch.div(local_cellxregion_ix, self.n_regions, rounding_mode="floor")
        localcellxregion_ix = local_cell_ix * self.n_regions + result.regionmapping.expand(2, -1).T.flatten()

        return CutsResult(
            coordinates=cut_coordinates,
            local_cellxregion_ix=local_cellxregion_ix,
            localcellxregion_ix=localcellxregion_ix,
            n_regions=len(minibatch.regions_oi),
            n_fragments=result.n_fragments,
            n_cuts=result.n_fragments * 2,
        )

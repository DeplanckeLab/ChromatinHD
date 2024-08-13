from __future__ import annotations

from typing import Union

import numpy as np
import torch
import time

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
from chromatinhd.loaders.minibatches import Minibatch
from chromatinhd.utils.numpy import indices_to_indptr


@dataclasses.dataclass
class FragmentsResult:
    coordinates: torch.Tensor
    "Coordinates of the left and right cut site for each fragment"

    local_regionxcell_ix: torch.Tensor
    "Local cell x region index"

    n_fragments: int
    "Number of fragments"

    regionmapping: torch.Tensor = None
    "Mapping from local cell x region index to region index"

    region_indptr: torch.Tensor = None
    "Indptr for regions"

    cells_oi: np.ndarray = None
    "Cells of interest"

    regions_oi: np.ndarray = None
    "Regions of interest"

    window: np.ndarray = None
    "Window of the region"

    n_total_regions: int = None
    "Total number of regions"

    lib: torch.Tensor = None
    "Library size for each cell"

    doublet_idx: torch.Tensor = None
    "Indices of doublets"

    @property
    def n_cells(self):
        return len(self.cells_oi)

    @property
    def n_regions(self):
        return len(self.regions_oi)

    def to(self, device):
        self.coordinates = self.coordinates.to(device)
        self.local_regionxcell_ix = self.local_regionxcell_ix.to(device)
        if self.regionmapping is not None:
            self.regionmapping = self.regionmapping.to(device)
        if self.lib is not None:
            self.lib = self.lib.to(device)
        self.region_indptr = self.region_indptr.to(device)
        return self

    @property
    def local_region_ix(self):
        return torch.div(self.local_regionxcell_ix, self.n_cells, rounding_mode="floor")

    @property
    def local_cell_ix(self):
        return torch.remainder(self.local_regionxcell_ix, self.n_cells)

    def filter_fragments(self, fragments_oi):
        assert len(fragments_oi) == self.n_fragments
        return FragmentsResult(
            coordinates=self.coordinates[fragments_oi],
            local_regionxcell_ix=self.local_regionxcell_ix[fragments_oi],
            regionmapping=self.regionmapping[fragments_oi],
            n_fragments=fragments_oi.sum(),
            cells_oi=self.cells_oi,
            regions_oi=self.regions_oi,
            window=self.window,
            n_total_regions=self.n_total_regions,
        )


class catchtime(object):
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.t = time.time() - self.t
        # print(self.name, self.t)


def open_memmap(path):
    import pickle

    filename = path.with_suffix(".dat")
    filename_meta = path.with_suffix(".meta")
    metadata = pickle.load(open(filename_meta, "rb"))
    return np.memmap(filename, dtype=metadata["dtype"], shape=metadata["shape"], mode="r")


class Fragments:
    """
    Basic loader for fragments. This requires either `regionxcell_indptr` (for a Fragments) or `regionxcell_fragmentixs_indptr` (for a FragmentsView) to be present.

    Example:
        ```
        loader = Fragments(fragments, regionxcell_batch_size=1000)
        minibatch = Minibatch(cells_oi=np.arange(100), regions_oi=np.arange(100))
        data = loader.load(minibatch)
        data.coordinates
        ```
    """

    regionxcell_batch_size: int

    preloaded = False

    out_coordinates: torch.Tensor
    out_regionmapping: torch.Tensor
    out_local_regionxcell_ix: torch.Tensor

    n_regions: int
    is_view: bool

    def __init__(
        self,
        fragments: Union[chromatinhd.data.fragments.Fragments, chromatinhd.data.fragments.FragmentsView],
        regionxcell_batch_size: int,
        n_fragment_per_regionxcell: int = None,
        buffer_size_multiplier=10,  # increase this if crashing
        provide_lib=False,
        provide_multiplets=True,
    ):
        """
        Parameters:
            fragments: Fragments object
            regionxcell_batch_size: maximum number of cell x region combinations that will be loaded
            n_fragment_per_regionxcell: estimated number of the number of fragments per cell x region combination, used for preallocation
        """
        self.regionxcell_batch_size = regionxcell_batch_size

        # store auxilliary information
        window = fragments.regions.window
        self.window = window

        # create buffers for coordinates
        if n_fragment_per_regionxcell is None:
            n_fragment_per_regionxcell = fragments.estimate_fragment_per_cellxregion()
        fragment_buffer_size = n_fragment_per_regionxcell * regionxcell_batch_size * buffer_size_multiplier
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

        elif isinstance(fragments, chromatinhd.data.fragments.view.FragmentsView):
            self.regionxcell_indptr_reader = fragments.regionxcell_indptr.open_reader()
            self.coordinates_reader = fragments.coordinates.open_reader()

            if "strand" in fragments.regions.coordinates.columns:
                self.region_strands = fragments.regions.coordinates["strand"].values
            else:
                self.region_strands = np.ones((len(fragments.regions.coordinates),), dtype=np.int8)
            self.region_centers = (fragments.regions.coordinates["start"] - fragments.regions.window[0]).values * (
                self.region_strands == 1
            ).astype(int) + (fragments.regions.coordinates["end"] + fragments.regions.window[0]).values * (
                self.region_strands == -1
            ).astype(
                int
            )
            self.is_view = True
        else:
            raise ValueError("fragments must be either a Fragments or FragmentsView object", type(fragments))

        self.n_cells = fragments.n_cells

        self.provide_multiplets = provide_multiplets

    def preload(self):
        self.out_fragmentixs = np.zeros((self.fragment_buffer_size,), dtype=np.int64)
        self.out_local_regionxcell_ix = np.zeros((self.fragment_buffer_size,), dtype=np.int64)

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

        if (len(minibatch.cells_oi) * len(minibatch.regions_oi)) > self.regionxcell_batch_size:
            raise ValueError(
                "Too many cell x region requested, increase regionxcell_batch_size at loader initialization"
            )

        if self.is_view:
            with catchtime("all") as t:
                with catchtime("arange") as t:
                    # load the fragment indices using pointers to the regionxcell fragment indices
                    regionxcell_ixs = (minibatch.regions_oi * self.n_cells + minibatch.cells_oi[:, None]).flatten()
                    n_fragments = fragments_helpers.multiple_arange(
                        np.array(self.regionxcell_indptr_reader[regionxcell_ixs, 0]),
                        np.array(self.regionxcell_indptr_reader[regionxcell_ixs, 1]),
                        self.out_fragmentixs,
                        self.out_local_regionxcell_ix,
                    )

                    assert n_fragments < self.fragment_buffer_size, "fragment buffer size too small"

                regionxcell_fragmentixs = self.out_fragmentixs[:n_fragments]
                local_regionxcell_ix = self.out_local_regionxcell_ix[:n_fragments]

                with catchtime("coordinates") as t:
                    regionmapping = minibatch.regions_oi[local_regionxcell_ix % minibatch.n_regions]
                    coordinates = self.coordinates_reader[
                        regionxcell_fragmentixs
                    ]  # this is typically the slowest part by far

                with catchtime("center") as t:
                    # center coordinates around region centers, flip based on strandedness
                    coordinates = (coordinates - self.region_centers[regionmapping][:, None]) * self.region_strands[
                        regionmapping
                    ][:, None]
        else:
            regionxcell_ixs = (minibatch.regions_oi[:, None] * self.n_cells + minibatch.cells_oi).flatten()
            n_fragments = fragments_helpers.multiple_arange(
                self.regionxcell_indptr_reader[regionxcell_ixs],
                self.regionxcell_indptr_reader[regionxcell_ixs + 1],
                self.out_fragmentixs,
                self.out_local_regionxcell_ix,
            )
            regionxcell_fragmentixs = np.resize(self.out_fragmentixs, n_fragments)
            coordinates = self.coordinates_reader[regionxcell_fragmentixs]  # this is typically the slowest part by far
            local_regionxcell_ix = np.resize(self.out_local_regionxcell_ix, n_fragments)
            local_region_ix = local_regionxcell_ix // minibatch.n_cells
            regionmapping = minibatch.regions_oi[local_regionxcell_ix // minibatch.n_cells]

        region_indptr = torch.from_numpy(indices_to_indptr(local_region_ix, minibatch.n_regions))

        # multiplets
        # if self.provide_multiplets:
        #     indptr = indices_to_indptr(local_regionxcell_ix, minibatch.n_cells * minibatch.n_regions)
        #     indptr_diff = np.diff(indptr)

        #     doublets = np.where(indptr_diff == 2)[0]
        #     doublet_idx = torch.from_numpy(np.stack([indptr[doublets], indptr[doublets] + 1], -1).flatten())

        #     triplets = np.where(indptr_diff == 2)[0]
        #     triplet_idx = np.stack([indptr[triplets], indptr[triplets] + 1, indptr[triplets] + 2], -1).flatten()
        # else:
        #     doublet_idx = None
        #     triplet_idx = None

        return FragmentsResult(
            coordinates=torch.from_numpy(coordinates),
            local_regionxcell_ix=torch.from_numpy(local_regionxcell_ix),
            n_fragments=n_fragments,
            regionmapping=torch.from_numpy(regionmapping),
            window=self.window,
            n_total_regions=self.n_regions,
            cells_oi=minibatch.cells_oi,
            regions_oi=minibatch.regions_oi,
            doublet_idx=doublet_idx,
            region_indptr=region_indptr,
        )


@dataclasses.dataclass
class CutsResult:
    coordinates: torch.Tensor
    local_regionxcell_ix: torch.Tensor
    n_regions: int
    n_fragments: int
    n_cuts: int
    window: np.ndarray

    @property
    def local_region_ix(self):
        return self.local_regionxcell_ix % self.n_regions

    @property
    def local_cell_ix(self):
        return torch.div(self.local_regionxcell_ix, self.n_regions, rounding_mode="floor")

    def to(self, device):
        self.coordinates = self.coordinates.to(device)
        self.local_regionxcell_ix = self.local_regionxcell_ix.to(device)
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
        local_regionxcell_ix = result.local_regionxcell_ix.expand(2, -1).T.flatten()

        selected = np.random.rand(len(cut_coordinates)) < 0.2
        cut_coordinates = cut_coordinates[selected]
        local_regionxcell_ix = local_regionxcell_ix[selected]

        return CutsResult(
            coordinates=cut_coordinates,
            local_regionxcell_ix=local_regionxcell_ix,
            n_regions=len(minibatch.regions_oi),
            n_fragments=result.n_fragments,
            n_cuts=result.n_fragments * 2,
            window=self.window,
        )

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


@dataclasses.dataclass
class FragmentsResult:
    coordinates: torch.Tensor
    "Coordinates of the left and right cut site for each fragment"

    local_cellxregion_ix: torch.Tensor
    "Local cell x region index"

    n_fragments: int
    "Number of fragments"

    regionmapping: torch.Tensor = None
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

    doublet_idx: torch.Tensor = None
    "Indices of doublets"

    libsize: torch.Tensor = None
    "Library size for each cell"

    @property
    def n_cells(self):
        return len(self.cells_oi)

    @property
    def n_regions(self):
        return len(self.regions_oi)

    def to(self, device):
        self.coordinates = self.coordinates.to(device)
        self.local_cellxregion_ix = self.local_cellxregion_ix.to(device)
        if self.regionmapping is not None:
            self.regionmapping = self.regionmapping.to(device)
        if self.libsize is not None:
            self.libsize = self.libsize.to(device)
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
        buffer_size_multiplier=10,  # increase this if crashing
        provide_libsize=False,
        provide_multiplets=True,
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
        fragment_buffer_size = n_fragment_per_cellxregion * cellxregion_batch_size * buffer_size_multiplier
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
        self.provide_libsize = provide_libsize

        if provide_libsize:
            library_size = fragments.libsize
            self.library_size = torch.from_numpy((library_size - library_size.mean()) / library_size.std()).float()

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
                np.array(self.regionxcell_indptr_reader[regionxcell_ixs, 0]),
                np.array(self.regionxcell_indptr_reader[regionxcell_ixs, 1]),
                self.out_fragmentixs,
                self.out_local_cellxregion_ix,
            )

            assert n_fragments < self.fragment_buffer_size, "fragment buffer size too small"

            regionxcell_fragmentixs = self.out_fragmentixs[:n_fragments]
            local_cellxregion_ix = self.out_local_cellxregion_ix[:n_fragments]

            regionmapping = minibatch.regions_oi[local_cellxregion_ix % minibatch.n_regions]
            coordinates = self.coordinates_reader[regionxcell_fragmentixs]  # this is typically the slowest part by far

            # center coordinates around region centers, flip based on strandedness
            coordinates = (coordinates - self.region_centers[regionmapping][:, None]) * self.region_strands[
                regionmapping
            ][:, None]
        else:
            regionxcell_ixs = (minibatch.regions_oi * self.n_cells + minibatch.cells_oi[:, None]).flatten()
            n_fragments = fragments_helpers.multiple_arange(
                self.regionxcell_indptr_reader[regionxcell_ixs],
                self.regionxcell_indptr_reader[regionxcell_ixs + 1],
                self.out_fragmentixs,
                self.out_local_cellxregion_ix,
            )
            regionxcell_fragmentixs = np.resize(self.out_fragmentixs, n_fragments)
            coordinates = self.coordinates_reader[regionxcell_fragmentixs]  # this is typically the slowest part by far
            local_cellxregion_ix = np.resize(self.out_local_cellxregion_ix, n_fragments)
            regionmapping = minibatch.regions_oi[local_cellxregion_ix % minibatch.n_regions]

        # multiplets
        if self.provide_multiplets:
            from chromatinhd.utils.numpy import indices_to_indptr

            indptr = indices_to_indptr(local_cellxregion_ix, minibatch.n_cells * minibatch.n_regions)
            indptr_diff = np.diff(indptr)

            doublets = np.where(indptr_diff == 2)[0]
            doublet_idx = torch.from_numpy(np.stack([indptr[doublets], indptr[doublets] + 1], -1).flatten())

            triplets = np.where(indptr_diff == 2)[0]
            triplet_idx = np.stack([indptr[triplets], indptr[triplets] + 1, indptr[triplets] + 2], -1).flatten()
        else:
            doublet_idx = None
            triplet_idx = None

        # libsize
        if self.provide_libsize:
            libsize = self.library_size[minibatch.cells_oi]
        else:
            libsize = None

        return FragmentsResult(
            coordinates=torch.from_numpy(coordinates),
            local_cellxregion_ix=torch.from_numpy(local_cellxregion_ix),
            n_fragments=n_fragments,
            regionmapping=torch.from_numpy(regionmapping),
            window=self.window,
            n_total_regions=self.n_regions,
            cells_oi=minibatch.cells_oi,
            regions_oi=minibatch.regions_oi,
            doublet_idx=doublet_idx,
            libsize=libsize,
        )


@dataclasses.dataclass
class CutsResult:
    coordinates: torch.Tensor
    local_cellxregion_ix: torch.Tensor
    n_regions: int
    n_fragments: int
    n_cuts: int
    window: np.ndarray

    @property
    def local_region_ix(self):
        return self.local_cellxregion_ix % self.n_regions

    @property
    def local_cell_ix(self):
        return torch.div(self.local_cellxregion_ix, self.n_regions, rounding_mode="floor")

    def to(self, device):
        self.coordinates = self.coordinates.to(device)
        self.local_cellxregion_ix = self.local_cellxregion_ix.to(device)
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

        n_cuts_per_fragment = result.coordinates.shape[1]
        local_cellxregion_ix = result.local_cellxregion_ix.expand(n_cuts_per_fragment, -1).T.flatten()

        # selected = np.random.rand(len(cut_coordinates)) < 0.2
        # cut_coordinates = cut_coordinates[selected]
        # local_cellxregion_ix = local_cellxregion_ix[selected]

        return CutsResult(
            coordinates=cut_coordinates,
            local_cellxregion_ix=local_cellxregion_ix,
            n_regions=len(minibatch.regions_oi),
            n_fragments=result.n_fragments,
            n_cuts=result.n_fragments * n_cuts_per_fragment,
            window=self.window,
        )


class FragmentsRegional:
    """
    Basic loader for fragments. This requires either `regionxcell_indptr` (for a Fragments) or `regionxcell_fragmentixs_indptr` (for a FragmentsView) to be present.

    Example:
        ```
        loader = Fragments(fragments, cellxregion_batch_size=1000)
        minibatch = Minibatch(cells_oi=np.arange(100), regions_oi=np.arange(100))
        data = loader.load(minibatch)
        data.coordinates
        ```
    """

    is_view = False

    cellxregion_batch_size: int

    preloaded = False

    out_coordinates: torch.Tensor
    out_regionmapping: torch.Tensor
    out_local_cellxregion_ix: torch.Tensor

    n_regions: int

    def __init__(
        self,
        fragments: Union[chromatinhd.data.fragments.Fragments, chromatinhd.data.fragments.FragmentsView],
        cellxregion_batch_size: int,
        region_oi,
        n_fragment_per_cellxregion: int = None,
        buffer_size_multiplier=10,  # increase this if crashing
        provide_libsize=False,
        provide_multiplets=False,
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
        fragment_buffer_size = n_fragment_per_cellxregion * cellxregion_batch_size * buffer_size_multiplier
        self.fragment_buffer_size = fragment_buffer_size

        self.n_regions = fragments.n_regions

        # load cache
        cache = fragments.get_cache(region_oi)
        self.regionxcell_indptr_reader = cache["regionxcell_indptr"]
        self.coordinates_reader = cache["coordinates"]

        self.n_cells = fragments.n_cells

        if provide_multiplets:
            raise NotImplementedError("provide_multiplets not implemented for FragmentsRegional")
        self.provide_multiplets = provide_multiplets

        # check if view
        if isinstance(fragments, chromatinhd.data.fragments.view.FragmentsView):
            self.is_view = True
            self.center = int(fragments.regions.coordinates.loc[region_oi]["tss"])
            self.strand = int(fragments.regions.coordinates.loc[region_oi]["strand"])

        self.provide_libsize = provide_libsize
        if provide_libsize:
            library_size = fragments.libsize
            self.library_size = torch.from_numpy((library_size - library_size.mean()) / library_size.std()).float()

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

        regionxcell_ixs = (minibatch.cells_oi[:, None]).flatten()
        n_fragments = fragments_helpers.multiple_arange(
            self.regionxcell_indptr_reader[regionxcell_ixs],
            self.regionxcell_indptr_reader[regionxcell_ixs + 1],
            self.out_fragmentixs,
            self.out_local_cellxregion_ix,
        )
        regionxcell_fragmentixs = np.resize(self.out_fragmentixs, n_fragments)
        coordinates = self.coordinates_reader[regionxcell_fragmentixs]
        local_cellxregion_ix = np.resize(self.out_local_cellxregion_ix, n_fragments)

        if self.is_view:
            coordinates = (coordinates - self.center) * self.strand  # .astype(np.int32)

        if self.provide_libsize:
            libsize = self.library_size[minibatch.cells_oi]
        else:
            libsize = None

        return FragmentsResult(
            coordinates=torch.from_numpy(coordinates),
            local_cellxregion_ix=torch.from_numpy(local_cellxregion_ix),
            n_fragments=n_fragments,
            window=self.window,
            n_total_regions=self.n_regions,
            cells_oi=minibatch.cells_oi,
            regions_oi=minibatch.regions_oi,
            libsize=libsize,
        )


class CutsRegional(FragmentsRegional):
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

        return CutsResult(
            coordinates=cut_coordinates,
            local_cellxregion_ix=local_cellxregion_ix,
            n_regions=len(minibatch.regions_oi),
            n_fragments=result.n_fragments,
            n_cuts=result.n_fragments * 2,
            window=self.window,
        )

from __future__ import annotations
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import math
import time
import functools

from chromatinhd.data.fragments import Fragments
from chromatinhd.data.regions import Regions
from chromatinhd.flow import TSV, Flow, Linked, PathLike, Stored
from chromatinhd.flow.tensorstore import Tensorstore, TensorstoreInstance
from chromatinhd.utils.numpy import indices_to_indptr


class FragmentsView(Flow):
    """
    A view of fragments based on regions that are a subset of the parent fragments. In a typical use case, the parent contains fragments for all chromosomes, while this the view focuses on specific regions.

    Only fragments that are fully inclusive (i.e., both left and right cut sites) within a region will be selected.
    """

    regionxcell_indptr: TensorstoreInstance = Tensorstore(
        dtype="<i8", chunks=[100000], compression="blosc", shape=(0, 2)
    )
    """Index of fragments in the parent fragments object"""

    parent: Fragments = Linked()
    """The parent fragments object from which this view is created"""

    regions: Regions = Linked()
    """The regions object"""

    parentregion_column: str = Stored()

    @classmethod
    def from_fragments(
        cls,
        parent: Fragments,
        regions: Regions,
        parentregion_column: str = "chrom",
        obs: pd.DataFrame = None,
        var: pd.DataFrame = None,
        path: PathLike = None,
        overwrite: bool = False,
    ):
        """
        Creates a fragments view from a parent fragments object and a regions object

        Parameters:
            parent:
                Parent fragments object. If a fragments view is provided, the parent of the parent will be used.
            regions:
                Regions object
            obs:
                DataFrame containing information about cells, will be copied from the fragments object if not provided
            parentregion_column:
                Column in the regions coordinates that links each new region to the regions of the original fragments. This is typically the chromosome column. This column should be present in both `parent.regions.coordinates` and `regions.coordinates`
            path:
                Path to store the fragments view
        """

        if isinstance(parent, FragmentsView):
            while isinstance(parent, FragmentsView):
                parent = parent.parent
        if not isinstance(parent, Fragments):
            raise ValueError("parent should be a Fragments object")
        if not isinstance(regions, Regions):
            raise ValueError("regions should be a Regions object", regions.__class__)

        # dummy proofing
        if parentregion_column not in regions.coordinates.columns:
            raise ValueError(
                f"Column {parentregion_column} not in regions coordinates. Available columns are {regions.coordinates.columns}"
            )
        if parentregion_column not in parent.regions.coordinates.columns:
            raise ValueError(
                f"Column {parentregion_column} not in fragments regions coordinates. Available columns are {parent.regions.coordinates.columns}"
            )
        if not (regions.coordinates[parentregion_column].isin(parent.regions.coordinates[parentregion_column])).all():
            raise ValueError(
                f"Not all regions are present in the parent fragments. Missing regions: {regions.coordinates[parentregion_column][~regions.coordinates[parentregion_column].isin(parent.regions.coordinates[parentregion_column])]}"
            )

        self = cls.create(
            parent=parent,
            regions=regions,
            path=path,
            parentregion_column=parentregion_column,
            obs=parent.obs if obs is None else obs,
            var=regions.coordinates if var is None else var,
            reset=overwrite,
        )

        return self

    def create_regionxcell_indptr(
        self,
        inclusive: tuple = (True, False),
        overwrite=True,
    ) -> FragmentsView:
        """
        Create index pointers (left and right) for the fragments associated to each regionxcell combination

        Parameters:
            batch_size:
                Number of regions to wait before saving the intermediate results and freeing up memory. Reduce this number to avoid running out of memory.
            inclusive:
                Whether to only include fragments that are only partially overlapping with the region. Must be a tuple indicating left and/or right inclusivity.
            overwrite:
                Whether to overwrite the existing index pointers.

        Returns:
            Same object but with the `regionxcell_indptr` populated

        """
        if self.regionxcell_indptr.exists() and not overwrite:
            return self

        mapping = self.parent.mapping[:]
        coordinates = self.parent.coordinates[:]

        # convert regions in parent to parent region ixs
        parentregion_to_parentregion_ix = self.parent.regions.coordinates.index.get_loc

        # reset the tensorstores
        regionxcell_indptr = np.zeros((len(self.regions.coordinates) * len(self.obs), 2), dtype=np.int64)

        # index pointers from parentregions to fragments
        parentregion_fragment_indptr = indices_to_indptr(mapping[:, 1], self.parent.n_regions)

        pbar = tqdm.tqdm(
            total=len(self.regions.coordinates),
            leave=False,
            desc="Processing regions",
        )

        self.regions.coordinates["ix"] = np.arange(len(self.regions))

        for parentregion, subcoordinates in self.regions.coordinates.groupby(self.parentregion_column):
            # extract in which parent region we need to look
            parentregion_ix = parentregion_to_parentregion_ix(parentregion)

            parentregion_start_ix, parentregion_end_ix = (
                parentregion_fragment_indptr[parentregion_ix],
                parentregion_fragment_indptr[parentregion_ix + 1],
            )

            # extract parent's mapping and coordinates
            coordinates_parentregion = coordinates[parentregion_start_ix:parentregion_end_ix]
            cellmapping_parentregion = mapping[parentregion_start_ix:parentregion_end_ix, 0]

            cell_indptr_parentregion = indices_to_indptr(cellmapping_parentregion, self.n_cells, dtype=np.int64)

            for region_id, region in subcoordinates.iterrows():
                region_ix = region["ix"]

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_description(f"Processing region {region_id}")

                # extract fragments lying within the region
                # depending on the strand, we either need to include the start or the end
                region["start_inclusive"] = region["start"] + (region["strand"] == -1)
                region["end_inclusive"] = region["end"] + (region["strand"] == -1)

                if inclusive == (True, False):
                    fragments_oi = (coordinates_parentregion[:, 0] >= region["start_inclusive"]) & (
                        coordinates_parentregion[:, 0] < region["end_inclusive"]
                    )
                    fragments_excluded_left = coordinates_parentregion[:, 0] < region["start_inclusive"]
                else:
                    raise NotImplementedError("For now, only left-inclusive fragments are supported")

                n_excluded_left = np.bincount(cellmapping_parentregion[fragments_excluded_left], minlength=self.n_cells)
                n_included = np.bincount(cellmapping_parentregion[fragments_oi], minlength=self.n_cells)

                cell_indptr_left = cell_indptr_parentregion[:-1] + n_excluded_left
                cell_indptr_right = cell_indptr_parentregion[:-1] + n_excluded_left + n_included

                cell_indptr = np.stack([cell_indptr_left, cell_indptr_right], axis=1)

                regionxcell_indptr[region_ix * self.n_cells : (region_ix + 1) * self.n_cells] = (
                    cell_indptr + parentregion_start_ix
                )

        regionxcell_indptr_writer = self.regionxcell_indptr.open_creator(
            shape=(len(self.regions.coordinates) * len(self.obs), 2)
        )
        regionxcell_indptr_writer[:] = regionxcell_indptr

        return self

    def create_regionxcell_indptr2(
        self,
        inclusive: tuple = (True, False),
        overwrite=True,
    ) -> FragmentsView:
        """
        Create index pointers (left and right) for the fragments associated to each regionxcell combination. This implementation is faster if there are many samples with a lot of fragments (e.g. minibulk data)

        Parameters:
            batch_size:
                Number of regions to wait before saving the intermediate results and freeing up memory. Reduce this number to avoid running out of memory.
            inclusive:
                Whether to only include fragments that are only partially overlapping with the region. Must be a tuple indicating left and/or right inclusivity.
            overwrite:
                Whether to overwrite the existing index pointers.

        Returns:
            Same object but with the `regionxcell_indptr` populated

        """
        if self.regionxcell_indptr.exists() and not overwrite:
            return self

        mapping = self.parent.mapping[:]
        coordinates = self.parent.coordinates[:]

        # convert regions in parent to parent region ixs
        parentregion_to_parentregion_ix = self.parent.regions.coordinates.index.get_loc

        # reset the tensorstores
        regionxcell_indptr = np.zeros((len(self.regions.coordinates) * len(self.obs), 2), dtype=np.int64)

        # index pointers from parentregions to fragments
        parent_regionxcell_indptr = indices_to_indptr(
            mapping[:, 1] * self.parent.n_cells + mapping[:, 0], self.parent.n_regions * self.parent.n_cells
        )

        pbar = tqdm.tqdm(
            total=len(self.regions.coordinates) * len(self.obs),
            leave=False,
            desc="Processing regions",
        )

        self.regions.coordinates["ix"] = np.arange(len(self.regions))
        self.regions.coordinates["start_inclusive"] = self.regions.coordinates["start"] + (
            self.regions.coordinates["strand"] == -1
        )
        self.regions.coordinates["end_inclusive"] = self.regions.coordinates["end"] + (
            self.regions.coordinates["strand"] == -1
        )

        lastupdate = time.time()
        i = 0

        for parentregion, subcoordinates in self.regions.coordinates.groupby(self.parentregion_column):
            # extract in which parent region we need to look
            parentregion_ix = parentregion_to_parentregion_ix(parentregion)

            for cell_ix in range(self.n_cells):
                # extract the parent region coordinates per cell
                parentregionxcell_ix = parentregion_ix * self.parent.n_cells + cell_ix
                parentregion_start_ix, parentregion_end_ix = parent_regionxcell_indptr[
                    parentregionxcell_ix : parentregionxcell_ix + 2
                ]

                coordinates_parentregion = coordinates[parentregion_start_ix:parentregion_end_ix]

                for region_id, region in subcoordinates.iterrows():
                    # extract fragments lying within the region
                    region_ix = region["ix"]
                    regionxcell_ix = region_ix * self.n_cells + cell_ix

                    if (pbar is not None) and (time.time() - lastupdate > 1):
                        pbar.update(i + 1)
                        i = 0
                        pbar.set_description(f"Processing region {region_id} {cell_ix}")
                        lastupdate = time.time()
                    else:
                        i += 1

                    # extract fragments lying within the region
                    if inclusive == (True, False):
                        n_excluded_left = np.searchsorted(coordinates_parentregion[:, 0], region["start_inclusive"])
                        n_included = (
                            np.searchsorted(coordinates_parentregion[:, 0], region["end_inclusive"]) - n_excluded_left
                        )
                    else:
                        raise NotImplementedError("For now, only left-inclusive fragments are supported")

                    regionxcell_indptr[regionxcell_ix] = [
                        parentregion_start_ix + n_excluded_left,
                        parentregion_start_ix + n_excluded_left + n_included,
                    ]

        regionxcell_indptr_writer = self.regionxcell_indptr.open_creator(
            shape=(len(self.regions.coordinates) * len(self.obs), 2)
        )
        regionxcell_indptr_writer[:] = regionxcell_indptr

        return self

    def filter_regions(self, regions: Regions, path: PathLike = None, overwrite=True) -> Fragments:
        """
        Filter based on new regions

        Parameters:
            regions:
                Regions to filter.
        Returns:
            A new Fragments object
        """

        # check if new regions are a subset of the existing ones
        if not regions.coordinates.index.isin(self.regions.coordinates.index).all():
            raise ValueError("New regions should be a subset of the existing ones")

        # create new fragments
        fragments = FragmentsView.create(
            parent=self.parent,
            regions=regions,
            path=path,
            parentregion_column=self.parentregion_column,
            obs=self.obs,
            var=regions.coordinates,
            reset=overwrite,
        )

        return fragments

    var: pd.DataFrame = TSV()
    """DataFrame containing information about regions."""

    obs: pd.DataFrame = TSV()
    """DataFrame containing information about cells."""

    @functools.cached_property
    def n_regions(self):
        """Number of regions"""
        return self.var.shape[0]

    @functools.cached_property
    def n_cells(self):
        """Number of cells"""
        return self.obs.shape[0]

    def estimate_fragment_per_cellxregion(self) -> int:
        """
        Estimate the expected number of fragments per regionxcell combination. This is used to estimate the buffer size for loading fragments.
        """
        return math.ceil((self.regionxcell_indptr[:, 1] - self.regionxcell_indptr[:, 0]).astype(float).mean())

    @property
    def coordinates(self):
        """
        Coordinates of the fragments, equal to the parent coordinates
        """
        return self.parent.coordinates

    @property
    def mapping(self):
        """
        Mapping of the fragments, equal to the parent mapping
        """
        return self.parent.mapping

    @property
    def counts(self):
        """
        Number of fragments per region and cell
        """
        return (self.regionxcell_indptr[:, 1] - self.regionxcell_indptr[:, 0]).reshape(self.regions.n_regions, -1).T

    _cache = None

    def get_cache(self, region_oi):
        """
        Get the cache for a specific region
        """

        if self._cache is None:
            self._cache = {}

        if region_oi in self._cache:
            return self._cache[region_oi]

        region_ix = self.regions.coordinates.index.get_loc(region_oi)
        regionxcell_ixs = region_ix * self.n_cells + np.arange(self.n_cells)

        indptrs = self.regionxcell_indptr[regionxcell_ixs]

        coordinates_reader = self.parent.coordinates.open_reader()

        n = []
        i = 0
        coordinates = []
        for start, end in indptrs:
            coordinates.append(coordinates_reader[start:end])
            i += end - start
            n.append(end - start)

        coordinates = np.concatenate(coordinates)
        local_cellxregion_ix = np.repeat(np.arange(len(indptrs)), n)
        regionxcell_indptr = indices_to_indptr(local_cellxregion_ix, len(indptrs), dtype=np.int64)

        self._cache[region_oi] = {
            "regionxcell_indptr": regionxcell_indptr,
            "coordinates": coordinates,
        }

        return self._cache[region_oi]

    _libsize = None

    @property
    def libsize(self):
        if self._libsize is None:
            self._libsize = np.bincount(self.mapping[:, 0], minlength=self.n_cells)
        return self._libsize

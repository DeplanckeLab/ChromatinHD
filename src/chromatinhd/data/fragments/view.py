from __future__ import annotations
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import math
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
            raise ValueError("regions should be a Regions object")

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
        regionxcell_indptr = self.regionxcell_indptr.open_creator(
            shape=(len(self.regions.coordinates) * len(self.obs), 2)
        )

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

        return self

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


# class FragmentsViewOld(Flow):
#     """
#     A view of fragments based on regions that are a subset of the parent fragments. In a typical use case, the parent contains fragments for all chromosomes, while this the view focuses on specific regions.

#     Only fragments that are fully inclusive (i.e., both left and right cut sites) within a region will be selected.
#     """

#     regionxcell_fragmentixs: TensorstoreInstance = Tensorstore(dtype="<i8", chunks=[100000], compression="blosc")
#     """Index of fragments in the parent fragments object"""

#     regionxcell_fragmentixs_indptr: TensorstoreInstance = Tensorstore(dtype="<i8", chunks=[100000], compression="blosc")
#     """Index pointers in the regionxcell_fragmentixs array for each regionxcell"""

#     parent: Fragments = Linked()
#     """The parent fragments object from which this view is created"""

#     regions: Regions = Linked()
#     """The regions object"""

#     parenregion_column = Stored()

#     @classmethod
#     def from_fragments(
#         cls,
#         parent: Fragments,
#         regions: Regions,
#         parentregion_column: str = "chrom",
#         obs: pd.DataFrame = None,
#         var: pd.DataFrame = None,
#         path: PathLike = None,
#         overwrite: bool = False,
#     ):
#         """
#         Creates a fragments view from a parent fragments object and a regions object

#         Parameters:
#             parent:
#                 Parent fragments object. If a fragments view is provided, the parent of the parent will be used.
#             regions:
#                 Regions object
#             obs:
#                 DataFrame containing information about cells, will be copied from the fragments object if not provided
#             path:
#                 Path to store the fragments view
#         """

#         if isinstance(parent, FragmentsView):
#             while isinstance(parent, FragmentsView):
#                 parent = parent.parent
#         if not isinstance(parent, Fragments):
#             raise ValueError("parent should be a Fragments object")
#         if not isinstance(regions, Regions):
#             raise ValueError("regions should be a Regions object")

#         self = cls.create(
#             parent=parent,
#             regions=regions,
#             path=path,
#             obs=parent.obs if obs is None else obs,
#             var=regions.coordinates if var is None else var,
#             reset=overwrite,
#         )

#         return self

#     def create_regionxcell_indptr(
#         self,
#         batch_size: int = 2000,
#         inclusive: bool = True,
#         overwrite=True,
#     ) -> FragmentsView:
#         """
#         Create the index pointers that can be used for fast access to fragments of a particular regionxcell combination

#         Parameters:
#             parentregion_column:
#                 Column in the regions coordinates that links each new region to the regions of the original fragments. This is typically the chromosome column. This column should be present in both `parent.regions.coordinates` and `regions.coordinates`
#             batch_size:
#                 Number of regions to wait before saving the intermediate results and freeing up memory. Reduce this number to avoid running out of memory.
#             inclusive:
#                 Whether to only include fragments that are only partially overlapping with the region. If False, partially overlapping fragments will be selected as well.
#             overwrite:
#                 Whether to overwrite the existing index pointers.

#         Returns:
#             Same object but with the `regionxcell_fragmentixs_indptr` and `regionxcell_fragmentixs` populated

#         """
#         # dummy proofing
#         if parentregion_column not in self.regions.coordinates.columns:
#             raise ValueError(
#                 f"Column {parentregion_column} not in regions coordinates. Available columns are {self.regions.coordinates.columns}"
#             )
#         if parentregion_column not in self.parent.regions.coordinates.columns:
#             raise ValueError(
#                 f"Column {parentregion_column} not in fragments regions coordinates. Available columns are {self.parent.regions.coordinates.columns}"
#             )
#         if not (
#             self.regions.coordinates[parentregion_column].isin(self.parent.regions.coordinates[parentregion_column])
#         ).all():
#             raise ValueError(
#                 f"Not all regions are present in the parent fragments. Missing regions: {self.regions.coordinates[parentregion_column][~self.regions.coordinates[parentregion_column].isin(self.parent.regions.coordinates[parentregion_column])]}"
#             )

#         if self.regionxcell_fragmentixs_indptr.exists() and not overwrite:
#             return self

#         mapping = self.parent.mapping[:]
#         coordinates = self.parent.coordinates[:]

#         # convert regions in parent to parent region ixs
#         self.parent.regions.coordinates["ix"] = np.arange(len(self.parent.regions.coordinates))
#         parentregion_to_parentregion_ix = self.parent.regions.coordinates["ix"].to_dict()

#         # set up empty arrays for the 0th iteration
#         fragmentixs_all = []
#         regionxcell_fragmentixs_indptr = []
#         region_ixs = []

#         fragmentixs_counter = 0

#         # reset the tensorstores
#         self.regionxcell_fragmentixs_indptr.open_creator(shape=(0,))
#         self.regionxcell_fragmentixs.open_creator(shape=(0,))

#         # index pointers from parentregions to fragments
#         parentregion_fragment_indptr = indices_to_indptr(mapping[:, 1], self.parent.n_regions)

#         loop = tqdm.tqdm(
#             enumerate(self.regions.coordinates.iterrows()),
#             total=len(self.regions.coordinates),
#             leave=False,
#             desc="Processing regions",
#         )
#         for region_ix, (region_id, region) in loop:
#             loop.set_description(f"Processing region {region_id}")

#             # extract in which parent region we need to look
#             parentregion_ix = parentregion_to_parentregion_ix[region[parentregion_column]]

#             parentregion_start_ix, parentregion_end_ix = (
#                 parentregion_fragment_indptr[parentregion_ix],
#                 parentregion_fragment_indptr[parentregion_ix + 1],
#             )

#             # extract parent's mapping and coordinates
#             coordinates_parentregion = coordinates[parentregion_start_ix:parentregion_end_ix]
#             cellmapping_parentregion = mapping[parentregion_start_ix:parentregion_end_ix, 0]

#             # extract fragments lying within the region
#             # depending on the strand, we either need to include the start or the end
#             region["start_inclusive"] = region["start"] + (region["strand"] == -1)
#             region["end_inclusive"] = region["end"] + (region["strand"] == -1)

#             if inclusive:
#                 fragments_oi = (coordinates_parentregion[:, 0] >= region["start_inclusive"]) & (
#                     coordinates_parentregion[:, 1] < region["end_inclusive"]
#                 )
#             else:
#                 raise NotImplementedError("For now, only inclusive fragments are supported")
#             local_cellmapping = cellmapping_parentregion[fragments_oi]

#             local_fragmentixs = np.argsort(local_cellmapping).astype(np.int64)

#             local_cell_fragmentixs_indptr = indices_to_indptr(
#                 local_cellmapping[local_fragmentixs], self.parent.n_cells, dtype=np.int64
#             )

#             regionxcell_fragmentixs = np.where(fragments_oi)[0] + parentregion_start_ix
#             cell_fragmentixs_indptr = local_cell_fragmentixs_indptr + fragmentixs_counter

#             fragmentixs_counter += len(local_fragmentixs)

#             fragmentixs_all.append(regionxcell_fragmentixs)
#             regionxcell_fragmentixs_indptr.append(
#                 cell_fragmentixs_indptr[:-1]
#             )  # remove the last pointer as it will be duplicated if there is a next region

#             region_ixs.append(region_ix)

#             # check whether we want to save to disk based on the number of fragments processed
#             # especially for wide or high number of regions this can be a game changer in memory consumption
#             if len(fragmentixs_all) > batch_size:
#                 loop.set_description("Saving to disk")
#                 # save to disk
#                 regionxcell_fragmentixs = np.concatenate(fragmentixs_all)
#                 regionxcell_fragmentixs_indptr = np.concatenate(regionxcell_fragmentixs_indptr)

#                 self.regionxcell_fragmentixs.extend(regionxcell_fragmentixs)
#                 self.regionxcell_fragmentixs_indptr.extend(regionxcell_fragmentixs_indptr)

#                 region_ixs = []
#                 fragmentixs_all = []
#                 regionxcell_fragmentixs_indptr = []

#         # add the last pointer if it is the last region
#         regionxcell_fragmentixs_indptr.append(np.array([fragmentixs_counter], dtype=np.int64))

#         # store data from final iteration
#         regionxcell_fragmentixs = np.concatenate(fragmentixs_all)
#         regionxcell_fragmentixs_indptr = np.concatenate(regionxcell_fragmentixs_indptr)

#         self.regionxcell_fragmentixs.extend(regionxcell_fragmentixs)
#         self.regionxcell_fragmentixs_indptr.extend(regionxcell_fragmentixs_indptr)

#         return self

#     var: pd.DataFrame = TSV()
#     """DataFrame containing information about regions."""

#     obs: pd.DataFrame = TSV()
#     """DataFrame containing information about cells."""

#     @functools.cached_property
#     def n_regions(self):
#         """Number of regions"""
#         return self.var.shape[0]

#     @functools.cached_property
#     def n_cells(self):
#         """Number of cells"""
#         return self.obs.shape[0]

#     def estimate_fragment_per_cellxregion(self) -> int:
#         """
#         Estimate the expected number of fragments per regionxcell combination. This is used to estimate the buffer size for loading fragments.
#         """
#         return math.ceil(self.regionxcell_fragmentixs.shape[0] / self.n_cells / self.n_regions)

#     @property
#     def coordinates(self):
#         """
#         Coordinates of the fragments, equal to the parent coordinates
#         """
#         return self.parent.coordinates

#     @property
#     def mapping(self):
#         """
#         Mapping of the fragments, equal to the parent mapping
#         """
#         return self.parent.mapping

#     @property
#     def regionxcell_counts(self):
#         """
#         Number of fragments per region and cell
#         """
#         return np.diff(self.regionxcell_fragmentixs_indptr[:]).reshape(self.regions.n_regions, -1)

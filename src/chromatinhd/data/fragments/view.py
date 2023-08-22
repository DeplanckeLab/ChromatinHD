from __future__ import annotations
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import math
import functools

from chromatinhd.data.fragments import Fragments
from chromatinhd.data.regions import Regions
from chromatinhd.flow import TSV, Flow, Linked, PathLike
from chromatinhd.flow.tensorstore import Tensorstore, TensorstoreInstance
from chromatinhd.utils.numpy import indices_to_indptr


compression = {
    "id": "blosc",
    "clevel": 3,
    "cname": "zstd",
    "shuffle": 2,
}
# compression = None
regionxcell_fragmentixs_indptr_spec = {
    "driver": "zarr",
    "kvstore": {
        "driver": "file",
    },
    "metadata": {
        "compressor": compression,
        "dtype": ">i8",
        "shape": [0],
        "chunks": [100000],
    },
}
fragmentixs_spec = {
    "driver": "zarr",
    "kvstore": {
        "driver": "file",
    },
    "metadata": {
        "compressor": compression,
        "dtype": ">i8",
        "shape": [0],
        "chunks": [100000],
    },
}


class FragmentsView(Flow):
    """
    A view of fragments based on regions that are a subset of the parent fragments. In a typical use case, the parent contains fragments for all chromosomes, while this the view focuses on specific regions.

    Only fragments that are fully inclusive (i.e., both left and right cut sites) within a region will be selected.
    """

    parent: Fragments = Linked()
    """The parent fragments object from which this view is created"""

    regions: Regions = Linked()
    """The regions object"""

    fragmentixs: TensorstoreInstance = Tensorstore(fragmentixs_spec)
    """Index of fragments in the parent fragments object"""

    regionxcell_fragmentixs_indptr: TensorstoreInstance = Tensorstore(regionxcell_fragmentixs_indptr_spec)
    """Index pointers in the fragmentixs array for each regionxcell"""

    @classmethod
    def from_fragments(
        cls,
        parent: Fragments,
        regions: Regions,
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

        self = cls.create(
            parent=parent,
            regions=regions,
            path=path,
            obs=parent.obs if obs is None else obs,
            var=regions.coordinates if var is None else var,
            reset=overwrite,
        )

        return self

    def create_regionxcell_indptr(
        self, parentregion_column: str = "chrom", batch_size: int = 500, inclusive: bool = True
    ) -> FragmentsView:
        """
        Create the index pointers that can be used for fast access to fragments of a particular regionxcell combination

        Parameters:
            parentregion_column:
                Column in the regions coordinates that links each new region to the regions of the original fragments. This is typically the chromosome column. This column should be present in both `parent.regions.coordinates` and `regions.coordinates`
            batch_size:
                Number of regions to wait before saving the intermediate results and freeing up memory. Reduce this number to avoid running out of memory.
            inclusive:
                Whether to only include fragments that are only partially overlapping with the region. If False, partially overlapping fragments will be selected as well.

        Returns:
            The same object, but with the regionxcell_fragmentixs_indptr and fragmentixs tensorstores filled

        """
        # dummy proofing
        if parentregion_column not in self.regions.coordinates.columns:
            raise ValueError(
                f"Column {parentregion_column} not in regions coordinates. Available columns are {self.regions.coordinates.columns}"
            )
        if parentregion_column not in self.parent.regions.coordinates.columns:
            raise ValueError(
                f"Column {parentregion_column} not in fragments regions coordinates. Available columns are {self.parent.regions.coordinates.columns}"
            )
        if not (
            self.regions.coordinates[parentregion_column].isin(self.parent.regions.coordinates[parentregion_column])
        ).all():
            raise ValueError(
                f"Not all regions are present in the parent fragments. Missing regions: {self.regions.coordinates[parentregion_column][~self.regions.coordinates[parentregion_column].isin(self.parent.regions.coordinates[parentregion_column])]}"
            )

        mapping = self.parent.mapping[:]
        coordinates = self.parent.coordinates[:]

        self.parent.regions.coordinates["ix"] = np.arange(len(self.parent.regions.coordinates))
        parentregion_to_parentregion_ix = self.parent.regions.coordinates["ix"].to_dict()

        fragmentixs_all = []
        fragmentixs_counter = 0
        regionxcell_fragmentixs_indptr = []
        region_ixs = []

        # resize regionxcell to the number of cells * regions
        self.regionxcell_fragmentixs_indptr.open_writer().resize(
            exclusive_max=[self.n_cells * self.n_regions + 1]
        ).result()

        # index pointers from parentregions to fragments
        parentregion_fragment_indptr = indices_to_indptr(mapping[:, 1], self.parent.n_regions)

        loop = tqdm.tqdm(
            enumerate(self.regions.coordinates.iterrows()),
            total=len(self.regions.coordinates),
            leave=False,
            desc="Processing regions",
        )
        for region_ix, (region_id, region) in loop:
            loop.set_description(f"Processing region {region_id}")
            parentregion_ix = parentregion_to_parentregion_ix[region[parentregion_column]]

            parentregion_start_ix, parentregion_end_ix = (
                parentregion_fragment_indptr[parentregion_ix],
                parentregion_fragment_indptr[parentregion_ix + 1],
            )

            coordinates_parentregion = coordinates[parentregion_start_ix:parentregion_end_ix]
            mapping_parentregion = mapping[parentregion_start_ix:parentregion_end_ix]

            if inclusive:
                fragments_oi = (coordinates_parentregion[:, 0] >= region["start"]) & (
                    coordinates_parentregion[:, 1] < region["end"]
                )
            else:
                raise NotImplementedError("For now, only inclusive fragments are supported")
            local_cellmapping = mapping_parentregion[fragments_oi, 0]

            local_fragmentixs = np.argsort(local_cellmapping).astype(np.int64)

            local_cell_fragmentixs_indptr = indices_to_indptr(
                local_cellmapping[local_fragmentixs], self.parent.n_cells, dtype=np.int64
            )

            fragmentixs = np.where(fragments_oi)[0] + parentregion_start_ix
            cell_fragmentixs_indptr = local_cell_fragmentixs_indptr + fragmentixs_counter

            fragmentixs_counter += len(local_fragmentixs)

            fragmentixs_all.append(fragmentixs)
            regionxcell_fragmentixs_indptr.append(
                cell_fragmentixs_indptr[:-1]
            )  # remove the last pointer as it will be duplicated if there is a next region

            region_ixs.append(region_ix)

            # check whether we want to save to disk based on the number of fragments processed
            # especially for wide or high number of regions this can be a game changer in memory consumption
            if len(fragmentixs_all) > batch_size:
                loop.set_description("Saving to disk")
                # save to disk
                fragmentixs = np.concatenate(fragmentixs_all)
                regionxcell_fragmentixs_indptr = np.concatenate(regionxcell_fragmentixs_indptr)

                self.fragmentixs.open_writer().resize(exclusive_max=[fragmentixs_counter]).result()

                self.fragmentixs[fragmentixs_counter - len(fragmentixs) : fragmentixs_counter] = fragmentixs
                self.regionxcell_fragmentixs_indptr[
                    region_ixs[0] * self.n_cells : (region_ixs[-1] + 1) * self.n_cells
                ] = regionxcell_fragmentixs_indptr

                region_ixs = []
                fragmentixs_all = []
                regionxcell_fragmentixs_indptr = []

        regionxcell_fragmentixs_indptr.append(
            np.array([fragmentixs_counter], dtype=np.int64)
        )  # add the last pointer if it is the last region

        fragmentixs = np.concatenate(fragmentixs_all)
        regionxcell_fragmentixs_indptr = np.concatenate(regionxcell_fragmentixs_indptr)

        self.fragmentixs.open_writer().resize(exclusive_max=[fragmentixs_counter]).result()

        self.fragmentixs[fragmentixs_counter - len(fragmentixs) : fragmentixs_counter] = fragmentixs
        self.regionxcell_fragmentixs_indptr[
            region_ixs[0] * self.n_cells : ((region_ixs[-1] + 1) * self.n_cells) + 1
        ] = regionxcell_fragmentixs_indptr

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

    def estimate_fragment_per_regionxcell(self) -> int:
        """
        Estimate the expected number of fragments per regionxcell combination. This is used to estimate the buffer size for loading fragments.
        """
        return math.ceil(self.fragmentixs.shape[0] / self.n_cells / self.n_regions * 2)

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

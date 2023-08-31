from __future__ import annotations
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import math
import functools

from chromatinhd.data.motifscan import Motifscan
from chromatinhd.data.regions import Regions
from chromatinhd.flow import TSV, Flow, Linked, PathLike
from chromatinhd.flow.tensorstore import Tensorstore, TensorstoreInstance


class MotifscanView(Flow):
    """
    A view of motifscan based on regions that are a subset of the parent motifscan. In a typical use case, the parent contains motifscan for all chromosomes, while this the view focuses on specific regions.
    """

    parent: Motifscan = Linked()
    """The parent motifscan object from which this view is created"""

    regions: Regions = Linked()
    """The regions object"""

    region_indptr: TensorstoreInstance = Tensorstore(shape=(0, 2), dtype="<i8", chunks=(100, 2), compression=None)
    """Index pointers for each region. The first column is the start of the region in the parent motifscan, the second column is the end of the region in the parent motifscan"""

    @property
    def motifs(self):
        """
        Motifs of the motifscan, equal to the parent motifs
        """
        return self.parent.motifs

    @property
    def coordinates(self):
        """
        Coordinates of the motif sites, equal to the parent coordinates
        """
        return self.parent.coordinates

    @classmethod
    def from_motifscan(
        cls,
        parent: Motifscan,
        regions: Regions,
        path: PathLike = None,
        overwrite: bool = False,
    ):
        """
        Creates a motifscan view from a parent motifscan object and a regions object

        Parameters:
            parent:
                Parent motifscan object. If a motifscan view is provided, the parent of the parent will be used.
            regions:
                Regions object
            obs:
                DataFrame containing information about cells, will be copied from the motifscan object if not provided
            path:
                Path to store the motifscan view
        """

        if isinstance(parent, MotifscanView):
            while isinstance(parent, MotifscanView):
                parent = parent.parent
        if not isinstance(parent, Motifscan):
            raise ValueError("parent should be a Motifscan object")
        if not isinstance(regions, Regions):
            raise ValueError("regions should be a Regions object")

        self = cls.create(
            parent=parent,
            regions=regions,
            path=path,
            reset=overwrite,
        )

        return self

    def create_region_indptr(self, parentregion_column: str = "chrom") -> MotifscanView:
        """
        Create the index pointers that can be used for fast access to motifscan of a particular regionxcell combination

        Parameters:
            parentregion_column:
                Column in the regions coordinates that links each new region to the regions of the original motifscan. This is typically the chromosome column. This column should be present in both `parent.regions.coordinates` and `regions.coordinates`

        Returns:
            Same object but with the `region_indptr` populated

        """
        # dummy proofing
        if parentregion_column not in self.regions.coordinates.columns:
            raise ValueError(
                f"Column {parentregion_column} not in regions coordinates. Available columns are {self.regions.coordinates.columns}"
            )
        if parentregion_column not in self.parent.regions.coordinates.columns:
            raise ValueError(
                f"Column {parentregion_column} not in motifscan regions coordinates. Available columns are {self.parent.regions.coordinates.columns}"
            )
        if not (
            self.regions.coordinates[parentregion_column].isin(self.parent.regions.coordinates[parentregion_column])
        ).all():
            raise ValueError(
                f"Not all regions are present in the parent motifscan. Missing regions: {self.regions.coordinates[parentregion_column][~self.regions.coordinates[parentregion_column].isin(self.parent.regions.coordinates[parentregion_column])]}"
            )

        # mapping = self.parent.mapping[:]
        # coordinates = self.parent.coordinates[:]

        # # convert regions in parent to parent region ixs
        self.parent.regions.coordinates["ix"] = np.arange(len(self.parent.regions.coordinates))
        parentregion_to_parentregion_ix = self.parent.regions.coordinates["ix"].to_dict()

        region_indptr = self.region_indptr.open_creator(shape=[int(self.regions.n_regions), 2])

        self.regions.coordinates["ix"] = np.arange(len(self.regions.coordinates))

        grouped = self.regions.coordinates.groupby(parentregion_column)
        loop = tqdm.tqdm(
            grouped,
            total=len(grouped),
            leave=False,
            desc="Processing regions",
        )
        for parentregion_id, regions_subset in loop:
            loop.set_description(f"Processing region {parentregion_id}")

            parentregion_ix = parentregion_to_parentregion_ix[parentregion_id]
            superregion_indptr_start, superregion_indptr_end = (
                self.parent.region_indptr[parentregion_ix].item(),
                self.parent.region_indptr[parentregion_ix + 1].item(),
            )

            parentcoordinates = self.parent.coordinates[superregion_indptr_start:superregion_indptr_end]

            indptr_start = np.searchsorted(parentcoordinates, regions_subset["start"])
            indptr_end = np.searchsorted(parentcoordinates, regions_subset["end"])

            indptr = np.stack([superregion_indptr_start + indptr_start, superregion_indptr_start + indptr_end], axis=1)

            region_indptr[regions_subset["ix"].values] = indptr

        return self

from __future__ import annotations
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import math
import functools

from chromatinhd.data.motifscan import Motifscan
from chromatinhd.data.regions import Regions
from chromatinhd.flow import TSV, Flow, Linked, PathLike, Stored
from chromatinhd.flow.tensorstore import Tensorstore, TensorstoreInstance


class MotifscanView(Flow):
    """
    A view of a motifscan, based on regions that are a subset of the parent motifscan. In a typical use case, the parent contains motifs for all chromosomes, while this view focuses on specific regions.
    """

    parent: Motifscan = Linked()
    """The parent motifscan object from which this view is created"""

    regions: Regions = Linked()
    """The regions object"""

    region_indptr: TensorstoreInstance = Tensorstore(shape=(0, 2), dtype="<i8", chunks=(100, 2), compression=None)
    """Index pointers for each region. The first column is the start of the region in the parent motifscan, the second column is the end of the region in the parent motifscan"""

    parentregion_column = Stored()
    """Column in the regions coordinates that links each new region to the regions of the original motifscan. This is typically the chromosome column. This column should be present in `regions.coordinates` and should refer to the index of the parent regions."""

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
        parent_region_column: str = "chrom",
        path: PathLike = None,
        overwrite: bool = False,
    ) -> MotifscanView:
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
            raise ValueError(f"parent should be a Motifscan object, {type(parent)} provided")
        if not isinstance(regions, Regions):
            raise ValueError("regions should be a Regions object")

        self = cls.create(
            parent=parent,
            regions=regions,
            parentregion_column=parent_region_column,
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

        # convert regions in parent to parent region ixs
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

    def get_slice(
        self,
        region_id=None,
        region_ix=None,
        start=None,
        end=None,
        return_scores=True,
        return_strands=True,
        return_indptr=False,
        motif_ixs=None,
    ) -> tuple:
        """
        Get the positions/scores/strandedness of motifs within a slice of the motifscan

        Parameters:
            region:
                Region id
            region_ix:
                Region index
            start:
                Start of the slice, in region coordinates
            end:
                End of the slice, in region coordinates

        Returns:
            Motifs positions, indices, scores and strands of the slice
        """
        # fix the readers of the parent
        self.parent.indices.fix_reader()
        self.parent.indptr.fix_reader()
        self.parent.scores.fix_reader()
        self.parent.strands.fix_reader()
        self.parent.region_indptr.fix_reader()
        self.parent.coordinates.fix_reader()

        if region_id is not None:
            region = self.regions.coordinates.loc[region_id]
        elif region_ix is not None:
            region = self.regions.coordinates.iloc[region_ix]
        else:
            raise ValueError("Either region or region_ix should be provided")
        region_id = region.name
        parentregion_to_parentregion_ix = self.parent.regions.coordinates.index.get_loc
        parentregion_ix = parentregion_to_parentregion_ix(
            self.regions.coordinates.loc[region_id, self.parentregion_column]
        )

        # get start and end, either whole region or subset
        if start is None:
            parent_start = int(region["start"])
        else:
            if region["strand"] == 1:
                parent_start = int(region["tss"] + start)
            else:
                parent_start = int(region["tss"] - start)
        if end is None:
            parent_end = int(region["end"])
        else:
            if region["strand"] == 1:
                parent_end = int(region["tss"] + end)
            else:
                parent_end = int(region["tss"] - end)

        if parent_start > parent_end:
            parent_start, parent_end = parent_end, parent_start

        parent_cumulative_start = np.clip(
            self.parent.regions.cumulative_region_lengths[parentregion_ix] + parent_start,
            self.parent.regions.cumulative_region_lengths[parentregion_ix],
            self.parent.regions.cumulative_region_lengths[parentregion_ix + 1],
        )
        parent_cumulative_end = np.clip(
            self.parent.regions.cumulative_region_lengths[parentregion_ix] + parent_end,
            self.parent.regions.cumulative_region_lengths[parentregion_ix],
            self.parent.regions.cumulative_region_lengths[parentregion_ix + 1],
        )

        indptr = self.parent.indptr[parent_cumulative_start : (parent_cumulative_end + 1)]

        indptr_start, indptr_end = indptr[0], indptr[-1]

        positions = (self.parent.coordinates[indptr_start:indptr_end] - region["tss"]) * region["strand"]
        indices = self.parent.indices[indptr_start:indptr_end]

        out = [positions, indices]

        if return_scores:
            out.append(self.parent.scores[indptr_start:indptr_end])
        if return_strands:
            out.append(self.parent.strands[indptr_start:indptr_end])

        if motif_ixs is not None:
            selection = np.isin(indices, motif_ixs)
            out = [x[selection] for x in out]

        if return_indptr:
            out.append(indptr - indptr_start)

        return out

    def count_slices(self, slices: pd.DataFrame) -> pd.DataFrame:
        """
        Get multiple slices of the motifscan

        Parameters:
            slices:
                DataFrame containing the slices to get. Each row should contain a region_ix, start and end column. The region_ix should refer to the index of the regions object. The start and end columns should contain the start and end of the slice, in region coordinates.

        Returns:
            DataFrame containing the counts of each motif (columns) in each slice (rows)
        """

        if self.regions.window is None:
            raise NotImplementedError("count_slices is only implemented for regions with a window")

        if "region_ix" not in slices:
            slices["region_ix"] = self.regions.coordinates.index.get_indexer(slices["region"])

        progress = enumerate(zip(slices["start"], slices["end"], slices["region_ix"]))
        progress = tqdm.tqdm(
            progress,
            total=len(slices),
            leave=False,
            desc="Counting slices",
            mininterval=1,
        )

        motif_counts = np.zeros((len(slices), self.n_motifs), dtype=int)
        for i, (relative_start, relative_end, region_ix) in progress:
            start = relative_start
            end = relative_end
            positions, indices = self.get_slice(
                region_ix=region_ix, start=start, end=end, return_scores=False, return_strands=False
            )
            motif_counts[i] = np.bincount(indices, minlength=self.n_motifs)
        motif_counts = pd.DataFrame(motif_counts, index=slices.index, columns=self.motifs.index)
        return motif_counts

    @property
    def n_motifs(self) -> int:
        """
        Number of motifs
        """
        return self.parent.n_motifs

    @property
    def scanned(self) -> bool:
        """
        Whether the motifscan is scanned
        """
        if self.o.parent.exists(self):
            return self.parent.scanned
        return False

    select_motif = Motifscan.select_motif
    select_motifs = Motifscan.select_motifs

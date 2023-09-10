from __future__ import annotations
import math

import numpy as np
import pandas as pd
import torch
import functools
import tqdm.auto as tqdm

from chromatinhd.data.regions import Regions
from chromatinhd.flow import TSV, Flow, Linked, Stored, StoredTensor, PathLike
from chromatinhd.flow.tensorstore import Tensorstore, TensorstoreInstance
from chromatinhd.utils import class_or_instancemethod

import pathlib


class RawFragments:
    def __init__(self, file):
        self.file = file


class Fragments(Flow):
    """
    Fragments positioned within regions. Fragments are sorted by the region, position within the region (left cut site) and cell.

    The object can also store several precalculated tensors that are used for efficient loading of fragments. See create_regionxcell_indptr for more information.
    """

    regions: Regions = Linked()
    """The regions in which (part of) the fragments are located and centered."""

    # coordinates: TensorstoreInstance = Tensorstore(dtype="<i4", chunks=None, compression=None, shape=[0, 2])
    coordinates: TensorstoreInstance = Tensorstore(dtype="<i4", chunks=[100000, 2], compression=None, shape=[0, 2])
    # coordinates: TensorstoreInstance = Tensorstore(dtype="<i4", chunks=[100000, 2], compression="blosc", shape=[0, 2])
    """Coordinates of the two cut sites."""

    mapping: TensorstoreInstance = Tensorstore(dtype="<i4", chunks=[100000, 2], compression="blosc", shape=[0, 2])
    """Mapping of a fragment to a cell (first column) and a region (second column)"""

    regionxcell_indptr: np.ndarray = Tensorstore(dtype="<i8", chunks=[100000], compression="blosc")
    """Index pointers to the regionxcell fragment positions"""

    def create_regionxcell_indptr(self) -> Fragments:
        """
        Creates pointers to each individual region x cell combination from the mapping tensor.

        Returns:
            The same object with `regionxcell_indptr` populated
        """
        regionxcell_ix = self.mapping[:, 1] * self.n_cells + self.mapping[:, 0]

        if not (np.diff(regionxcell_ix) >= 0).all():
            raise ValueError("Fragments should be ordered by regionxcell (ascending)")

        n_regionxcell = self.n_regions * self.n_cells
        regionxcell_indptr = np.pad(np.cumsum(np.bincount(regionxcell_ix, minlength=n_regionxcell), 0), (1, 0))
        assert self.coordinates.shape[0] == regionxcell_indptr[-1]
        if not (np.diff(regionxcell_indptr) >= 0).all():
            raise ValueError("Fragments should be ordered by regionxcell (ascending)")
        self.regionxcell_indptr[:] = regionxcell_indptr

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

    @property
    def local_cellxregion_ix(self):
        return self.mapping[:, 0] * self.n_regions + self.mapping[:, 1]

    def estimate_fragment_per_cellxregion(self):
        return math.ceil(self.coordinates.shape[0] / self.n_cells / self.n_regions)

    @class_or_instancemethod
    def from_fragments_tsv(
        cls,
        fragments_file: PathLike,
        regions: Regions,
        obs: pd.DataFrame,
        cell_column: str = None,
        path: PathLike = None,
        overwrite: bool = True,
        reuse: bool = False,
    ) -> Fragments:
        """
        Create a Fragments object from a fragments tsv file

        Parameters:
            fragments_file:
                Location of the fragments tab-separate file created by e.g. CellRanger or sinto
            obs:
                DataFrame containing information about cells.
                The index should be the cell names as present in the fragments file.
                Alternatively, the column containing cell ids can be specified using the `cell_column` argument.
            regions:
                Regions from which the fragments will be extracted.
            cell_column:
                Column name in the `obs` DataFrame containing the cell names.
                If None, the index of the `obs` DataFrame is used.
            path:
                Folder in which the fragments data will be stored.
            overwrite:
                Whether to overwrite the data if it already exists.
            reuse:
                Whether to reuse existing data if it exists.
        Returns:
            A new Fragments object
        """

        if isinstance(fragments_file, str):
            fragments_file = pathlib.Path(fragments_file)
        if isinstance(path, str):
            path = pathlib.Path(path)
        if not fragments_file.exists():
            raise FileNotFoundError(f"File {fragments_file} does not exist")
        if not overwrite and path.exists() and not reuse:
            raise FileExistsError(
                f"Folder {path} already exists, use `overwrite=True` to overwrite, or `reuse=True` to reuse existing data"
            )

        # regions information
        var = pd.DataFrame(index=regions.coordinates.index)
        var["ix"] = np.arange(var.shape[0])

        # cell information
        obs = obs.copy()
        obs["ix"] = np.arange(obs.shape[0])
        if cell_column is None:
            cell_to_cell_ix = obs["ix"].to_dict()
        else:
            cell_to_cell_ix = obs.set_index(cell_column)["ix"].to_dict()

        self = cls.create(path=path, obs=obs, var=var, regions=regions, reset=overwrite)

        # read the fragments file
        try:
            import pysam
        except ImportError as e:
            raise ImportError(
                "pysam is required to read fragments files. Install using `pip install pysam` or `conda install -c bioconda pysam`"
            ) from e
        fragments_tabix = pysam.TabixFile(str(fragments_file))

        # process regions
        pbar = tqdm.tqdm(
            enumerate(regions.coordinates.iterrows()),
            total=regions.coordinates.shape[0],
            leave=False,
            desc="Processing fragments",
        )

        self.mapping.open_creator()
        self.coordinates.open_creator()

        for region_ix, (region_id, region_info) in pbar:
            pbar.set_description(f"{region_id}")
            fetched = fragments_tabix.fetch(
                region_info["chrom"],
                region_info["start"],
                region_info["end"],
                parser=pysam.asTuple(),
            )
            strand = region_info["strand"]
            if "tss" in region_info:
                tss = region_info["tss"]
            else:
                tss = region_info["start"]

            coordinates_raw = []
            mapping_raw = []

            fetched = fragments_tabix.fetch(
                region_info["chrom"],
                region_info["start"],
                region_info["end"],
                parser=pysam.asTuple(),
            )

            fetched = fetched
            # pbar = tqdm.tqdm(fetched, leave=False, desc="Processing fragments"

            for fragment in fetched:
                cell = fragment[3]

                # only store the fragment if the cell is actually of interest
                if cell in cell_to_cell_ix:
                    coordinates_raw.append(int(fragment[1]))
                    coordinates_raw.append(int(fragment[2]))

                    # add mapping of cell/region
                    mapping_raw.append(cell_to_cell_ix[fragment[3]])
                    mapping_raw.append(region_ix)
            coordinates_raw = (np.array(coordinates_raw).reshape(-1, 2).astype(np.int32) - tss) * strand
            mapping_raw = np.array(mapping_raw).reshape(-1, 2).astype(np.int32)

            # sort by region, coordinate (of left cut sites), and cell
            sorted_idx = np.lexsort((coordinates_raw[:, 0], mapping_raw[:, 0], mapping_raw[:, 1]))
            mapping_raw = mapping_raw[sorted_idx]
            coordinates_raw = coordinates_raw[sorted_idx]

            self.mapping.extend(mapping_raw)
            self.coordinates.extend(coordinates_raw)

            # # expand the dimensions of mapping and coordinates
            # new_shape = (mapping.shape[0] + (mapping.shape[0] - n_fragments_processed + mapping_raw.shape[0]), 2)

            # mapping = mapping.resize(exclusive_max=new_shape).result()
            # coordinates = coordinates.resize(exclusive_max=new_shape).result()

            # # write
            # mapping[n_fragments_processed : n_fragments_processed + mapping_raw.shape[0]] = mapping_raw
            # coordinates[n_fragments_processed : n_fragments_processed + coordinates_raw.shape[0]] = coordinates_raw

            del mapping_raw
            del coordinates_raw

        return self

    def filter_regions(self, regions: Regions, path: PathLike = None) -> Fragments:
        """
        Filter based on new regions

        Parameters:
            regions:
                Regions to filter.
        Returns:
            A new Fragments object
        """

        # test if new regions are a subset of the existing ones
        if not regions.coordinates.index.isin(self.regions.coordinates.index).all():
            raise ValueError("New regions should be a subset of the existing ones")

        # filter regions
        self.regions.coordinates["ix"] = np.arange(self.regions.coordinates.shape[0])
        regions.coordinates["ix"] = self.regions.coordinates["ix"].loc[regions.coordinates.index]
        fragments_oi = np.isin(self.mapping[:, 1], regions.coordinates["ix"])

        mapping = self.mapping[fragments_oi]
        coordinates = self.coordinates[fragments_oi]
        var = self.regions.coordinates.copy()
        var["original_ix"] = np.arange(var.shape[0])
        var = var.loc[regions.coordinates.index].copy()
        var["ix"] = np.arange(var.shape[0])
        mapping[:, 1] = var.set_index("original_ix").loc[mapping[:, 1], "ix"].values

        # Sort `coordinates` and `mapping` according to `mapping`
        sorted_idx = np.lexsort((coordinates[:, 0], mapping[:, 0], mapping[:, 1]))
        mapping = mapping[sorted_idx]
        coordinates = coordinates[sorted_idx]

        return Fragments.create(
            coordinates=coordinates, mapping=mapping, regions=regions, var=var, obs=self.obs, path=path
        )

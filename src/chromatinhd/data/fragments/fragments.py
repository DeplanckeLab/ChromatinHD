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

    coordinates: TensorstoreInstance = Tensorstore(dtype="<i4", chunks=[100000, 2], compression=None, shape=[0, 2])
    """Coordinates of the two cut sites."""

    mapping: TensorstoreInstance = Tensorstore(dtype="<i4", chunks=[100000, 2], compression="blosc", shape=[0, 2])
    """Mapping of a fragment to a cell (first column) and a region (second column)"""

    regionxcell_indptr: np.ndarray = Tensorstore(dtype="<i8", chunks=[100000], compression="blosc")
    """Index pointers to the regionxcell fragment positions"""

    def create_regionxcell_indptr(self, overwrite=False) -> Fragments:
        """
        Creates pointers to each individual region x cell combination from the mapping tensor.

        Returns:
            The same object with `regionxcell_indptr` populated
        """

        if self.o.regionxcell_indptr.exists(self) and not overwrite:
            return self

        regionxcell_ix = self.mapping[:, 1] * self.n_cells + self.mapping[:, 0]

        if not (np.diff(regionxcell_ix) >= 0).all():
            raise ValueError("Fragments should be ordered by regionxcell (ascending)")

        if not self.mapping[:, 0].max() < self.n_cells:
            raise ValueError("First column of mapping should be smaller than the number of cells")

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
        overwrite: bool = False,
        reuse: bool = True,
        batch_size: int = 1e6,
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
            batch_size:
                Number of fragments to process before saving. Lower this number if you run out of memory.
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

        mapping_processed = []
        coordinates_processed = []

        for region_ix, (region_id, region_info) in pbar:
            pbar.set_description(f"{region_id}")

            strand = region_info["strand"]
            if "tss" in region_info:
                tss = region_info["tss"]
            else:
                tss = region_info["start"]

            coordinates_raw, mapping_raw = _fetch_fragments_region(
                fragments_tabix=fragments_tabix,
                chrom=region_info["chrom"],
                start=region_info["start"],
                end=region_info["end"],
                tss=tss,
                strand=strand,
                cell_to_cell_ix=cell_to_cell_ix,
                region_ix=region_ix,
            )

            mapping_processed.append(mapping_raw)
            coordinates_processed.append(coordinates_raw)

            if sum(mapping_raw.shape[0] for mapping_raw in mapping_processed) >= batch_size:
                self.mapping.extend(np.concatenate(mapping_processed).astype(np.int32))
                self.coordinates.extend(np.concatenate(coordinates_processed).astype(np.int32))
                mapping_processed = []
                coordinates_processed = []

            del mapping_raw
            del coordinates_raw

        if len(mapping_processed) > 0:
            self.mapping.extend(np.concatenate(mapping_processed).astype(np.int32))
            self.coordinates.extend(np.concatenate(coordinates_processed).astype(np.int32))

        return self

    @class_or_instancemethod
    def from_multiple_fragments_tsv(
        cls,
        fragments_files: [PathLike],
        regions: Regions,
        obs: pd.DataFrame,
        cell_column: str = "cell_original",
        batch_column: str = "batch",
        path: PathLike = None,
        overwrite: bool = False,
        reuse: bool = True,
        batch_size: int = 1e6,
    ) -> Fragments:
        """
        Create a Fragments object from multiple fragments tsv file

        Parameters:
            fragments_files:
                Location of the fragments tab-separate file created by e.g. CellRanger or sinto
            obs:
                DataFrame containing information about cells.
                The index should be the cell names as present in the fragments file.
                Alternatively, the column containing cell ids can be specified using the `cell_column` argument.
            regions:
                Regions from which the fragments will be extracted.
            cell_column:
                Column name in the `obs` DataFrame containing the cell names.
            batch_column:
                Column name in the `obs` DataFrame containing the batch indices.
                If None, will default to batch
            path:
                Folder in which the fragments data will be stored.
            overwrite:
                Whether to overwrite the data if it already exists.
            reuse:
                Whether to reuse existing data if it exists.
            batch_size:
                Number of fragments to process before saving. Lower this number if you run out of memory.
        Returns:
            A new Fragments object
        """

        if not isinstance(fragments_files, (list, tuple)):
            fragments_files = [fragments_files]

        fragments_files_ = []
        for fragments_file in fragments_files:
            if isinstance(fragments_file, str):
                fragments_file = pathlib.Path(fragments_file)
            if not fragments_file.exists():
                raise FileNotFoundError(f"File {fragments_file} does not exist")
            fragments_files_.append(fragments_file)
        fragments_files = fragments_files_

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

        if batch_column is None:
            raise ValueError("batch_column should be specified")
        if batch_column not in obs.columns:
            raise ValueError(f"Column {batch_column} not in obs")
        if obs[batch_column].dtype != "int":
            raise ValueError(f"Column {batch_column} should be an integer column")
        if not obs[batch_column].max() == len(fragments_files) - 1:
            raise ValueError(f"Column {batch_column} should contain values between 0 and {len(fragments_files) - 1}")
        cell_to_cell_ix_batches = [
            obs.loc[obs[batch_column] == batch].set_index(cell_column)["ix"] for batch in obs[batch_column].unique()
        ]

        self = cls.create(path=path, obs=obs, var=var, regions=regions, reset=overwrite)

        # read the fragments file
        try:
            import pysam
        except ImportError as e:
            raise ImportError(
                "pysam is required to read fragments files. Install using `pip install pysam` or `conda install -c bioconda pysam`"
            ) from e
        fragments_tabix_batches = [pysam.TabixFile(str(fragments_file)) for fragments_file in fragments_files]

        # process regions
        pbar = tqdm.tqdm(
            enumerate(regions.coordinates.iterrows()),
            total=regions.coordinates.shape[0],
            leave=False,
            desc="Processing fragments",
        )

        self.mapping.open_creator()
        self.coordinates.open_creator()

        mapping_processed = []
        coordinates_processed = []

        for region_ix, (region_id, region_info) in pbar:
            pbar.set_description(f"{region_id}")

            strand = region_info["strand"]
            if "tss" in region_info:
                tss = region_info["tss"]
            else:
                tss = region_info["start"]

            mapping_raw = []
            coordinates_raw = []

            for fragments_tabix, cell_to_cell_ix in zip(fragments_tabix_batches, cell_to_cell_ix_batches):
                coordinates_raw_batch, mapping_raw_batch = _fetch_fragments_region(
                    fragments_tabix=fragments_tabix,
                    chrom=region_info["chrom"],
                    start=region_info["start"],
                    end=region_info["end"],
                    tss=tss,
                    strand=strand,
                    cell_to_cell_ix=cell_to_cell_ix,
                    region_ix=region_ix,
                )
                print(len(coordinates_raw_batch))
                mapping_raw.append(mapping_raw_batch)
                coordinates_raw.append(coordinates_raw_batch)

            mapping_raw = np.concatenate(mapping_raw)
            coordinates_raw = np.concatenate(coordinates_raw)

            # sort by region, coordinate (of left cut sites), and cell
            sorted_idx = np.lexsort((coordinates_raw[:, 0], mapping_raw[:, 0], mapping_raw[:, 1]))
            mapping_raw = mapping_raw[sorted_idx]
            coordinates_raw = coordinates_raw[sorted_idx]

            mapping_processed.append(mapping_raw)
            coordinates_processed.append(coordinates_raw)

            if sum(mapping_raw.shape[0] for mapping_raw in mapping_processed) >= batch_size:
                self.mapping.extend(np.concatenate(mapping_processed).astype(np.int32))
                self.coordinates.extend(np.concatenate(coordinates_processed).astype(np.int32))
                mapping_processed = []
                coordinates_processed = []

            del mapping_raw
            del coordinates_raw

        if len(mapping_processed) > 0:
            self.mapping.extend(np.concatenate(mapping_processed).astype(np.int32))
            self.coordinates.extend(np.concatenate(coordinates_processed).astype(np.int32))

        return self

    @class_or_instancemethod
    def from_alignments(
        cls,
        obs: pd.DataFrame,
        regions: Regions,
        file_column: str = "path",
        alignment_column: str = None,
        remove_duplicates: bool = None,
        path: PathLike = None,
        overwrite: bool = False,
        reuse: bool = True,
        batch_size: int = 10e6,
        paired: bool = True,
    ) -> Fragments:
        """
        Create a Fragments object from multiple alignment (bam/sam) files, each pertaining to a single cell or sample

        Parameters:
            obs:
                DataFrame containing information about cells/samples.
                The index will be used as the name of the cell for future reference.
                The DataFrame should contain a column with the path to the alignment file (specified in the file_column) or a column with the alignment object itself (specified in the alignment_column).
                Any additional data about cells/samples can be stored in this dataframe as well.
            regions:
                Regions from which the fragments will be extracted.
            file_column:
                Column name in the `obs` DataFrame containing the path to the alignment file.
            alignment_column:
                Column name in the `obs` DataFrame containing the alignment object.
                If None, the alignment object will be loaded using the `file_column`.
            remove_duplicates:
                Whether to remove duplicate fragments within a sample or cell. This is commonly done for single-cell data, but not necessarily for bulk data. If the data is paired, duplicates will be removed by default. If the data is single-end, duplicates will not be removed by default.
            path:
                Folder in which the fragments data will be stored.
            overwrite:
                Whether to overwrite the data if it already exists.
            reuse:
                Whether to reuse existing data if it exists.
            batch_size:
                Number of fragments to process before saving. Lower this number if you run out of memory. Increase the number if you want to speed up the process, particularly if disk I/O is slow.
            paired:
                Whether the reads are paired-end or single-end. If paired, the coordinates of the two cut sites will be stored. If single-end, only the coordinate of only one cut site will be stored. Note that this also affects the default value of `remove_duplicates`.
        """
        # regions information
        var = pd.DataFrame(index=regions.coordinates.index)
        var["ix"] = np.arange(var.shape[0])

        # check path and overwrite
        if path is not None:
            if isinstance(path, str):
                path = pathlib.Path(path)
            if not overwrite and path.exists() and not reuse:
                raise FileExistsError(
                    f"Folder {path} already exists, use `overwrite=True` to overwrite, or `reuse=True` to reuse existing data"
                )
        if overwrite:
            reuse = False

        self = cls.create(path=path, obs=obs, var=var, regions=regions, reset=overwrite)

        if reuse:
            return self

        # load alignment files
        try:
            import pysam
        except ImportError as e:
            raise ImportError(
                "pysam is required to read alignment files. Install using `pip install pysam` or `conda install -c bioconda pysam`"
            ) from e

        if alignment_column is None:
            if file_column not in obs.columns:
                raise ValueError(f"Column {file_column} not in obs")
            alignments = {}
            for cell, cell_info in obs.iterrows():
                alignments[cell] = pysam.Samfile(cell_info[file_column], "rb")
        else:
            if alignment_column not in obs.columns:
                raise ValueError(f"Column {alignment_column} not in obs")
            alignments = obs[alignment_column].to_dict()

        # process regions
        pbar = tqdm.tqdm(
            enumerate(regions.coordinates.iterrows()),
            total=regions.coordinates.shape[0],
            leave=False,
            desc="Processing fragments",
        )

        self.mapping.open_creator()
        self.coordinates.open_creator()

        mapping_processed = []
        coordinates_processed = []

        for region_ix, (region_id, region_info) in pbar:
            pbar.set_description(f"{region_id}")

            chrom = region_info["chrom"]
            start = region_info["start"]
            end = region_info["end"]
            strand = region_info["strand"]
            if "tss" in region_info:
                tss = region_info["tss"]
            else:
                tss = region_info["start"]

            # process cell/sample
            for cell_ix, (cell_id, alignment) in enumerate(alignments.items()):
                if paired:
                    coordinates_raw = _process_paired(
                        alignment=alignment,
                        chrom=chrom,
                        start=start,
                        end=end,
                        remove_duplicates=True if remove_duplicates is None else remove_duplicates,
                    )
                    coordinates_raw = (np.array(coordinates_raw).reshape(-1, 2).astype(np.int32) - tss) * strand
                else:
                    coordinates_raw = _process_single(
                        alignment=alignment,
                        chrom=chrom,
                        start=start,
                        end=end,
                        remove_duplicates=False if remove_duplicates is None else remove_duplicates,
                    )
                    coordinates_raw = (np.array(coordinates_raw).reshape(-1, 1).astype(np.int32) - tss) * strand

                mapping_raw = np.stack(
                    [
                        np.repeat(cell_ix, len(coordinates_raw)),
                        np.repeat(region_ix, len(coordinates_raw)),
                    ],
                    axis=1,
                )

                # sort by region, coordinate (of left cut sites), and cell
                sorted_idx = np.lexsort((coordinates_raw[:, 0], mapping_raw[:, 0], mapping_raw[:, 1]))
                mapping_raw = mapping_raw[sorted_idx]
                coordinates_raw = coordinates_raw[sorted_idx]

                if len(mapping_raw) > 0:
                    mapping_processed.append(mapping_raw)
                    coordinates_processed.append(coordinates_raw)

                if sum(mapping_raw.shape[0] for mapping_raw in mapping_processed) >= batch_size:
                    self.mapping.extend(np.concatenate(mapping_processed).astype(np.int32))
                    self.coordinates.extend(np.concatenate(coordinates_processed).astype(np.int32))
                    mapping_processed = []
                    coordinates_processed = []

                del mapping_raw
                del coordinates_raw

        # add final fragments
        if len(mapping_processed) > 0:
            self.mapping.extend(np.concatenate(mapping_processed).astype(np.int32))
            self.coordinates.extend(np.concatenate(coordinates_processed).astype(np.int32))

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

        # filter region info
        self.regions.coordinates["ix"] = np.arange(self.regions.coordinates.shape[0])
        regions.coordinates["ix"] = self.regions.coordinates["ix"].loc[regions.coordinates.index]

        var = self.regions.coordinates.copy()
        var["original_ix"] = np.arange(var.shape[0])
        var = var.loc[regions.coordinates.index].copy()
        var["ix"] = np.arange(var.shape[0])

        # filter coordinates/mapping
        fragments_oi = np.isin(self.mapping[:, 1], regions.coordinates["ix"])

        mapping = self.mapping[fragments_oi].copy()
        mapping[:, 1] = var.set_index("original_ix").loc[mapping[:, 1], "ix"].values
        coordinates = self.coordinates[fragments_oi].copy()

        # Sort `coordinates` and `mapping` according to `mapping`
        sorted_idx = np.lexsort((coordinates[:, 0], mapping[:, 0], mapping[:, 1]))
        mapping = mapping[sorted_idx]
        coordinates = coordinates[sorted_idx]

        fragments = Fragments.create(
            coordinates=coordinates, mapping=mapping, regions=regions, var=var, obs=self.obs, path=path, reset=overwrite
        )

        return fragments

    def filter_cells(self, cells, path: PathLike = None) -> Fragments:
        """
        Filter based on new cells

        Parameters:
            cells:
                Cells to filter.
        Returns:
            A new Fragments object
        """

        # check if new cells are a subset of the existing ones
        if not pd.Series(cells).isin(self.obs.index).all():
            raise ValueError("New cells should be a subset of the existing ones")

        # filter region info
        self.obs["ix"] = np.arange(self.obs.shape[0])
        obs = self.obs.loc[cells].copy()
        obs["original_ix"] = self.obs["ix"].loc[obs.index]
        obs["ix"] = np.arange(obs.shape[0])

        # filter coordinates/mapping
        fragments_oi = np.isin(self.mapping[:, 0], obs["original_ix"])

        mapping = self.mapping[fragments_oi].copy()
        mapping[:, 0] = obs.set_index("original_ix").loc[mapping[:, 0], "ix"].values
        coordinates = self.coordinates[fragments_oi]

        # Sort `coordinates` and `mapping` according to `mapping`
        sorted_idx = np.lexsort((coordinates[:, 0], mapping[:, 0], mapping[:, 1]))
        mapping = mapping[sorted_idx]
        coordinates = coordinates[sorted_idx]

        return Fragments.create(
            coordinates=coordinates, mapping=mapping, regions=self.regions, var=self.var, obs=obs, path=path
        )

    @property
    def counts(self):
        """
        Counts of fragments per cell x region
        """
        cellxregion_ix = self.mapping[:, 0] * self.n_regions + self.mapping[:, 1]
        counts = np.bincount(cellxregion_ix, minlength=self.n_cells * self.n_regions).reshape(
            self.n_cells, self.n_regions
        )
        return counts

    @property
    def nonzero(self):
        return self.counts > 0

    _single_region_cache = None

    def get_cache(self, region_oi):
        if self._single_region_cache is None:
            self._single_region_cache = {}
        if region_oi not in self._single_region_cache:
            region = self.var.index.get_loc(region_oi)
            regionxcell_indptr = self.regionxcell_indptr.open_reader(
                {"context": {"data_copy_concurrency": {"limit": 1}}}
            )[region * self.n_cells : (region + 1) * self.n_cells + 1]
            coordinates = self.coordinates.open_reader(
                {
                    "context": {
                        "data_copy_concurrency": {"limit": 1},
                    }
                }
            )[regionxcell_indptr[0] : regionxcell_indptr[-1]]
            regionxcell_indptr = regionxcell_indptr - regionxcell_indptr[0]

            self._single_region_cache[region_oi] = {
                "regionxcell_indptr": regionxcell_indptr,
                "coordinates": coordinates,
            }
        return self._single_region_cache[region_oi]

    _libsize = None

    @property
    def libsize(self):
        if self._libsize is None:
            self._libsize = np.bincount(self.mapping[:, 0], minlength=self.n_cells)
        return self._libsize


def _process_paired(chrom, start, end, alignment, remove_duplicates=True):
    fragments_dict = {}
    for segment in alignment.fetch(chrom, start, end):
        fragments_dict = update_fragment_dict(fragments_dict, segment, 30, 1000, 0)

    # contains the cache of the fragments
    fragment_set = set()
    coordinates_raw = []
    for fragment in fragments_dict.values():
        if fragment[3]:
            if remove_duplicates:
                if fragment[1] + fragment[2] in fragment_set:
                    continue
                fragment_set.add(fragment[1] + fragment[2])
            coordinates_raw.append(int(fragment[1]))
            coordinates_raw.append(int(fragment[2]))
    return coordinates_raw


def _process_single(chrom, start, end, alignment, remove_duplicates=False, min_mapq=30, shifts=[4, -5]):
    coordinates_raw = []
    for segment in alignment.fetch(chrom, start, end):
        mapq = segment.mapping_quality
        if mapq < min_mapq:
            continue
        is_reverse = segment.is_reverse
        # correct for 9 bp Tn5 shift
        if is_reverse:
            position = segment.reference_end + shifts[1]
        else:
            position = segment.reference_start + shifts[0]
        coordinates_raw.append(int(position))
    return coordinates_raw


# modified and adapted from sinto
def update_fragment_dict(fragments, segment, min_mapq, max_dist, min_dist, shifts=[4, -5]):
    """Update dictionary of ATAC fragments
    Takes a new aligned segment and adds information to the dictionary,
    returns a modified version of the dictionary

    Positions are 0-based
    Reads aligned to the + strand are shifted +4 bp (configurable by shifts)
    Reads aligned to the - strand are shifted -5 bp (configurable by shifts)

    Parameters
    ----------
    fragments : dict
        A dictionary containing ATAC fragment information
    segment : pysam.AlignedSegment
        An aligned segment
    min_mapq : int
        Minimum MAPQ to retain fragment
    max_dist : int
        Maximum allowed distance between fragment start and end sites
    min_dist : int
        Minimum allowed distance between fragment start and end sites
    shifts : list
        List of adjustments made to fragment position. First element defines + strand
        shift, second element defines - strand shift.
    """

    mapq = segment.mapping_quality
    if mapq < min_mapq:
        return fragments
    chromosome = segment.reference_name
    qname = segment.query_name
    rstart = segment.reference_start
    rend = segment.reference_end
    is_reverse = segment.is_reverse
    if (rend is None) or (rstart is None):
        return fragments
    # correct for 9 bp Tn5 shift
    if is_reverse:
        rend = rend + shifts[1]
    else:
        rstart = rstart + shifts[0]
    fragments = add_to_fragments(fragments, qname, chromosome, rstart, rend, is_reverse, max_dist, min_dist)
    return fragments


def add_to_fragments(fragments, qname, chromosome, rstart, rend, is_reverse, max_dist, min_dist):
    """Add new fragment information to dictionary

    Parameters
    ----------

    fragments : dict
        A dictionary containing all the fragment information
    qname : str
        Read name
    chromosome : str
        Chromosome name
    rstart : int
        Alignment start position
    rend : int
        Alignment end position
    is_reverse : bool
        Read is aligned to reverse strand
    max_dist : int
        Maximum allowed fragment size
    min_dist : int
        Minimum allowed fragment size
    """
    if qname in fragments.keys():
        if is_reverse:
            current_coord = fragments[qname][1]
            if current_coord is None:
                # read aligned to the wrong strand, don't include
                del fragments[qname]
            elif ((rend - current_coord) > max_dist) or ((rend - current_coord) < min_dist):
                # too far away, don't include
                del fragments[qname]
            else:
                fragments[qname][2] = rend
                fragments[qname][3] = True
        else:
            current_coord = fragments[qname][2]
            if current_coord is None:
                del fragments[qname]
            elif ((current_coord - rstart) > max_dist) or ((current_coord - rstart) < min_dist):
                del fragments[qname]
    else:
        # new read pair, add to dictionary
        fragments[qname] = [
            chromosome,  # chromosome    0
            None,  # start         1
            None,  # end           2
            False,  # complete      3
        ]
        if is_reverse:
            fragments[qname][2] = rend
        else:
            fragments[qname][1] = rstart
    return fragments


def _fetch_fragments_region(fragments_tabix, chrom, start, end, tss, strand, cell_to_cell_ix, region_ix):
    import pysam

    fetched = fragments_tabix.fetch(
        chrom,
        max(start, 0),
        end,
        parser=pysam.asTuple(),
    )

    coordinates_raw = []
    mapping_raw = []

    fetched = fetched

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

    return coordinates_raw, mapping_raw

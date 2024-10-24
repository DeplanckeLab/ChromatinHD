import pathlib
from re import A
from typing import Union, Dict

import numpy as np
import pandas as pd
import torch
import tqdm.auto as tqdm

from chromatinhd import get_default_device
from chromatinhd.data.regions import Regions
from chromatinhd.flow import (
    CompressedNumpyFloat64,
    CompressedNumpyInt64,
    Flow,
    Linked,
    Stored,
    StoredDataFrame,
    PathLike,
)
from chromatinhd.flow.tensorstore import Tensorstore, TensorstoreInstance
from chromatinhd.utils.torch import ind2ptr
from chromatinhd.utils.numpy import ind2ptr as ind2ptr_numpy
from chromatinhd.utils import indices_to_indptr, indices_to_indptr_chunked

# try to load the shared library
# typically, this will be installed as a python extension
try:
    from . import scan_helpers  # pylint: disable=C0413,E0611
# however, during developement, we want to load the cython source directly
except ImportError:
    import pyximport

    pyximport.install(
        reload_support=True,
        language_level=3,
        setup_args=dict(include_dirs=[np.get_include()]),
        build_in_temp=False,
    )
    from . import scan_helpers  # pylint: disable=C0413,E0611


class Motifscan(Flow):
    """
    A sprase representation of locations of different motifs in regions of the genome
    """

    regions = Linked()
    "The regions"

    indptr: TensorstoreInstance = Tensorstore(dtype="<i8", chunks=(10000,))
    "The index pointers for each position in the regions"

    region_indptr: TensorstoreInstance = Tensorstore(dtype="<i8", chunks=(1000,), compression=None)
    "The index pointers for region"

    coordinates: TensorstoreInstance = Tensorstore(dtype="<i4", chunks=(10000,))
    "Coordinate associated to each site"

    region_indices: TensorstoreInstance = Tensorstore(dtype="<i4", chunks=(10000,))
    "Region index associated to each site"

    indices: TensorstoreInstance = Tensorstore(dtype="<i4", chunks=(10000,))
    "Motif index associated to each site"

    # positions: TensorstoreInstance = Tensorstore(dtype="<i8", chunks=(10000,))
    # "Cumulative coordinate of each site"

    scores: TensorstoreInstance = Tensorstore(dtype="<f4", chunks=(10000,))
    "Scores associated with each detected site"

    strands: TensorstoreInstance = Tensorstore(dtype="<f4", chunks=(10000,))
    "Strand associated with each detected site"

    shape = Stored()

    motifs = StoredDataFrame()
    "Dataframe storing auxilliary information for each motif"

    @classmethod
    def from_pwms(
        cls,
        pwms: dict,
        regions: Regions,
        fasta_file: Union[str, pathlib.Path] = None,
        region_onehots: Dict[np.ndarray, torch.Tensor] = None,
        cutoffs: Union[int, float, pd.Series] = None,
        cutoff_col: str = None,
        min_cutoff=3.0,
        motifs: pd.DataFrame = None,
        device: str = None,
        batch_size: int = 50000000,
        path: Union[str, pathlib.Path] = None,
        overwrite: bool = True,
        reuse: bool = False,
    ):
        """
        Create a motifscan object from a set of pwms and a set of regions

        Parameters:
            pwms:
                A dictionary of pwms, where the keys are the motif ids and the values are the pwms
            regions:
                A regions object
            fasta_file:
                The location of the fasta file containing the genome
            region_onehots:
                A dictionary containing the onehot encoding of each region. If not given, the onehot encoding will be extracted from the fasta file
            motifs:
                A dataframe containing auxilliary information for each motif
            cutoffs:
                A dictionary containing the cutoffs for each motif.
            cutoff_col:
                The column in the motifs dataframe containing the cutoffs
            device:
                The device to use for the scanning
            batch_size:
                The batch size to use for scanning. Decrease this if the GPU runs out of memory
            path:
                The folder where the motifscan data will be stored.
            overwrite:
                Whether to overwrite existing motifscan data
            reuse:
                Whether to reuse existing motifscan data
        """

        if device is None:
            device = get_default_device()

        self = cls(path)

        if ((reuse) or (not overwrite)) and self.o.coordinates.exists(self):
            if not reuse:
                import warnings

                warnings.warn("Motifscan already exists. Use overwrite=True to overwrite, reuse=True to ignore this warning.")
            return self

        if overwrite:
            self.reset()

        self.motifs = motifs
        self.regions = regions

        # check or create cutoffs
        if cutoffs is None:
            if cutoff_col is None:
                raise ValueError("Either motifs+cutoff_col or cutoffs need to be specified.")
            if motifs is None:
                raise ValueError("Either motifs+cutoff_col or cutoffs need to be specified. motifs is not given")

            cutoffs = motifs[cutoff_col].to_dict()
        else:
            if isinstance(cutoffs, (float, int)):
                cutoffs = {motif: cutoffs for motif in pwms.keys()}
            elif isinstance(cutoffs, pd.Series):
                cutoffs = cutoffs.to_dict()
            else:
                raise ValueError("cutoffs should be a float, int, dict or pd.Series")
            assert set(cutoffs.keys()) == set(pwms.keys())

        # check or create motifs
        if motifs is None:
            motifs = pd.DataFrame(
                {
                    "motif": list(pwms.keys()),
                }
            ).set_index("motif")

        # divide regions into batches according to batch size
        region_coordinates = regions.coordinates

        region_coordinates = divide_regions_in_batches(region_coordinates, batch_size=batch_size)

        # load in fasta file
        if fasta_file is not None:
            import pysam

            fasta = pysam.FastaFile(fasta_file)
        else:
            fasta = None
            if region_onehots is None:
                raise ValueError("Either fasta_file or region_onehots need to be specified")

        self.indices.open_creator()
        self.scores.open_creator()
        self.strands.open_creator()
        self.coordinates.open_creator()
        self.region_indices.open_creator()

        # do the actual counting by looping over the batches, extract the sequences and scanning
        progress = tqdm.tqdm(region_coordinates.groupby("batch"))
        cur_region_index = 0
        for batch, region_coordinates_batch in progress:
            # extract onehot
            if fasta is None:
                sequences = [fasta.fetch(chrom, start, end + 1) for chrom, start, end in region_coordinates_batch[["chrom", "start", "end"]].values]
                if not all(len(sequence) == len(sequences[0]) for sequence in sequences):
                    raise ValueError("All regions/sequences should have the same length")
                onehot = create_onehots(sequences).permute(0, 2, 1)
            else:
                if region_onehots is None:
                    if fasta_file is None:
                        raise ValueError("fasta_file must be provided if fasta and region_onehots is not provided")
                    progress.set_description("Extracting sequences")
                    region_onehots = create_region_onehots(regions, fasta_file)
                onehot = torch.stack([region_onehots[region] for region in region_coordinates_batch.index]).permute(0, 2, 1)
            onehot = onehot.to(device)

            progress.set_description(f"Scanning batch {batch} {region_coordinates_batch.index[0]}-{region_coordinates_batch.index[-1]}")

            assert onehot.shape[1] == 4
            assert onehot.shape[2] == region_coordinates_batch["len"].iloc[0], (
                onehot.shape[2],
                region_coordinates_batch["len"].iloc[0],
            )

            scores_raw = []
            indices_raw = []
            coordinates_raw = []
            strands_raw = []
            region_indices_raw = []
            for motif_ix, motif in tqdm.tqdm(enumerate(motifs.index)):
                cutoff = cutoffs[motif]

                if cutoff < min_cutoff:
                    cutoff = min_cutoff

                # get pwm
                pwm = pwms[motif]
                if not torch.is_tensor(pwm):
                    pwm = torch.from_numpy(pwm)
                pwm2 = pwm.to(dtype=torch.float32, device=onehot.device).transpose(1, 0)

                (
                    scores,
                    positions,
                    strands,
                ) = scan(onehot, pwm2, cutoff=cutoff)

                coordinates = positions.astype(np.int32) % onehot.shape[-1]

                region_indices = positions // onehot.shape[-1] + cur_region_index

                if "tss" in regions.coordinates:
                    coordinates = coordinates + (self.regions.coordinates["start"] - self.regions.coordinates["tss"]).values[region_indices]

                coordinates_raw.append(coordinates)
                indices_raw.append(np.full_like(coordinates, motif_ix, dtype=np.int32))
                strands_raw.append(strands)
                scores_raw.append(scores)
                region_indices_raw.append(region_indices)

            # concatenate raw values (sorted by motif)
            coordinates = np.concatenate(coordinates_raw)
            indices = np.concatenate(indices_raw)
            strands = np.concatenate(strands_raw)
            scores = np.concatenate(scores_raw)
            region_indices = np.concatenate(region_indices_raw)

            # sort according to position
            sorted_idx = np.lexsort([coordinates, region_indices])
            indices = indices[sorted_idx]
            scores = scores[sorted_idx]
            strands = strands[sorted_idx]
            coordinates = coordinates[sorted_idx]
            region_indices = region_indices[sorted_idx]

            # store batch
            self.indices.extend(indices)
            self.scores.extend(scores)
            self.strands.extend(strands)
            self.coordinates.extend(coordinates)
            self.region_indices.extend(region_indices)

            # update current region index
            cur_region_index += len(region_coordinates_batch)

        return self

    def create_region_indptr(self, overwrite=False):
        """
        Populate the region_indptr
        """

        if self.o.region_indptr.exists(self) and not overwrite:
            return

        region_indices_reader = self.region_indices.open_reader()
        self.region_indptr = indices_to_indptr_chunked(region_indices_reader, self.regions.n_regions, dtype=np.int64)

    def create_indptr(self, overwrite=False):
        """
        Populate the indptr
        """

        if self.o.indptr.exists(self) and not overwrite:
            return

        if self.regions.width is not None:
            indptr = self.indptr.open_creator(shape=((self.regions.n_regions * self.regions.width) + 1,), dtype=np.int64)
            region_width = self.regions.width
            for region_ix, (region_start, region_end) in tqdm.tqdm(enumerate(zip(self.region_indptr[:-1], self.region_indptr[1:]))):
                indptr[region_ix * region_width : (region_ix + 1) * region_width] = indices_to_indptr(self.coordinates[region_start:region_end], self.regions.width)[:-1] + region_start
            indptr[-1] = region_end
        else:
            indptr = self.indptr.open_creator(shape=(self.regions.cumulative_region_lengths[-1] + 1,), dtype=np.int64)
            for region_ix, (region_start, region_end) in tqdm.tqdm(enumerate(zip(self.region_indptr[:-1], self.region_indptr[1:]))):
                region_start_position = self.regions.cumulative_region_lengths[region_ix]
                region_end_position = self.regions.cumulative_region_lengths[region_ix + 1]
                indptr[region_start_position:region_end_position] = (
                    indices_to_indptr_chunked(
                        self.coordinates[region_start:region_end],
                        region_end_position - region_start_position,
                    )[:-1]
                    + region_start
                )
            indptr[-1] = region_end

    @classmethod
    def from_positions(cls, positions, indices, scores, strands, regions, motifs, path=None):
        """
        Create a motifscan object from positions, indices, scores, strands, regions and motifs
        """
        self = cls(path=path)

        # sort the positions
        sorted_idx = np.argsort(positions)

        self.positions = positions[sorted_idx]
        self.indices = indices[sorted_idx]
        self.scores = scores[sorted_idx]
        self.strands = strands[sorted_idx]
        self.regions = regions
        self.motifs = motifs

        return self

    def filter(self, motif_ids, path=None):
        """
        Select a subset of motifs
        """

        self.motifs["ix"] = np.arange(len(self.motifs))
        motif_ixs = self.motifs.loc[motif_ids, "ix"]

        selected_sites = np.isin(self.indices, motif_ixs)

        new = self.__class__(path=path).create(
            regions=self.regions,
            positions=self.positions[selected_sites],
            indices=self.indices[selected_sites],
            scores=self.scores[selected_sites],
            strands=self.strands[selected_sites],
            motifs=self.motifs.loc[motif_ids],
        )

        new.create_indptr()
        return new

    @property
    def n_motifs(self):
        return len(self.motifs)

    @property
    def scanned(self):
        return self.o.indices.exists(self)

    def get_slice(
        self,
        region_ix=None,
        region_id=None,
        start=None,
        end=None,
        return_indptr=False,
        return_scores=True,
        return_strands=True,
        motif_ixs=None,
    ):
        """
        Get a slice of the motifscan

        Parameters:
            region:
                Region id
            start:
                Start of the slice, in region coordinates
            end:
                End of the slice, in region coordinates

        Returns:
            Motifs positions, indices, scores and strands of the slice
        """
        if region_id is not None:
            region = self.regions.coordinates.loc[region_id]
        elif region_ix is not None:
            region = self.regions.coordinates.iloc[region_ix]
        else:
            raise ValueError("Either region or region_ix should be provided")
        if region_ix is None:
            region_ix = self.regions.coordinates.index.get_indexer([region_id])[0]

        if self.regions.width is not None:
            # get slice for fixed width regions
            width = self.regions.width

            if start is None:
                start = self.regions.window[0]
            if end is None:
                end = self.regions.window[1]

            if self.o.indptr.exists(self):
                start = region_ix * width + start
                end = region_ix * width + end
                indptr = self.indptr[start : end + 1]
                indptr_start, indptr_end = indptr[0], indptr[-1]
                indptr = indptr - indptr[0]
            else:
                region_start = self.region_indptr[region_ix]
                region_end = self.region_indptr[region_ix + 1]
                coordinates = self.coordinates[region_start:region_end]
                indptr_start = coordinates.searchsorted(start - 1) + region_start
                indptr_end = coordinates.searchsorted(end) + region_start

            coordinates = self.coordinates[indptr_start:indptr_end]
            indices = self.indices[indptr_start:indptr_end]

            out = [coordinates, indices]
            if return_scores:
                out.append(self.scores[indptr_start:indptr_end])
            if return_strands:
                out.append(self.strands[indptr_start:indptr_end])

            if motif_ixs is not None:
                selection = np.isin(indices, motif_ixs)
                out = [x[selection] for x in out]

            if return_indptr:
                out.append(indptr)

            return out
        else:
            # get slice for variable width regions
            assert start is not None
            assert end is not None

            if self.o.indptr.exists(self):
                start = self.regions.cumulative_region_lengths[region_ix] + start
                end = self.regions.cumulative_region_lengths[region_ix] + end
                indptr = self.indptr[start : end + 1]
                indptr_start, indptr_end = indptr[0], indptr[-1]
                indptr = indptr - indptr[0]
            else:
                region_start = self.region_indptr[region_ix]
                region_end = self.region_indptr[region_ix + 1]
                coordinates = self.coordinates[region_start:region_end]
                indptr_start = coordinates.searchsorted(start - 1) + region_start
                indptr_end = coordinates.searchsorted(end) + region_start

            coordinates = self.coordinates[indptr_start:indptr_end]
            indices = self.indices[indptr_start:indptr_end]

            out = [coordinates, indices]
            if return_scores:
                out.append(self.scores[indptr_start:indptr_end])
            if return_strands:
                out.append(self.strands[indptr_start:indptr_end])

            if motif_ixs is not None:
                selection = np.isin(indices, motif_ixs)
                out = [x[selection] for x in out]

            if return_indptr:
                out.append(indptr)

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

        # if self.regions.window is None:
        #     raise NotImplementedError("count_slices is only implemented for regions with a window")

        if "region_ix" not in slices:
            slices["region_ix"] = self.regions.coordinates.index.get_indexer(slices["region"])

        progress = enumerate(zip(slices["start"], slices["end"], slices["region_ix"]))
        progress = tqdm.tqdm(
            progress,
            total=len(slices),
            leave=False,
            desc="Counting slices",
        )

        motif_counts = np.zeros((len(slices), self.n_motifs), dtype=int)
        for i, (relative_start, relative_end, region_ix) in progress:
            start = relative_start
            end = relative_end
            positions, indices = self.get_slice(
                region_ix=region_ix,
                start=start,
                end=end,
                return_scores=False,
                return_strands=False,
                return_indptr=False,
            )
            motif_counts[i] = np.bincount(indices, minlength=self.n_motifs)
        motif_counts = pd.DataFrame(motif_counts, index=slices.index, columns=self.motifs.index)
        return motif_counts

    def select_motif(self, x=None, symbol=None):
        if symbol is not None:
            return self.motifs.loc[self.motifs["symbol"] == symbol].index[0]
        # return motifscan.motifs.loc[motifscan.motifs.index.str.contains(str)].sort_values("quality").index[0]
        return self.motifs.loc[self.motifs.index.str.contains(x)].index[0]

    def select_motifs(self, x=None, symbol=None):
        if symbol is not None:
            return self.motifs.loc[self.motifs["symbol"] == symbol].index.tolist()
        return self.motifs.loc[self.motifs.index.str.contains(x)].index.tolist()


def divide_regions_in_batches(region_coordinates, batch_size=10):
    region_coordinates["len"] = region_coordinates["end"] - region_coordinates["start"]
    region_coordinates["cumlen"] = (region_coordinates["end"] - region_coordinates["start"]).cumsum()
    region_coordinates["same_length"] = region_coordinates["len"] == region_coordinates["len"].iloc[0]
    region_coordinates["cumlen"] = region_coordinates["cumlen"] + (batch_size * ~region_coordinates["same_length"])
    region_coordinates["batch"] = (region_coordinates["cumlen"] // batch_size).astype(int)
    # region_coordinates["batch"] = np.argsort(region_coordinates["batch"])

    region_coordinates["ix"] = np.arange(region_coordinates.shape[0])

    return region_coordinates


def read_pwms(pwms_file):
    if str(pwms_file).endswith(".txt"):
        pwms = {}
        motif = None
        motif_id = None
        with open(pwms_file) as f:
            for line in f:
                if line.startswith(">"):
                    if motif is not None:
                        pwms[motif_id] = motif
                    motif_id = line[1:].strip("\n")
                    motif = []
                else:
                    motif.append([float(x) for x in line.split("\t")])

        pwms = {motif_id: torch.tensor(pwm) for motif_id, pwm in pwms.items()}
    elif str(pwms_file).endswith(".tar.gz"):
        import tarfile

        pwms = {}
        with tarfile.open(pwms_file, "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    with tar.extractfile(member) as f:
                        firstline = f.readline().decode()
                        motif_id = firstline[1:].strip("\n")
                        pwm = np.loadtxt(f, skiprows=1)
                        pwms[motif_id] = torch.tensor(pwm)

    return pwms


def scan(onehot, pwm, cutoff=0.0):
    """
    Use 1D convolutions to scan a motif over a (one-hot) sequence
    """
    assert onehot.shape[1] == 4
    assert pwm.shape[0] == 4

    onehot = onehot.float()
    pwm = pwm.float()
    pwm_rev = pwm.flip(1)
    onehot_comp = onehot[:, [3, 2, 1, 0]]
    positive = torch.nn.functional.conv1d(onehot, pwm.unsqueeze(0))[:, 0]

    found_positive = positive >= cutoff
    scores_positive = positive[found_positive]
    positions_positive = torch.stack(torch.where(found_positive)).to(torch.int64)
    positions_positive = positions_positive[0] * (onehot.shape[-1]) + positions_positive[1]

    negative = torch.nn.functional.conv1d(onehot_comp, pwm_rev.unsqueeze(0))[:, 0]

    found_negative = negative >= cutoff
    scores_negative = negative[found_negative]
    positions_negative = torch.stack(torch.where(found_negative)).to(torch.int64)
    positions_negative = (positions_negative[0]) * (onehot.shape[-1]) + positions_negative[1]

    return (
        torch.cat([scores_positive, scores_negative]).cpu().numpy(),
        torch.cat([positions_positive, positions_negative], -1).to(torch.int64).cpu().numpy(),
        torch.cat(
            [
                torch.ones_like(scores_positive, dtype=torch.int8),
                -torch.ones_like(scores_negative, dtype=torch.int8),
            ]
        )
        .cpu()
        .numpy(),
    )


def create_onehots(sequences):
    onehots = []
    for sequence in sequences:
        onehot = np.zeros((len(sequence), 4), dtype=np.int8)
        scan_helpers.seq_to_onehot(sequence.upper().encode(), onehot)
        onehots.append(torch.from_numpy(onehot))
    return torch.stack(onehots).to(torch.float32)


def create_region_onehots(regions: Regions, fasta_file: PathLike, coordinates=None):
    import pysam

    region_onehots = {}
    fasta = pysam.FastaFile(fasta_file)

    if coordinates is None:
        coordinates = regions.coordinates

    for region_id, region in tqdm.tqdm(coordinates.iterrows()):
        # clip start if needed
        start = int(region["start"])
        sequences = fasta.fetch(region["chrom"], np.clip(start, 0, 999999999), region["end"])
        onehot = create_onehots([sequences])

        # pad if needed
        if start < 0:
            onehot = torch.nn.functional.pad(onehot, (0, 0, np.clip(start, 0, 999999999) - start, 0, 0, 0))
        elif onehot.shape[1] < region["end"] - region["start"]:
            onehot = torch.nn.functional.pad(onehot, (0, 0, 0, region["end"] - region["start"] - onehot.shape[1], 0, 0))
        assert onehot.shape[1] == region["end"] - region["start"], (
            onehot.shape,
            region["end"] - region["start"],
            start,
            region["end"],
        )
        if region["strand"] == -1:
            onehot = onehot.flip(1).flip(2)  # reverse complement
        region_onehots[region_id] = onehot[0]
    return region_onehots

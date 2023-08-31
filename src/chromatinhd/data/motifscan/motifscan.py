import pathlib
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

    indices: TensorstoreInstance = Tensorstore(dtype="<i4", chunks=(10000,))
    "Motif index associated to each site"

    positions: TensorstoreInstance = Tensorstore(dtype="<i8", chunks=(10000,))
    "Cumulative coordinate of each site"

    scores: TensorstoreInstance = Tensorstore(dtype="<f4", chunks=(10000,))
    "Scores associated with each detected site"

    strands: TensorstoreInstance = Tensorstore(dtype="<f4", chunks=(10000,))
    "Strand associated with each detected site"

    shape = Stored()

    n_motifs = Stored()
    "Number of motifs"

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

        if ((reuse) or (not overwrite)) and self.get("positions").exists(self):
            if not reuse:
                import warnings

                warnings.warn(
                    "Motifscan already exists. Use overwrite=True to overwrite, reuse=True to ignore this warning."
                )
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
        self.positions.open_creator()

        # do the actual counting by looping over the batches, extract the sequences and scanning
        for batch, region_coordinates_batch in tqdm.tqdm(region_coordinates.groupby("batch")):
            if fasta is None:
                sequences = [
                    fasta.fetch(chrom, start, end + 1)
                    for chrom, start, end in region_coordinates_batch[["chrom", "start", "end"]].values
                ]
                assert (
                    len(set(len(sequence) for sequence in sequences)) == 1
                ), "All regions/sequences should have the same length"
                onehot2 = create_onehots(sequences).permute(0, 2, 1)
            else:
                onehot2 = torch.stack([region_onehots[region] for region in region_coordinates_batch.index]).permute(
                    0, 2, 1
                )
            onehot2 = onehot2.to(device)

            assert onehot2.shape[1] == 4
            assert onehot2.shape[2] == region_coordinates_batch["len"].iloc[0]
            for motif_ix, motif in enumerate(motifs.index):
                cutoff = cutoffs[motif]

                if cutoff < 0:
                    raise ValueError(f"Cutoff for motif {motif} is negative, but should be positive.")

                # get pwm
                pwm = pwms[motif]
                if not torch.is_tensor(pwm):
                    pwm = torch.from_numpy(pwm)
                pwm2 = pwm.to(dtype=torch.float32, device=onehot2.device).transpose(1, 0)

                (
                    scores,
                    positions,
                    strands,
                ) = scan(onehot2, pwm2, cutoff=cutoff)

                coordinates = positions[1].astype(np.int32)
                positions = (
                    self.regions.cumulative_region_lengths[(positions[0] + region_coordinates_batch["ix"].values[0])]
                    + positions[1]
                )

                # sort by position
                sorted_idx = np.argsort(positions)
                indices = np.full_like(positions, motif_ix, dtype=np.int32)
                indices = indices[sorted_idx]
                scores = scores[sorted_idx]
                strands = strands[sorted_idx]
                coordinates = coordinates[sorted_idx]
                positions = positions[sorted_idx]

                self.indices.extend(indices)
                self.scores.extend(scores)
                self.strands.extend(strands)
                self.coordinates.extend(coordinates)
                self.positions.extend(positions)

        return self

    # def create_indptr(self):
    #     """
    #     Populate the indptr
    #     """
    #     self.indptr = ind2ptr_numpy(self.positions[:], self.regions.region_lengths.sum())

    def create_region_indptr(self):
        """
        Populate the region_indptr
        """

        def indices_to_indptr_chunked(gen, n):
            counts = np.zeros(n + 1, dtype=np.int32)
            cur_value = 0
            for values in gen:
                bincount = np.bincount(values - cur_value)
                counts[(cur_value + 1) : (cur_value + len(bincount) + 1)] += bincount
                cur_value = values[-1]
            indptr = np.cumsum(counts, dtype=np.int32)
            return indptr

        def read_motifscan_regions(reader, chunk_size, cumulative_region_lengths):
            for start in range(0, reader.shape[0], chunk_size):
                end = min(start + chunk_size, reader.shape[0])
                positions = reader[start:end]
                values = np.searchsorted(cumulative_region_lengths, positions) - 1
                yield values

        positions_reader = self.positions.open_reader()

        self.region_indptr = indices_to_indptr_chunked(
            read_motifscan_regions(positions_reader, 10000, self.regions.cumulative_region_lengths),
            self.regions.n_regions,
        )

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


def divide_regions_in_batches(region_coordinates, batch_size=10):
    region_coordinates["len"] = region_coordinates["end"] - region_coordinates["start"]
    region_coordinates["cumlen"] = (region_coordinates["end"] - region_coordinates["start"]).cumsum()
    region_coordinates["same_length"] = region_coordinates["len"] == region_coordinates["len"].iloc[0]
    region_coordinates["cumlen"] = region_coordinates["cumlen"] + (batch_size * ~region_coordinates["same_length"])
    region_coordinates["batch"] = (region_coordinates["cumlen"] // batch_size).astype(int)
    region_coordinates["batch"] = np.argsort(region_coordinates["batch"])

    region_coordinates["ix"] = np.arange(region_coordinates.shape[0])

    return region_coordinates


def read_pwms(pwms_file):
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

    return pwms


def scan(onehot, pwm, cutoff=0.0):
    assert onehot.shape[1] == 4
    assert pwm.shape[0] == 4

    k = pwm.shape[1]

    onehot = onehot.float()
    pwm = pwm.float()
    pwm_rev = pwm.flip(1)
    onehot_comp = onehot[:, [3, 2, 1, 0]]
    positive = torch.nn.functional.conv1d(onehot, pwm.unsqueeze(0))[:, 0]

    found_positive = positive >= cutoff
    scores_positive = positive[found_positive]
    positions_positive = torch.stack(torch.where(found_positive)).to(torch.int64)

    negative = torch.nn.functional.conv1d(onehot_comp, pwm_rev.unsqueeze(0))[:, 0]

    found_negative = negative >= cutoff
    scores_negative = negative[found_negative]
    positions_negative = torch.stack(torch.where(found_negative)).to(torch.int64)
    positions_negative[-1, :] = positions_negative[-1, :] + k - 1

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


def create_region_onehots(regions: Regions, fasta_file: PathLike):
    import pysam

    region_onehots = {}
    fasta = pysam.FastaFile(fasta_file)

    for region_id, region in tqdm.tqdm(regions.coordinates.iterrows()):
        sequences = fasta.fetch(region["chrom"], region["start"], region["end"] + 1)
        onehot = create_onehots([sequences])
        region_onehots[region_id] = onehot[0]
    return region_onehots

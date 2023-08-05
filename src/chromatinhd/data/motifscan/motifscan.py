import pathlib
from typing import Union

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
)
from chromatinhd.utils.torch import ind2ptr


class Motifscan(Flow):
    """
    A sprase representation of locations of different motifs in regions of the genome
    """

    regions = Linked()
    "The regions"

    indptr = CompressedNumpyInt64()
    "The index pointers for each position in the regions"

    position = CompressedNumpyInt64()
    "Position associated to each site"

    indices = CompressedNumpyInt64()
    "Motif index associated to each site"

    scores = CompressedNumpyFloat64()
    "Scores associated with each detected site"

    strands = CompressedNumpyFloat64()
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
        fasta_file: Union[str, pathlib.Path],
        path: Union[str, pathlib.Path],
        cutoffs: Union[int, float, pd.Series] = None,
        cutoff_col: str = None,
        motifs: pd.DataFrame = None,
        device=None,
        batch_size: int = 5000000,
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
            motifs:
                A dataframe containing auxilliary information for each motif
            path:
                The folder where the motifscan data will be stored.
            cutoffs:
                A dictionary containing the cutoffs for each motif.
            cutoff_col:
                The column in the motifs dataframe containing the cutoffs
            device:
                The device to use for the scanning
            batch_size:
                The batch size to use for scanning. Lower batch size if the GPU runs out of memory
        """

        if device is None:
            device = get_default_device()

        self = cls(path)

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

        region_size = region_coordinates["end"].values[0] - region_coordinates["start"].values[0]

        # load in fasta file
        import pysam

        fasta = pysam.FastaFile(fasta_file)

        # do the actual counting by looping over the batches, extract the sequences and scan
        positions = []
        indices = []
        scores = []
        strands = []

        for batch, region_coordinates_batch in tqdm.tqdm(region_coordinates.groupby("batch")):
            sequences = [
                fasta.fetch(chrom, start, end + 1)
                for chrom, start, end in region_coordinates_batch[["chrom", "start", "end"]].values
            ]
            assert (
                len(set(len(sequence) for sequence in sequences)) == 1
            ), "All regions/sequences should have the same length"
            onehot = torch.stack([create_onehot(digitize_sequence(sequence)) for sequence in sequences]).to(device)
            for motif_ix, motif in enumerate(motifs.index):
                cutoff = cutoffs[motif]

                # get pwm
                pwm = pwms[motif]
                if not torch.is_tensor(pwm):
                    pwm = torch.from_numpy(pwm)
                pwm = pwm.to(dtype=torch.float32, device=onehot.device)

                (
                    scores_motif,
                    positions_motif,
                    strands_motif,
                ) = scan(onehot, pwm, cutoff=cutoff)

                positions_motif[0] = positions_motif[0] + region_coordinates_batch["ix"].values[0]

                positions.append(positions_motif)
                indices.append(torch.ones_like(scores_motif, dtype=torch.int) * motif_ix)
                scores.append(scores_motif)
                strands.append(strands_motif)

        # positions = [batch_dim (region and position), site]
        positions = torch.concat(positions, -1)
        positions = positions[0] * region_size + positions[1]
        indices = torch.concat(indices, -1)
        scores = torch.concat(scores, -1)
        strands = torch.concat(strands, -1)

        # sort by position
        sorted_idx = torch.argsort(positions)
        positions = positions[sorted_idx]
        indices = indices[sorted_idx]
        scores = scores[sorted_idx]
        strands = strands[sorted_idx]

        # calculate indptr
        indptr = ind2ptr(positions, region_size * len(region_coordinates))

        # store
        self.positions = positions.cpu().numpy()
        self.indptr = indptr.cpu().numpy()
        self.indices = indices.cpu().numpy()
        self.scores = scores.cpu().numpy()
        self.strands = strands.cpu().numpy()
        self.motifs = motifs

        return self


def divide_regions_in_batches(region_coordinates, batch_size=10):
    region_coordinates["len"] = region_coordinates["end"] - region_coordinates["start"]

    assert len(region_coordinates["len"].unique()) == 1, "All regions should have the same size"

    region_coordinates["cumlen"] = (region_coordinates["end"] - region_coordinates["start"]).cumsum()

    region_coordinates["batch"] = (region_coordinates["cumlen"] // batch_size).astype(int)

    region_coordinates["batch"] = (region_coordinates["batch"] - region_coordinates["batch"].min()).astype(int)

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


def scan(onehot, pwm, cutoff=0):
    n = onehot.shape[-2]
    k = pwm.shape[-2]

    # forward strand
    positive = torch.zeros(((*onehot.shape[:-2], n - k + 1)), device=onehot.device)
    for i in range(k):
        # to save memory we do the matrix multiplication once per motif position
        # this does not cause a significant slowdown
        x = torch.matmul(onehot, pwm[[i]].T)
        positive += x[..., i : n - k + i + 1, 0]
    del x

    found_positive = positive >= cutoff
    scores_positive = positive[found_positive]
    positions_positive = torch.stack(torch.where(found_positive)).to(torch.int)

    # reverse (complement) strand
    onehot_comp = onehot[..., [3, 2, 1, 0]]
    pwm_rev = pwm.flip(0)
    negative = torch.zeros(((*onehot.shape[:-2], n - k + 1)), device=onehot.device)
    for i in range(k):
        x = torch.matmul(onehot_comp, pwm_rev[[i]].T)
        negative += x[..., i : n - k + i + 1, 0]
    del x

    found_negative = negative >= cutoff
    scores_negative = negative[found_negative]
    positions_negative = torch.stack(torch.where(found_negative)).to(torch.int)
    positions_negative[-1, :] = positions_negative[-1, :] + k - 1

    # return stacked score across forward or reverse strands
    return (
        torch.cat([scores_positive, scores_negative]),
        torch.cat([positions_positive, positions_negative], -1).to(torch.int),
        torch.cat(
            [
                torch.ones_like(scores_positive, dtype=torch.int8),
                -torch.ones_like(scores_negative, dtype=torch.int8),
            ]
        ),
    )


def create_onehot(seq):
    """
    Sequence contains integers 0 (A), 1 (C), 2 (G), 3 (T), and 4 (N)
    """
    return torch.tensor(np.eye(5, dtype=np.float32)[seq][:, :-1])


def digitize_sequence(sequence):
    translate_table = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "N": 4,
        "a": 0,
        "c": 1,
        "g": 2,
        "t": 3,
    }  # alphabetic order
    return np.array([translate_table[x] for x in sequence], dtype=np.int8)

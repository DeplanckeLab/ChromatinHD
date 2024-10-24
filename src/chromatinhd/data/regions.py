from __future__ import annotations

from typing import List, Optional
import pathlib

import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import functools

from chromatinhd.flow import TSV, Flow, PathLike, Stored


def center(slices, promoter, columns=["start", "end"]):
    slices = slices.copy()
    if slices.shape[0] == 0:
        slices = pd.DataFrame(columns=["start", "end"])
    else:
        slices[columns] = [
            [
                (peak["start"] - promoter["tss"]) * int(promoter["strand"]),
                (peak["end"] - promoter["tss"]) * int(promoter["strand"]),
            ][:: int(promoter["strand"])]
            for _, peak in slices.iterrows()
        ]
    return slices


def center_multiple(slices, coordinates):
    assert coordinates.index.name in slices.columns

    slices = slices[[col for col in slices.columns if col != "strand"]].join(
        coordinates[["tss", "strand"]], on=coordinates.index.name, rsuffix="_genome"
    )
    slices[["start", "end"]] = [
        [
            (slice["start"] - slice["tss"]) * int(slice["strand"]),
            (slice["end"] - slice["tss"]) * int(slice["strand"]),
        ][:: int(slice["strand"])]
        for _, slice in slices.iterrows()
    ]
    return slices


def uncenter(slices, promoter, columns=["start", "end"]):
    slices = slices.copy()
    if slices.shape[0] == 0:
        slices = pd.DataFrame(columns=["start", "end"])
    elif isinstance(slices, pd.Series):
        slices[columns] = [
            (slices["start"] * int(promoter["strand"]) + promoter["tss"]),
            (slices["end"] * int(promoter["strand"]) + promoter["tss"]),
        ][:: int(promoter["strand"])]
    else:
        slices[columns] = [
            [
                (peak["start"] * int(promoter["strand"]) + promoter["tss"]),
                (peak["end"] * int(promoter["strand"]) + promoter["tss"]),
            ][:: int(promoter["strand"])]
            for _, peak in slices.iterrows()
        ]
    slices["chrom"] = promoter["chrom"]
    return slices


def uncenter_multiple(slices, coordinates):
    if "region_ix" not in slices.columns:
        slices["region_ix"] = coordinates.index.get_indexer(slices["region"])
    coordinates_oi = coordinates.iloc[slices["region_ix"]].copy()

    slices["chrom"] = coordinates_oi["chrom"].values

    slices["start_genome"] = np.where(
        coordinates_oi["strand"] == 1,
        (slices["start"] * coordinates_oi["strand"].astype(int).values + coordinates_oi["tss"].values),
        (slices["end"] * coordinates_oi["strand"].astype(int).values + coordinates_oi["tss"].values),
    )
    slices["end_genome"] = np.where(
        coordinates_oi["strand"] == 1,
        (slices["end"] * coordinates_oi["strand"].astype(int).values + coordinates_oi["tss"].values),
        (slices["start"] * coordinates_oi["strand"].astype(int).values + coordinates_oi["tss"].values),
    )
    return slices


class Regions(Flow):
    """
    Regions in the genome
    """

    coordinates = TSV(columns=["chrom", "start", "end"])
    "Coordinates dataframe of the regions, with columns chrom, start, end"

    window = Stored()

    @classmethod
    def from_transcripts(
        cls,
        transcripts: pd.DataFrame,
        window: [list, np.ndarray],
        path: PathLike = None,
        max_n_regions: Optional[int] = None,
        overwrite=True,
    ) -> Regions:
        """
        Create regions from a dataframe of transcripts,
        using a specified window around each transcription start site.

        Parameters:
            transcripts:
                Dataframe of transcripts, with columns chrom, start, end, strand, transcript
            window:
                Window around each transcription start site. Should be a 2-element array, e.g. [-10000, 10000]
            path:
                Folder in which the regions data will be stored
            max_n_regions:
                Maximum number of region to use. If None, all regions are used.
        Returns:
            Regions
        """
        transcripts["tss"] = transcripts["start"] * (transcripts["strand"] == 1) + transcripts["end"] * (transcripts["strand"] == -1)

        regions = transcripts[["chrom", "tss", "transcript"]].copy()

        regions["strand"] = transcripts["strand"]
        regions["positive_strand"] = (regions["strand"] == 1).astype(int)
        regions["negative_strand"] = (regions["strand"] == -1).astype(int)
        regions["chrom"] = transcripts.loc[regions.index, "chrom"]

        regions["start"] = (regions["tss"] + window[0] * (regions["strand"] == 1) - window[1] * (regions["strand"] == -1)).astype(int)
        regions["end"] = (regions["tss"] + window[1] * (regions["strand"] == -1) - window[0] * (regions["strand"] == 1)).astype(int)

        if max_n_regions is not None:
            regions = regions.iloc[:max_n_regions]

        return cls.create(
            path=path,
            coordinates=regions[["chrom", "start", "end", "tss", "strand", "transcript"]],
            window=window,
            reset=overwrite,
        )

    def filter(self, region_ids: List[str], path: PathLike = None, overwrite=True) -> Regions:
        """
        Select a subset of regions

        Parameters:
            region_ids:
                Genes to filter. Should be a pandas Series with the index being the ensembl transcript ids.
            path:
                Path to store the filtered regions
        Returns:
            Regions with only the specified region_ids
        """

        return Regions.create(coordinates=self.coordinates.loc[region_ids], window=self.window, path=path, reset=overwrite)

    @property
    def window_width(self):
        if self.window is None:
            return None
        return self.window[1] - self.window[0]

    region_width = window_width
    width = window_width
    "Width of the regions, None if regions do not have a fixed width"

    @classmethod
    def from_chromosomes_file(cls, chromosomes_file: PathLike, path: PathLike = None, filter_chromosomes=True, overwrite: bool = True) -> Regions:
        """
        Create regions based on a chromosomes file, e.g. hg38.chrom.sizes

        Parameters:
            chromosomes_file:
                Path to chromosomes file, tab separated, with columns chrom, size
            path:
                Folder in which the regions data will be stored
        Returns:
            Regions
        """

        chromosomes = pd.read_csv(chromosomes_file, sep="\t", names=["chrom", "size"])
        chromosomes["start"] = 0
        chromosomes["end"] = chromosomes["size"]
        chromosomes["strand"] = 1
        chromosomes = chromosomes[["chrom", "start", "end", "strand"]]
        chromosomes = chromosomes.set_index("chrom", drop=False)
        chromosomes.index.name = "region"

        if filter_chromosomes:
            chromosomes = chromosomes.loc[~chromosomes["chrom"].isin(["chrM", "chrMT"])]
            chromosomes = chromosomes.loc[~chromosomes["chrom"].str.contains("_")]
            chromosomes = chromosomes.loc[~chromosomes["chrom"].str.contains(r"\.")]

        chromosomes = chromosomes.sort_values("chrom")

        return cls.create(
            path=path,
            coordinates=chromosomes,
            window=None,
            reset=overwrite,
        )

    @functools.cached_property
    def n_regions(self):
        return self.coordinates.shape[0]

    @functools.cached_property
    def region_lengths(self):
        return (self.coordinates["end"] - self.coordinates["start"]).values

    @functools.cached_property
    def region_starts(self):
        return self.coordinates["start"].values

    @functools.cached_property
    def cumulative_region_lengths(self):
        return np.pad(np.cumsum(self.coordinates["end"].values - self.coordinates["start"].values), (1, 0))

    @property
    def var(self):
        return self.coordinates

    @var.setter
    def var(self, value):
        self.coordinates = value

    def __len__(self):
        return self.n_regions


def select_tss_from_fragments(transcripts: pd.DataFrame, fragments_file: PathLike, window: [np.ndarray, tuple] = (-100, 100)) -> pd.DataFrame:
    """
    Select the TSS with the most fragments within a window of the TSS

    Parameters:
        transcripts:
            Dataframe of transcripts, with columns chrom, tss, ensembl_gene_id.
        fragments_file:
            Path to fragments file
        window:
            Window around the TSS to count fragments
    Returns:
        Dataframe of transcripts, with columns chrom, tss and n_fragments, with index being the gene id
    """
    if not ([col in transcripts.columns for col in ["chrom", "tss", "ensembl_gene_id"]]):
        raise ValueError("Transcripts should have columns chrom, tss, ensembl_gene_id. ")

    try:
        import pysam
    except ImportError:
        raise ImportError("Please install the pysam package `pip install pysam` or `conda install pysam`")

    try:
        fragments_tabix = pysam.TabixFile(str(fragments_file))
    except OSError as error:
        raise ValueError("fragments file is not indexed, please run `pysam.tabix_index(file, preset = 'bed')`") from error

    nfrags = []
    for chrom, tss in tqdm.tqdm(zip(transcripts["chrom"], transcripts["tss"]), total=transcripts.shape[0]):
        frags = list(fragments_tabix.fetch(chrom, tss + window[0], tss + window[1]))
        nfrags.append(len(frags))
    transcripts["n_fragments"] = nfrags
    selected_transcripts = transcripts.reset_index().sort_values("n_fragments", ascending=False).groupby("ensembl_gene_id").first()
    selected_transcripts.index.name = "gene"
    return selected_transcripts

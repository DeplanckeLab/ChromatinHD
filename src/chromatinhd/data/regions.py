from __future__ import annotations
from chromatinhd.flow import Flow, Stored, TSV
import pandas as pd
import numpy as np
import pathlib
import tqdm.auto as tqdm
from chromatinhd.flow import PathLike


class Regions(Flow):
    """
    Regions, typically centered around a transcription start site
    """

    coordinates = TSV(columns=["chrom", "start", "end"])
    window = Stored()

    @classmethod
    def from_transcripts(cls, transcripts: pd.DataFrame, window: [list, np.ndarray], path: pathlib.Path) -> Regions:
        """
        Create regions from a Dataframe of canonical transcripts,
        using a specified window around each transcription start site.

        Parameters:
            transcripts:
                Dataframe of canonical transcripts, with columns chrom, start, end, strand, ensembl_transcript_id
            window:
                Window around each transcription start site. Should be a 2-element array, e.g. [-10000, 10000]
            path:
                Folder in which the regions data will be stored
        """
        transcripts["tss"] = transcripts["start"] * (transcripts["strand"] == 1) + transcripts["end"] * (
            transcripts["strand"] == -1
        )

        regions = transcripts[["chrom", "tss", "ensembl_transcript_id"]].copy()

        regions["strand"] = transcripts["strand"]
        regions["positive_strand"] = (regions["strand"] == 1).astype(int)
        regions["negative_strand"] = (regions["strand"] == -1).astype(int)
        regions["chrom"] = transcripts.loc[regions.index, "chrom"]

        regions["start"] = regions["tss"] + window[0] * (regions["strand"] == 1) - window[1] * (regions["strand"] == -1)
        regions["end"] = regions["tss"] + window[1] * (regions["strand"] == -1) - window[0] * (regions["strand"] == 1)

        return cls.create(
            path=path,
            coordinates=regions[["chrom", "start", "end", "tss", "strand", "ensembl_transcript_id"]],
            window=window,
        )

    def filter_genes(self, genes, path=None) -> Regions:
        """
        Filter genes to those in the regions

        Parameters:
            genes:
                Genes to filter. Should be a pandas Series with the index being the ensembl transcript ids.
            path:
                Path to store the filtered regions
        Returns:
            Regions with only the specified genes
        """

        return Regions.create(coordinates=self.coordinates.loc[genes], window=self.window, path=path)


def select_tss_from_fragments(
    transcripts: pd.DataFrame, fragments_file: PathLike, window: [np.ndarray, tuple] = (-100, 100)
) -> pd.DataFrame:
    """
    Select the TSS with the most fragments within a window of the TSS

    Parameters:
        transcripts:
            Dataframe of transcripts, with columns chrom, tss, ensembl_gene_id
        fragments_file:
            Path to fragments file
        window:
            Window around the TSS to count fragments
    Returns:
        Dataframe of transcripts, with columns chrom, tss, ensembl_gene_id, n_fragments
    """
    if not ([col in transcripts.columns for col in ["chrom", "tss", "ensembl_gene_id"]]):
        raise ValueError("Transcripts should have columns chrom, tss, ensembl_gene_id. ")

    import pysam

    fragments_tabix = pysam.TabixFile(str(fragments_file))

    nfrags = []
    for chrom, tss in tqdm.tqdm(zip(transcripts["chrom"], transcripts["tss"]), total=transcripts.shape[0]):
        frags = list(fragments_tabix.fetch(chrom, tss + window[0], tss + window[1]))
        nfrags.append(len(frags))
    transcripts["n_fragments"] = nfrags
    selected_transcripts = (
        transcripts.reset_index().sort_values("n_fragments", ascending=False).groupby("ensembl_gene_id").first()
    )
    selected_transcripts.index.name = "gene"
    return selected_transcripts

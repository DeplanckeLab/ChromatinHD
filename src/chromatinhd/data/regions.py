from chromatinhd.flow import Flow, Stored, TSV
import typing
import pandas as pd
import numpy as np
import pathlib

class Regions(Flow):
    """
    Regions, typically centered around a transcription start site
    """

    coordinates = TSV("coordinates", columns = ["chrom", "start", "end"])
    window = Stored("window")
    
    @classmethod
    def from_genes(cls, genes:pd.DataFrame, window:np.ndarray, path:pathlib.Path):
        regions = genes[["chrom", "start", "end"]].copy()

        regions["tss"] = [
            genes_row["start"] if genes_row["strand"] == +1 else genes_row["end"]
            for _, genes_row in genes.loc[regions.index].iterrows()
        ]
        regions["strand"] = genes["strand"]
        regions["positive_strand"] = (regions["strand"] == 1).astype(int)
        regions["negative_strand"] = (regions["strand"] == -1).astype(int)
        regions["chrom"] = genes.loc[regions.index, "chrom"]

        regions["start"] = (
            regions["tss"]
            + window[0] * (regions["strand"] == 1)
            - window[1] * (regions["strand"] == -1)
        )
        regions["end"] = (
            regions["tss"]
            + window[1] * (regions["strand"] == -1)
            - window[0] * (regions["strand"] == 1)
        )

        return cls.create(
            path = path,
            coordinates = regions[["chrom", "start", "end", "tss", "strand"]],
            window = window
        )
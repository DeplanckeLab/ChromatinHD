from chromatinhd.flow import Flow, CompressedNumpyInt64, CompressedNumpyFloat64, Stored
from chromatinhd.utils import indptr_to_indices
import pandas as pd


class Motifscan(Flow):
    indptr = CompressedNumpyInt64("indptr")
    indices = CompressedNumpyInt64("indices")
    data = CompressedNumpyFloat64("data")
    shape = Stored("shape")
    n_motifs = Stored("n_motifs")

    _motifs = None

    @property
    def motifs(self):
        if self._motifs is None:
            self._motifs = pd.read_pickle(self.path / "motifs.pkl")
        return self._motifs

    @motifs.setter
    def motifs(self, value):
        value.index.name = "gene"
        value.to_pickle(self.path / "motifs.pkl")
        self._motifs = value


class GWAS(Motifscan):
    association = Stored("association")
    window = Stored("window")

    def get_motifdata(self, promoter: pd.Series):
        assert "ix" in promoter.index
        gene_ix = promoter["ix"]

        indptr_start = gene_ix * (self.window[1] - self.window[0])
        indptr_end = (gene_ix + 1) * (self.window[1] - self.window[0])

        print(indptr_start, indptr_end)

        indptr = self.indptr[indptr_start:indptr_end]
        motif_indices = self.indices[indptr[0] : indptr[-1]]
        position_indices = indptr_to_indices(indptr - indptr[0]) + self.window[0]

        plotdata_snps = pd.DataFrame(
            {
                "position": position_indices,
                "motif": self.motifs.iloc[motif_indices].index.values,
            }
        )
        plotdata_snps = (
            plotdata_snps.groupby("position").agg({"motif": list}).reset_index()
        )

        for position in plotdata_snps["position"]:
            genome_position = (
                promoter["tss"]
                + position * promoter["strand"]
                + 1 * (promoter["strand"] == -1)
            )
            assoc = self.association.loc[
                (self.association["chr"] == promoter.chr)
                & (self.association["start"] == genome_position)
            ]
            if len(assoc) > 0:
                plotdata_snps.loc[
                    plotdata_snps["position"] == position, "rsid"
                ] = assoc["snp"].values[0]
                plotdata_snps.loc[
                    plotdata_snps["position"] == position, "snp_main"
                ] = assoc["snp_main"].values[0]
        return plotdata_snps

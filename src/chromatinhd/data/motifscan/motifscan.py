from chromatinhd.flow import Flow, CompressedNumpyInt64, CompressedNumpyFloat64, Stored
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

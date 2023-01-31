from chromatinhd.flow import Flow, CompressedNumpyInt64, CompressedNumpyFloat64, Stored
import pandas as pd


class Motiftrack(Flow):
    scores = CompressedNumpyFloat64("scores")

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

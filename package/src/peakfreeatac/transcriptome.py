import numpy as np
import pandas as pd
import pickle

import dataclasses
import pathlib

@dataclasses.dataclass
class Transcriptome():
    path:pathlib.Path

    @property
    def var(self):
        return pd.read_table(self.path / "var.tsv", index_col = 0)
    @var.setter
    def var(self, value):
        value.index.name = "peak"
        value.to_csv(self.path / "var.tsv", sep = "\t")

    @property
    def obs(self):
        return pd.read_table(self.path / "obs.tsv", index_col = 0)
    @obs.setter
    def obs(self, value):
        value.index.name = "peak"
        value.to_csv(self.path / "obs.tsv", sep = "\t")

    _adata = None
    @property
    def adata(self):
        if self._adata is None:
            self._adata = pickle.load((self.path / "adata.pkl").open("rb"))
        return self._adata
    @adata.setter
    def adata(self, value):
        pickle.dump(value, (self.path / "adata.pkl").open("wb"))
        self._adata = value

import numpy as np
import pandas as pd
import pickle

import pathlib

from peakfreeatac.flow import Flow

class Fragments(Flow):
    _coordinates = None
    @property
    def coordinates(self):
        if self._coordinates is None:
            self._coordinates = pickle.load((self.path / "coordinates.pkl").open("rb"))
        return self._coordinates
    @coordinates.setter
    def coordinates(self, value):
        pickle.dump(value, (self.path / "coordinates.pkl").open("wb"))
        self._coordinates = value

    _mapping = None
    @property
    def mapping(self):
        if self._mapping is None:
            self._mapping = pickle.load((self.path / "mapping.pkl").open("rb"))
        return self._mapping
    @mapping.setter
    def mapping(self, value):
        pickle.dump(value, (self.path / "mapping.pkl").open("wb"))
        self._mapping = value

    @property
    def var(self):
        return pd.read_table(self.path / "var.tsv", index_col = 0)
    @var.setter
    def var(self, value):
        value.index.name = "gene"
        value.to_csv(self.path / "var.tsv", sep = "\t")

    @property
    def obs(self):
        return pd.read_table(self.path / "obs.tsv", index_col = 0)
    @obs.setter
    def obs(self, value):
        value.index.name = "cell"
        value.to_csv(self.path / "obs.tsv", sep = "\t")

    _n_genes = None
    @property
    def n_genes(self):
        if self._n_genes is None:
            self._n_genes = self.var.shape[0]
        return self._n_genes
    _n_cells = None
    @property
    def n_cells(self):
        if self._n_cells is None:
            self._n_cells = self.obs.shape[0]
        return self._n_cells
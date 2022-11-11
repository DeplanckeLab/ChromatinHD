import numpy as np
import pandas as pd
import pickle

from peakfreeatac.flow import Flow

class Transcriptome(Flow):
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

    def gene_id(self, symbol):
        assert all(pd.Series(symbol).isin(self.var["symbol"])), set(
            pd.Series(symbol)[~pd.Series(symbol).isin(self.var["symbol"])]
        )
        return self.var.reset_index("gene").set_index("symbol").loc[symbol]["gene"]

    def symbol(self, gene_id):
        assert all(pd.Series(gene_id).isin(self.var.index)), set(
            pd.Series(gene_id)[~pd.Series(gene_id).isin(self.var.index)]
        )
        return self.var.loc[gene_id]["symbol"]

    def create_X(self):
        X_scipy = self.adata.X
        import latenta as la
        X = la.sparse.COOMatrix.from_scipy_csr(X_scipy)
        X.populate_mapping()

        self.X = X

    _X = None
    @property
    def X(self):
        if self._X is None:
            self._X = pickle.load((self.path / "X.pkl").open("rb"))
        return self._X
    @X.setter
    def X(self, value):
        pickle.dump(value, (self.path / "X.pkl").open("wb"))
        self._X = value
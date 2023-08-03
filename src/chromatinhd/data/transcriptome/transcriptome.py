import numpy as np
import pandas as pd
import pickle

from chromatinhd.flow import Flow, Stored, StoredDict
from chromatinhd import sparse
from chromatinhd.utils import Unpickler


class Transcriptome(Flow):
    _var = None

    @property
    def var(self):
        if self._var is None:
            self._var = pd.read_table(self.path / "var.tsv", index_col=0)
        return self._var

    @var.setter
    def var(self, value):
        value.index.name = "gene"
        value.to_csv(self.path / "var.tsv", sep="\t")
        self._var = value

    _obs = None

    @property
    def obs(self):
        if self._obs is None:
            self._obs = pd.read_table(self.path / "obs.tsv", index_col=0)
        return self._obs

    @obs.setter
    def obs(self, value):
        value.index.name = "cell"
        value.to_csv(self.path / "obs.tsv", sep="\t")
        self._obs = value

    adata = Stored("adata")
    """
    Anndata object containing the transcriptome data.
    """

    def gene_id(self, symbol):
        """
        Get the gene id for a given gene symbol.
        """
        assert all(pd.Series(symbol).isin(self.var["symbol"])), set(
            pd.Series(symbol)[~pd.Series(symbol).isin(self.var["symbol"])]
        )
        return self.var.reset_index("gene").set_index("symbol").loc[symbol]["gene"]

    def symbol(self, gene_id):
        """
        Get the gene symbol for a given gene ID (e.g. Ensembl ID)
        """
        assert all(pd.Series(gene_id).isin(self.var.index)), set(
            pd.Series(gene_id)[~pd.Series(gene_id).isin(self.var.index)]
        )
        return self.var.loc[gene_id]["symbol"]

    def gene_ix(self, symbol):
        """
        Get the gene index for a given gene symbol.
        """
        self.var["ix"] = np.arange(self.var.shape[0])
        assert all(pd.Series(symbol).isin(self.var["symbol"])), set(
            pd.Series(symbol)[~pd.Series(symbol).isin(self.var["symbol"])]
        )
        return self.var.reset_index("gene").set_index("symbol").loc[symbol]["ix"]

    def create_X(self):
        X_scipy = self.adata.X
        if isinstance(X_scipy, np.ndarray):
            import scipy.sparse

            X_scipy = scipy.sparse.csr_matrix(X_scipy)
        X = sparse.COOMatrix.from_scipy_csr(X_scipy)
        X.populate_mapping()

        self.X = X

    X = Stored("X")

    @classmethod
    def from_adata(cls, adata, path):
        """
        Create a Transcriptome object from an AnnData object.
        """
        transcriptome = cls(path=path)
        transcriptome.adata = adata
        transcriptome.layers["X"] = adata.X
        transcriptome.var = adata.var
        transcriptome.obs = adata.obs
        return transcriptome

    layers = StoredDict("layers", Stored)
    """
    Dictionary of layers, such as raw, normalized and imputed data.
    """


class ClusterTranscriptome(Flow):
    var = Stored("var")
    obs = Stored("obs")
    adata = Stored("adata")
    X = Stored("X")

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

    def gene_ix(self, symbol):
        self.var["ix"] = np.arange(self.var.shape[0])
        assert all(pd.Series(symbol).isin(self.var["symbol"])), set(
            pd.Series(symbol)[~pd.Series(symbol).isin(self.var["symbol"])]
        )
        return self.var.reset_index("gene").set_index("symbol").loc[symbol]["ix"]


class ClusteredTranscriptome(Flow):
    donors_info = Stored("donors_info")
    clusters_info = Stored("clusters_info")
    var = Stored("var")
    X = Stored("X")

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

    def gene_ix(self, symbol):
        self.var["ix"] = np.arange(self.var.shape[0])
        assert all(pd.Series(symbol).isin(self.var["symbol"])), set(
            pd.Series(symbol)[~pd.Series(symbol).isin(self.var["symbol"])]
        )
        return self.var.reset_index("gene").set_index("symbol").loc[symbol]["ix"]

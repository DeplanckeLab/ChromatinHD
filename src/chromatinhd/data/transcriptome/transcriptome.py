import numpy as np
import pandas as pd
import pickle
import pathlib
from typing import Union

from chromatinhd.flow import Flow, Stored, StoredDict, TSV
from chromatinhd import sparse
from chromatinhd.utils import Unpickler


class Transcriptome(Flow):
    """
    A transcriptome containing counts for each gene in each cell.
    """

    var = TSV("var", index_name="gene")
    obs = TSV("obs", index_name="gene")

    adata = Stored("adata")
    "Anndata object containing the transcriptome data."

    def gene_id(self, symbol, column="symbol"):
        """
        Get the gene id for a given gene symbol.
        """
        assert all(pd.Series(symbol).isin(self.var[column])), set(
            pd.Series(symbol)[~pd.Series(symbol).isin(self.var[column])]
        )
        return self.var.reset_index("gene").set_index(column).loc[symbol]["gene"]

    def symbol(self, gene_id, column="symbol"):
        """
        Get the gene symbol for a given gene ID (e.g. Ensembl ID).
        """
        assert all(pd.Series(gene_id).isin(self.var.index)), set(
            pd.Series(gene_id)[~pd.Series(gene_id).isin(self.var.index)]
        )
        return self.var.loc[gene_id][column]

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
    "Raw counts for each gene in each cell."

    @classmethod
    def from_adata(cls, adata, path: Union[pathlib.Path, str]):
        """
        Create a Transcriptome object from an AnnData object.

        Parameters:
            adata:
                Anndata object containing the transcriptome data.
            path:
                Folder in which the transcriptome data will be stored.
        """
        transcriptome = cls(path=path)
        transcriptome.adata = adata
        transcriptome.layers["X"] = adata.X
        transcriptome.var = adata.var
        transcriptome.obs = adata.obs
        return transcriptome

    layers = StoredDict("layers", Stored)
    "Dictionary of layers, such as raw, normalized and imputed data."


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

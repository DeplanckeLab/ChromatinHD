import numpy as np
import pandas as pd
import pathlib
from typing import Union

from chromatinhd.flow import Flow, Stored, StoredDict, TSV
from chromatinhd import sparse


class Transcriptome(Flow):
    """
    A transcriptome containing counts for each gene in each cell.
    """

    var: pd.DataFrame = TSV(index_name="gene")
    obs: pd.DataFrame = TSV(index_name="cell")

    adata = Stored()
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

    X = Stored()
    "The main transcriptome data, typically normalized counts."

    @classmethod
    def from_adata(cls, adata, path: Union[pathlib.Path, str] = None):
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

        for k, v in adata.layers.items():
            transcriptome.layers[k] = v
        transcriptome.X = adata.X
        transcriptome.var = adata.var
        transcriptome.obs = adata.obs
        return transcriptome

    layers = StoredDict(Stored)
    "Dictionary of layers, such as raw, normalized and imputed data."

    def filter_genes(self, genes, path=None):
        """
        Filter genes

        Parameters:
            genes:
                Genes to filter. Should be a pandas Series with the index being the ensembl transcript ids.
        """

        self.var["ix"] = np.arange(self.var.shape[0])
        gene_ixs = self.var["ix"].loc[genes]

        layers = {}
        for k, v in self.layers.items():
            layers[k] = v[:, gene_ixs]
        X = self.X[:, gene_ixs]

        return Transcriptome.create(
            var=self.var.loc[genes],
            obs=self.obs,
            X=X,
            layers=layers,
            path=path,
        )

    def get_X(self, gene_ids):
        """
        Get the counts for a given set of genes.
        """
        gene_ixs = self.var.index.get_loc(gene_ids)
        value = self.X[:, gene_ixs]

        if sparse.is_scipysparse(value):
            value = np.array(value.todense())
            if isinstance(gene_ids, str):
                value = value[:, 0]
        return value


class ClusterTranscriptome(Flow):
    var = Stored()
    obs = Stored()
    adata = Stored()
    X = Stored()

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
    donors_info = Stored()
    clusters_info = Stored()
    var = Stored()
    X = Stored()

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

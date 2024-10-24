from __future__ import annotations

import numpy as np
import pandas as pd
import pathlib
from typing import Union

from chromatinhd.flow import Flow, Stored, StoredDict, TSV
from chromatinhd.flow.tensorstore import Tensorstore
from chromatinhd import sparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import scanpy as sc

def get_diffexp(adata, key = "rank_genes_groups", groups = None):
    import scanpy as sc
    groups = adata.uns[key]["names"].dtype.names
    diffexp = pd.concat(
        [
            sc.get.rank_genes_groups_df(adata, group=group, key = key)
            .assign(symbol=lambda x: adata.var.loc[x.names]["symbol"].values)
            .assign(group=group)
            .rename(columns={"names": "gene"})
            for group in groups
        ]
    ).set_index(["group", "gene"])
    return diffexp

class Transcriptome(Flow):
    """
    A transcriptome containing counts for each gene in each cell.
    """

    var: pd.DataFrame = TSV(index_name="gene")
    obs: pd.DataFrame = TSV(index_name="cell")

    adata = Stored()
    "Anndata object containing the transcriptome data."

    def gene_id(self, symbol, column="symbol", optional=False, found=False):
        """
        Get the gene id for a given gene symbol.
        """
        if found:
            gene_id = self.var.reset_index().groupby(column).first().reindex(symbol)["gene"]
            return gene_id
        if optional:
            symbol = pd.Series(symbol)[pd.Series(symbol).isin(self.var[column])]

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

    @classmethod
    def from_adata(
        cls,
        adata: sc.AnnData,
        path: Union[pathlib.Path, str] = None,
        overwrite=False,
    ):
        """
        Create a Transcriptome object from an AnnData object.

        Parameters:
            adata:
                Anndata object containing the transcriptome data.
            path:
                Folder in which the transcriptome data will be stored.
            overwrite:
                Whether to overwrite the data if it already exists.
        """

        transcriptome = cls(path=path, reset=overwrite)
        transcriptome.adata = adata

        for k, v in adata.layers.items():
            if sparse.is_scipysparse(v):
                v = np.array(v.todense())
            transcriptome.layers[k] = v.astype("<f4")
        if sparse.is_scipysparse(adata.X):
            v = np.array(adata.X.todense()).astype("<f4")
        else:
            v = adata.X.astype("<f4")
        transcriptome.layers["X"] = v
        transcriptome.var = adata.var
        transcriptome.obs = adata.obs
        return transcriptome

    @property
    def X(self):
        return self.layers[list(self.layers.keys())[0]]

    @X.setter
    def X(self, value):
        self.layers["X"] = value

    layers = StoredDict(Tensorstore, kwargs=dict(dtype="<f4"))
    "Dictionary of layers, such as raw, normalized and imputed data."

    def filter_genes(self, genes, path=None):
        """
        Filter genes

        Parameters:
            genes:
                Genes to filter.
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

    def filter_cells(self, cells, path=None):
        """
        Filter cells

        Parameters:
            cells:
                Cells to filter.
        """

        self.obs["ix"] = np.arange(self.obs.shape[0])
        cell_ixs = self.obs["ix"].loc[cells]

        layers = {}
        for k, v in self.layers.items():
            layers[k] = v[cell_ixs, :]
        X = self.X[cell_ixs, :]

        if self.o.adata.exists(self):
            adata = self.adata[cell_ixs, :]
        else:
            adata = None

        return Transcriptome.create(var=self.var, obs=self.obs.loc[cells], X=X, layers=layers, path=path, adata=adata)

    def get_X(self, gene_ids, layer=None):
        """
        Get the counts for a given set of genes.
        """
        gene_ixs = self.var.index.get_loc(gene_ids)

        if layer is None:
            value = self.X[:, gene_ixs]
        else:
            value = self.layers[layer][:, gene_ixs]

        if sparse.is_scipysparse(value):
            value = np.array(value.todense())
            if isinstance(gene_ids, str):
                value = value[:, 0]
        return value

    def get_diffexp(self, key = "rank_genes_groups", groups = None):
        return get_diffexp(self.adata, key, groups)


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

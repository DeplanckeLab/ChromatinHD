# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocess

# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import torch

import pickle

import scanpy as sc

import tqdm.auto as tqdm
import io

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"
main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k"
genome = "GRCh38"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Download

# %%
# ! mkdir -p {folder_data_preproc}

# %%
import pathlib

# %%
file = folder_data_preproc/"filtered_feature_bc_matrix.h5"
if not file.exists():
    # !wget {main_url}_filtered_feature_bc_matrix.h5 -O {file}

# %%
if pathlib.Path(chd.get_git_root().parent / "chromatinhd_manuscript" / "output/data" / dataset_name / "atac_fragments.tsv.gz").exists():
    # !rm {folder_data_preproc}/fragments.tsv.gz
    # !ln -s {chd.get_git_root().parent / "chromatinhd_manuscript" / "output/data" / dataset_name / "atac_fragments.tsv.gz"} {folder_data_preproc}/fragments.tsv.gz
    # !wget {main_url}_atac_fragments.tsv.gz.tbi -O {folder_data_preproc}/fragments.tsv.gz.tbi

# %% [markdown]
# ## Create transcriptome

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %% [markdown]
# ### Read and process

# %%
adata = sc.read_10x_h5(folder_data_preproc / "filtered_feature_bc_matrix.h5")

# %%
adata.var.index.name = "symbol"
adata.var = adata.var.reset_index()
adata.var.index = adata.var["gene_ids"]
adata.var.index.name = "gene"

# %%
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_counts=1000)
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_genes=200)
print(adata.obs.shape[0])
sc.pp.filter_genes(adata, min_cells=100)
print(adata.var.shape[0])

# %%
transcripts = chd.biomart.get_transcripts(chd.biomart.Dataset.from_genome(genome), gene_ids=adata.var.index.unique())
pickle.dump(transcripts, (folder_data_preproc / 'transcripts.pkl').open("wb"))

# %%
# only retain genes that have at least one ensembl transcript
adata = adata[:, adata.var.index.isin(transcripts["ensembl_gene_id"])]

# %%
sc.external.pp.scrublet(adata)

# %%
adata.obs["doublet_score"].plot(kind="hist")

# %%
adata.obs["doublet"] = (adata.obs["doublet_score"] > 0.1).astype("category")

print(adata.obs.shape[0])
adata = adata[~adata.obs["doublet"].astype(bool)]
print(adata.obs.shape[0])

# %%
size_factor = np.median(np.array(adata.X.sum(1)))
adata.uns["size_factor"] = size_factor

# %%
adata.raw = adata

# %%
sc.pp.normalize_total(adata, size_factor)

# %%
sc.pp.log1p(adata)

# %%
sc.pp.pca(adata)

# %%
sc.pp.highly_variable_genes(adata)

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
import pickle

# %%
adata.layers["normalized"] = adata.X
adata.layers["counts"] = adata.raw.X

# %%
import magic

magic_operator = magic.MAGIC(knn=30, solver = "approximate")
X_smoothened = magic_operator.fit_transform(adata.X)
adata.layers["magic"] = X_smoothened

# %%
pickle.dump(adata, (folder_data_preproc / 'adata.pkl').open("wb"))

# %% [markdown]
# ## Interpret and subset

# %%
adata = pickle.load((folder_data_preproc / 'adata.pkl').open("rb"))

# %%
sc.tl.leiden(adata, resolution=2.0)
sc.pl.umap(adata, color="leiden")

# %%
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon", key_added  = "wilcoxon")
# genes_oi = sc.get.rank_genes_groups_df(adata, group=None, key='rank_genes_groups').sort_values("scores", ascending = False).groupby("group").head(2)["names"]

# %%
sc.pl.rank_genes_groups_dotplot(adata, n_genes=100, key="wilcoxon", groupby="leiden", gene_symbols="symbol", groups = ["14"])

# %%
import io
marker_annotation = pd.read_table(
    io.StringIO(
        """ix	symbols	celltype
0	IL7R, CD3D	CD4 naive T
0	IL7R, CD3D, ITGB1	CD4 memory T
1	CD14, LYZ	CD14+ Monocytes
2	MS4A1, IL4R, CD79A	naive B
2	MS4A1, CD79A, TNFRSF13B	memory B
3	CD8A, CD3D	CD8 naive T
4	GNLY, NKG7, GZMA, GZMB, NCAM1	NK
4	GNLY, NKG7, CD3D, CCL5, GZMA, CD8A	CD8 activated T
4	SLC4A10	MAIT
5	FCGR3A, MS4A7	FCGR3A+ Monocytes
5	CD27, JCHAIN	Plasma
6	TCF4	pDCs
6	CST3, CD1C	cDCs
"""
    )
).set_index("celltype")
marker_annotation["symbols"] = marker_annotation["symbols"].str.split(", ")


# %%
# Define cluster score for all markers
def evaluate_partition(
    anndata, marker_dict, gene_symbol_key=None, partition_key="louvain_r1"
):
    # Inputs:
    #    anndata         - An AnnData object containing the data set and a partition
    #    marker_dict     - A dictionary with cell-type markers. The markers should be stores as anndata.var_names or
    #                      an anndata.var field with the key given by the gene_symbol_key input
    #    gene_symbol_key - The key for the anndata.var field with gene IDs or names that correspond to the marker
    #                      genes
    #    partition_key   - The key for the anndata.obs field where the cluster IDs are stored. The default is
    #                      'louvain_r1'

    # Test inputs
    if partition_key not in anndata.obs.columns.values:
        print("KeyError: The partition key was not found in the passed AnnData object.")
        print(
            "   Have you done the clustering? If so, please tell pass the cluster IDs with the AnnData object!"
        )
        raise

    if (gene_symbol_key != None) and (
        gene_symbol_key not in anndata.var.columns.values
    ):
        print(
            "KeyError: The provided gene symbol key was not found in the passed AnnData object."
        )
        print(
            "   Check that your cell type markers are given in a format that your anndata object knows!"
        )
        raise

    if gene_symbol_key:
        gene_ids = anndata.var[gene_symbol_key]
    else:
        gene_ids = anndata.var_names

    clusters = np.unique(anndata.obs[partition_key])
    n_clust = len(clusters)
    n_groups = len(marker_dict)

    marker_res = np.zeros((n_groups, n_clust))
    z_scores = sc.pp.scale(anndata, copy=True)

    i = 0
    for group in marker_dict:
        # Find the corresponding columns and get their mean expression in the cluster
        j = 0
        for clust in clusters:
            cluster_cells = np.in1d(z_scores.obs[partition_key], clust)
            marker_genes = np.in1d(gene_ids, marker_dict[group])
            marker_res[i, j] = z_scores.X[np.ix_(cluster_cells, marker_genes)].mean()
            j += 1
        i += 1

    variances = np.nanvar(marker_res, axis=0)
    if np.all(np.isnan(variances)):
        print(
            "No variances could be computed, check if your cell markers are in the data set."
        )
        print(
            "Maybe the cell marker IDs do not correspond to your gene_symbol_key input or the var_names"
        )
        raise

    marker_res_df = pd.DataFrame(marker_res, columns=clusters, index=marker_dict.keys())

    # Return the median of the variances over the clusters
    return marker_res_df



# %%
cluster_celltypes = evaluate_partition(
    adata, marker_annotation["symbols"].to_dict(), "symbol", partition_key="leiden"
).idxmax()

adata.obs["celltype"] = adata.obs["celltype"] = cluster_celltypes[
    adata.obs["leiden"]
].values
adata.obs["celltype"] = adata.obs["celltype"] = adata.obs[
    "celltype"
].astype(str)

# %%
adata.obs["log_n_counts"] = np.log(adata.obs["n_counts"])
sc.pl.umap(adata, color=["celltype", "log_n_counts", "leiden"], legend_loc="on data")

# %%
adata = adata[~adata.obs["leiden"].isin(["21", "23"])]

# %%
sc.pl.umap(adata, color=["celltype", "log_n_counts", "leiden"], legend_loc="on data")

# %%
pickle.dump(adata, (folder_data_preproc / 'adata_annotated.pkl').open("wb"))

# %% [markdown]
# ## TSS

# %%
adata = pickle.load((folder_data_preproc / 'adata_annotated.pkl').open("rb"))

# %%
transcripts = pickle.load((folder_data_preproc / 'transcripts.pkl').open("rb"))
transcripts = transcripts.loc[transcripts["ensembl_gene_id"].isin(adata.var.index)]

# %%
fragments_file = folder_data_preproc / "fragments.tsv.gz"
selected_transcripts = chd.data.regions.select_tss_from_fragments(transcripts, fragments_file)

# %%
plt.scatter(adata.var["means"], np.log(transcripts.groupby("ensembl_gene_id")["nfrags"].max()))

# %%
pickle.dump(selected_transcripts, (folder_data_preproc / 'selected_transcripts.pkl').open("wb"))

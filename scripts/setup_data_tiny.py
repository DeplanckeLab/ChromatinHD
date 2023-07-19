# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
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
import pathlib

# %%
import chromatinhd as chd

# %%
# install the genome if it doesn't exist yet
# this can take a couple of minutes
import genomepy
genomepy.install_genome("GRCh38")

# %%
folder_data = chd.get_git_root() / "data"
raw_folder = chd.get_git_root() / "tmp" / "pbmc10k"

dataset_name = "tiny"
main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k"
genome_name = "GRCh38"
organism = "hs"

folder_data_preproc = raw_folder
folder_data_preproc.mkdir(exist_ok=True, parents=True)

genome = genomepy.Genome("GRCh38")

# %% [markdown]
# ## Download

# %%
if not (folder_data_preproc / "filtered_feature_bc_matrix.h5").exists():
    ! wget {main_url}_filtered_feature_bc_matrix.h5 -O {folder_data_preproc}/filtered_feature_bc_matrix.h5

# %%
if not (folder_data_preproc / "atac_fragments.tsv.gz").exists():
    !wget {main_url}_atac_fragments.tsv.gz -O {folder_data_preproc}/atac_fragments.tsv.gz

# %%
if not (folder_data_preproc / "atac_fragments.tsv.gz.tbi").exists():
    !wget {main_url}_atac_fragments.tsv.gz.tbi -O {folder_data_preproc}/atac_fragments.tsv.gz.tbi

# %%
!ls -lh {folder_data_preproc}

# %% [markdown]
# ### Genes

# %% [markdown]
# Get the gene annotaton from ensembl biomart

# %%
biomart_dataset = chd.biomart.Dataset()
genes = biomart_dataset.get([
    biomart_dataset.attribute("ensembl_gene_id"),
    biomart_dataset.attribute("transcript_start"),
    biomart_dataset.attribute("transcript_end"),
    biomart_dataset.attribute("end_position"),
    biomart_dataset.attribute("start_position"),
    biomart_dataset.attribute("ensembl_transcript_id"),
    biomart_dataset.attribute("chromosome_name"),
    biomart_dataset.attribute("strand"),
    biomart_dataset.attribute("external_gene_name"),
    biomart_dataset.attribute("transcript_is_canonical"),
    biomart_dataset.attribute("transcript_biotype")
], filters = [
    biomart_dataset.filter("transcript_biotype", value = "protein_coding"),
])
genes["chrom"] = "chr" + genes["chromosome_name"].astype(str)
genes = genes.sort_values("transcript_is_canonical").groupby("ensembl_gene_id").first()
genes = genes.rename(
    columns = {
        "transcript_start": "start",
        "transcript_end": "end",
        "external_gene_name": "symbol",
    }
)
genes.index.name = "gene"
genes = genes.drop(["chromosome_name", "transcript_biotype", "transcript_is_canonical", "start_position", "end_position"], axis = 1)

# remove genes without a symbol
genes = genes.loc[~pd.isnull(genes["symbol"])]

# remove genes on alternative assemblies
genes = genes.loc[~genes["chrom"].str.contains("_")]

# %%
assert (
    genes.groupby(level=0).size().mean() == 1
), "For each gene, there should only be one transcript"

# %%
genes.to_csv(folder_data_preproc / "genes.csv")

# %% [markdown]
# ## Create transcriptome

# %% [markdown]
# Read in the transcriptome

# %%
genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col=0)
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %% [markdown]
# ### Read and process

# %%
adata = sc.read_10x_h5(folder_data_preproc / "filtered_feature_bc_matrix.h5")
adata.var.index.name = "symbol"
adata.var = adata.var.reset_index()
adata.var.index = adata.var["gene_ids"]
adata.var.index.name = "gene"

all_gene_ids = sorted(
    list(set(genes.loc[genes["chrom"] != "chrMT"].index) & set(adata.var.index))
)

adata = adata[:, all_gene_ids]

# %%
adata.var_names_make_unique()

# %%
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_counts=1000)
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_genes=200)
print(adata.obs.shape[0])
sc.pp.filter_genes(adata, min_cells=3)
print(adata.var.shape[0])

# %%
# you may have to install scrublet, `pip install scrublet`
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
sc.pp.normalize_total(adata, size_factor)

# %%
sc.pp.log1p(adata)

# %%
sc.pp.pca(adata)

# %%
sc.pp.highly_variable_genes(adata)

# %%
adata.var["n_cells"] = np.array((adata.X > 0).sum(0))[0]

# %%
# adata = adata[:, adata.var["dispersions_norm"].sort_values(ascending = False)[:5000].index]
print(adata.var.shape[0])

genes_oi = (
    adata.var.query("n_cells > 100")["dispersions_norm"]
    .sort_values(ascending=False)[:5000]
    .index.tolist()
)

adata = adata[:, genes_oi]
print(adata.var.shape[0])

all_gene_ids = adata.var.index

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
adata.var["chrom"] = genes["chrom"]

# %%
# may have to install leiden algorithm, `pip install leidenalg`
sc.tl.leiden(adata, resolution=1.0)

# %%
sc.pl.umap(adata, color="leiden", legend_loc="on data")

# %%
adata = adata[~adata.obs["leiden"].isin(["15", "16"])]

# %%
sc.pl.umap(adata, color="leiden")

# %%
transcriptome.adata = adata
transcriptome.var = adata.var
transcriptome.obs = adata.obs

# %%
transcriptome.layers["X"] = adata.X

# %%
fig, ax = plt.subplots()
sns.scatterplot(adata.var, x="means", y="dispersions_norm")
ax.set_xscale("log")
# %%
# may have to install magic, `pip install magic-impute`
import magic

magic_operator = magic.MAGIC(knn=30)
X_smoothened = magic_operator.fit_transform(adata.X)
adata.layers["magic"] = X_smoothened

# %%
transcriptome.layers["magic"] = X_smoothened

# %%
genes_oi = adata.var.sort_values("dispersions_norm", ascending=False).index[:10]
genes_oi = transcriptome.gene_id(["CCL4"])
sc.pl.umap(adata, color=genes_oi, title=transcriptome.symbol(genes_oi), layer="magic")

# %%
sc.pl.umap(adata, color="leiden", legend_loc="on data")

# %% [markdown]
# ### Interpret PBMC10K

# %%
sc.tl.rank_genes_groups(adata, "leiden", method="t-test")
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, gene_symbols = "symbol")

# %%
import io

# %%
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
# marker_annotation = marker_annotation.explode("symbols")

# %%
sc.pl.umap(
    adata,
    color=transcriptome.gene_id(
        marker_annotation.query("celltype == 'cDCs'")["symbols"].explode()
    ),
)
# %%
sc.pl.umap(
    adata,
    color=transcriptome.gene_id(
        "CD1C"
    ),
)


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

# %%
transcriptome.adata.obs["celltype"] = adata.obs["celltype"] = cluster_celltypes[
    adata.obs["leiden"]
].values
transcriptome.adata.obs["celltype"] = adata.obs["celltype"] = adata.obs[
    "celltype"
].astype(str)
# adata.obs.loc[adata.obs["leiden"] == "4", "celltype"] = "NKT"

# %%
transcriptome.adata.obs["log_n_counts"] = np.log(transcriptome.adata.obs["n_counts"])

# %%
sc.pl.umap(transcriptome.adata, color=["celltype", "log_n_counts", "leiden"])
sc.pl.umap(
    transcriptome.adata,
    color=transcriptome.gene_id(marker_annotation["symbols"].explode()),
    title=marker_annotation["symbols"].explode(),
)
# %%
transcriptome.obs = adata.obs
transcriptome.adata = adata

# %% [markdown]
# ## Create windows

# %% [markdown]
# ### Read genes and transcriptome

# %%
genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col=0)
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %% [markdown]
# ### Creating regions

# %%
region_name, region_window = "10k10k", (-10000, 10000)

# %%
all_gene_ids = transcriptome.var.index

# %%
regions = chd.data.Regions.from_genes(
    genes.loc[transcriptome.var.index],
    region_window,
    path = folder_data_preproc / "regions" / region_name
)

# %% [markdown]
# #### Create fragments
fragments = chd.data.Fragments.from_fragments_tsv(
    folder_data_preproc / "atac_fragments.tsv.gz",
    regions,
    transcriptome.obs,
    path = folder_data_preproc / "fragments" / regions.path.name,
)

# %%
fragments = chd.data.Fragments(folder_data_preproc / "fragments" / regions.path.name)

# %%
fragments.create_cellxgene_indptr()

# %%
fig, ax = plt.subplots()
bins = np.arange(0, 1000, 10)
sizes = fragments.coordinates[:, 1] - fragments.coordinates[:, 0]
bincount = np.bincount(np.digitize(sizes, bins))
ax.bar(bins, bincount[:-1], width=10, lw = 0)

# %% [markdown]
# ## Add MAGIC

# %%
transcriptome = chd.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
adata = transcriptome.adata

# %%
import magic

magic_operator = magic.MAGIC(knn=30)
X_smoothened = magic_operator.fit_transform(adata.X)

# %%
transcriptome.layers["magic"] = X_smoothened

# %% [markdown]
# ## Store small dataset

# %%
folder_dataset = chd.get_git_root() / "src" / "chromatinhd" / "data" / "examples" / "pbmc10ktiny"
folder_dataset.mkdir(exist_ok=True, parents=True)

# %%
transcriptome_original = chd.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments_original = chd.data.Fragments(folder_data_preproc / "fragments" / "10k10k")

# %%
genes_oi = transcriptome_original.var.sort_values("dispersions_norm", ascending=False).index[:50]
# genes_oi = transcriptome_original.var.loc[transcriptome_original.gene_id(["IL1B"])].index
adata = sc.AnnData(
    X = transcriptome_original.adata[:, genes_oi].layers["magic"],
    obs = transcriptome_original.obs,
    var = transcriptome_original.var.loc[genes_oi]
)


# %%
adata.write(folder_dataset / "transcriptome.h5ad", compression="gzip")

# %%
import pysam
fragments_tabix = pysam.TabixFile(str(folder_data_preproc / "atac_fragments.tsv.gz"))

coordinates = fragments_original.regions.coordinates.loc[adata.var.index]
# %%
fragments_new = []
for i, (gene, promoter_info) in tqdm.tqdm(
    enumerate(coordinates.iterrows()), total=coordinates.shape[0]
):
    start = max(0, promoter_info["start"])
    
    fragments_promoter = fragments_tabix.fetch(promoter_info["chrom"], start, promoter_info["end"])
    fragments_new.extend(list(fragments_promoter))

fragments_new = pd.DataFrame([x.split("\t") for x in fragments_new], columns = ["chrom", "start", "end", "cell", "nreads"])
fragments_new["start"] = fragments_new["start"].astype(int)
fragments_new = fragments_new.sort_values(["chrom", "start", "cell"])

# %%
import gzip
fragments_new.to_csv(folder_dataset / "fragments.tsv", sep="\t", index=False, header = False)

# %%
import pysam
pysam.tabix_compress(folder_dataset / "fragments.tsv", folder_dataset / "fragments.tsv.gz", force=True)
# %%
!ls -lh {folder_dataset}
# %%
!tabix -p bed {folder_dataset / "fragments.tsv.gz"}
# %%
folder_dataset
# %%


# %% [markdown]
# ## Try to create dataset

# %%
gene_ix = transcriptome_original.gene_ix("IL1B")
transcriptome_original.obs["n_fragments"] = torch.bincount(fragments_original.mapping[fragments_original.mapping[:, 1] == gene_ix, 0], minlength = transcriptome_original.obs.shape[0])
transcriptome_original.obs["expression"] = transcriptome_original.layers["magic"][:, gene_ix]

# %%
fig, ax = plt.subplots()
sns.scatterplot(transcriptome_original.obs, x="n_fragments", y="expression", alpha = 0.1, s = 1, ax = ax)
# %%
import pathlib
import shutil

dataset_folder = pathlib.Path("/tmp/chromatinhd/pbmc10ktiny")
dataset_folder.mkdir(exist_ok=True, parents=True)

for file in dataset_folder.iterdir():
    if file.is_file():
        file.unlink()
    else:
        shutil.rmtree(file)
# %%
import pkg_resources

DATA_PATH = pathlib.Path(
    pkg_resources.resource_filename("chromatinhd", "data/examples/pbmc10ktiny/")
)

# copy all files from data path to dataset folder
for file in DATA_PATH.iterdir():
    shutil.copy(file, dataset_folder / file.name)
# %%
!ls {dataset_folder}
# %%
import scanpy as sc

adata = sc.read(dataset_folder / "transcriptome.h5ad")
# %%
transcriptome = chd.data.Transcriptome.from_adata(
    adata, path=dataset_folder / "transcriptome"
)

# %%
biomart_dataset = chd.biomart.Dataset.from_genome("GRCh38")
canonical_transcripts = chd.biomart.get_canonical_transcripts(
    biomart_dataset, transcriptome.var.index
)
# %%
regions = chd.data.Regions.from_canonical_transcripts(
    canonical_transcripts,
    path=dataset_folder / "regions",
    window=[-10000, 10000],
)
# %%
if not (dataset_folder / "fragments.tsv.gz.tbi").exists():
    !tabix {dataset_folder}/fragments.tsv.gz
# %%
fragments = chd.data.Fragments.from_fragments_tsv(
    dataset_folder / "fragments.tsv.gz",
    regions,
    obs = transcriptome.obs,
    path=dataset_folder / "fragments",
)
# %%
fragments.mapping.shape

# %%
gene_ix = transcriptome.gene_ix("IL1B")
transcriptome.obs["n_fragments"] = torch.bincount(fragments.mapping[fragments.mapping[:, 1] == gene_ix, 0], minlength = transcriptome.obs.shape[0])
transcriptome.obs["expression"] = transcriptome.layers["X"][:, gene_ix]

# %%
fig, ax = plt.subplots()
sns.regplot(transcriptome.obs, x="n_fragments", y="expression", ax = ax)
# %%

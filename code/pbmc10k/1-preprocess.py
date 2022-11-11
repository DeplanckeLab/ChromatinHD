# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocess

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')

import torch

import pickle

import scanpy as sc

import tqdm.auto as tqdm

# %%
import peakfreeatac as pfa

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"
folder_data_preproc = folder_data / "pbmc10k"
folder_data_preproc.mkdir(exist_ok = True, parents = True)

# %%
import torch

# %% [markdown]
# ## Download

# %% [markdown]
# https://support.10xgenomics.com/single-cell-atac/software/pipelines/latest/algorithms/overview

# %%
main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k"

# %%
# ! wget {main_url}_filtered_feature_bc_matrix.h5 -O {folder_data_preproc}/filtered_feature_bc_matrix.h5

# %%
# !wget {main_url}_atac_fragments.tsv.gz -O {folder_data_preproc}/atac_fragments.tsv.gz

# %%
# !wget {main_url}_atac_fragments.tsv.gz.tbi -O {folder_data_preproc}/atac_fragments.tsv.gz.tbi

# %%
# !wget {main_url}_atac_peaks.bed -O {folder_data_preproc}/atac_peaks.bed

# %%
# !cat {folder_data_preproc}/atac_peaks.bed | sed '/^#/d' > {folder_data_preproc}/peaks.tsv

# %%
# !wget {main_url}_atac_peak_annotation.tsv -O {folder_data_preproc}/peak_annot.tsv

# %%
# !wget {main_url}_atac_cut_sites.bigwig -O {folder_data_preproc}/atac_cut_sites.bigwig

# %%
# !wget http://ftp.ensembl.org/pub/release-107/gff3/homo_sapiens/Homo_sapiens.GRCh38.107.gff3.gz -O {folder_data_preproc}/genes.gff.gz

# %%
# !zcat {folder_data_preproc}/genes.gff.gz | grep -vE "^#" | awk '$3 == "gene"' > {folder_data_preproc}/genes.gff

# %%
# !wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes -O  {folder_data_preproc}/chromosome.sizes

# %%
gff = pd.read_table(folder_data_preproc / "genes.gff", sep = "\t", names = ["chr", "type", "__", "start", "end", "dot", "strand", "dot2", "info"])
genes = gff.copy()
genes["chr"] = "chr" + genes["chr"]
genes["symbol"] = genes["info"].str.split(";").str[1].str[5:]
genes["gene"] = genes["info"].str.split(";").str[0].str[8:]
genes = genes.set_index("gene", drop = False)

# %%
import pybedtools

# %%
genes.query("symbol == 'PCNA'")

# %% [markdown]
# ## Create transcriptome

# %%
import peakfreeatac.transcriptome

# %%
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc)

# %%
adata = sc.read_10x_h5(transcriptome.path / "filtered_feature_bc_matrix.h5")

# %%
chromosomes = ["chr" + str(i) for i in range(23)] + ["chrX", "chrY"]

# %%
adata.var.index.name = "symbol"
adata.var = adata.var.reset_index()
adata.var.index = adata.var["gene_ids"]
adata.var.index.name = "gene"

all_gene_ids = sorted(list(set(genes.loc[genes["chr"].isin(chromosomes)]["gene"]) & set(adata.var.index)))

adata = adata[:, all_gene_ids]

# %%
adata.var_names_make_unique()

# %%
sc.pp.normalize_per_cell(adata)

# %%
sc.pp.log1p(adata)

# %%
sc.pp.pca(adata)

# %%
sc.pp.highly_variable_genes(adata)

# %%
adata = adata[:, adata.var["dispersions_norm"].sort_values(ascending = False)[:5000].index]

all_gene_ids = adata.var.index

# %%
sc.pp.neighbors(adata)

# %%
sc.tl.umap(adata)

# %%
transcriptome.adata = adata
transcriptome.var = adata.var
transcriptome.obs = adata.obs

# %%
sc.pl.umap(adata, color = transcriptome.gene_id(["ANPEP", "FCGR3A"]))

# %%
adata.var.query("dispersions_norm > 0.5").index.to_series().to_json((folder_data_preproc/"variable_genes.json").open("w"))

# %% [markdown]
# ## Create windows

# %% [markdown]
# ### Create windows around genes

# %%
genewindows = genes.copy()
genewindows["start"] = np.maximum(genewindows["start"] - 20000, 0)
genewindows["end"] = genewindows["end"] + 20000

# %%
genewindows_bed = pybedtools.BedTool.from_dataframe(genewindows[["chr", "start", "end", "strand", "gene"]])

# %%
peaks = pybedtools.BedTool(folder_data_preproc/"atac_peaks.bed")

# %%
intersect = genewindows_bed.intersect(peaks, wo = True)
intersect = intersect.to_dataframe()

# %%
intersect["peak_id"] = intersect["strand"] + ":" + intersect["thickStart"].astype(str) + "-" + intersect["thickEnd"].astype(str)

# %%
gene_peak_links = intersect[["score", "peak_id"]].copy()
gene_peak_links.columns = ["gene", "peak"]

# %%
gene_peak_links.to_csv(folder_data_preproc / "gene_peak_links_20k.csv")

# %% [markdown]
# ### Creating promoters

# %%
import tabix

# %%
fragments_tabix = tabix.open(str(folder_data_preproc / "atac_fragments.tsv.gz"))

# %% [markdown]
# #### Define promoters

# %%
import pybedtools

# %%
all_gene_ids = transcriptome.var.index

# %%
promoters = pd.DataFrame(index = all_gene_ids)

# %%
promoters["tss"] = [genes_row["start"] if genes_row["strand"] == "+" else genes_row["end"] for _, genes_row in genes.loc[promoters.index].iterrows()]
promoters["strand"] = (genes["strand"] + "1").astype(int)
promoters["positive_strand"] = (promoters["strand"] == 1).astype(int)
promoters["negative_strand"] = (promoters["strand"] == -1).astype(int)
promoters["chr"] = genes.loc[promoters.index, "chr"]

# %%
padding_positive = 2000
padding_negative = 4000
promoters["start"] = promoters["tss"] - padding_negative * promoters["positive_strand"] - padding_positive * promoters["negative_strand"]
promoters["end"] = promoters["tss"] + padding_negative * promoters["negative_strand"] + padding_positive * promoters["positive_strand"]

# %%
promoters = promoters.drop(columns = ["positive_strand", "negative_strand"], errors = "ignore")

# %%
promoters

# %%
promoters.to_csv(folder_data_preproc / "promoters.csv")

# %% [markdown]
# #### Create fragments

# %%
var = pd.DataFrame(index = promoters.index)
var["ix"] = np.arange(var.shape[0])

n_genes = var.shape[0]

# %%
obs = transcriptome.adata.obs[[]].copy()
obs.index.name = "cell"
obs["ix"] = np.arange(obs.shape[0])

cell_to_cell_ix = obs["ix"].to_dict()

n_cells = obs.shape[0]

# %%
gene_to_fragments = [[] for i in var["ix"]]
cell_to_fragments = [[] for i in obs["ix"]]

# %%
fragments_raw = []
fragment_mappings_raw = []

for i, (gene, promoter_info) in tqdm.tqdm(enumerate(promoters.iterrows())):
    gene_ix = var.loc[gene, "ix"]
    fragments_promoter = fragments_tabix.query(*promoter_info[["chr", "start", "end"]])
    
    for fragment in fragments_promoter:
        cell = fragment[3]
        
        # only store the fragment if the cell is actually of interest
        if cell in cell_to_cell_ix:
            # add raw data of fragment relative to tss
            fragments_raw.append([
                (int(fragment[1]) - promoter_info["tss"]) * promoter_info["strand"],
                (int(fragment[2]) - promoter_info["tss"]) * promoter_info["strand"]
            ][::promoter_info["strand"]])

            # add mapping of cell/gene
            fragment_mappings_raw.append([
                cell_to_cell_ix[fragment[3]],
                gene_ix
            ])

# %%
import torch

# %%
import pathlib
import peakfreeatac.fragments
fragments = pfa.fragments.Fragments(pathlib.Path("./"))

# %%
fragments.var = var
fragments.obs = obs

# %% [markdown]
# Create fragments tensor

# %%
# we use int32 for smaller memory usage
# will have to be converted to floats anyway...
coordinates = torch.tensor(np.array(fragments_raw, dtype = np.int32))

# %%
# int64 is needed for torch_scatter
mapping = torch.tensor(np.array(fragment_mappings_raw), dtype = torch.int64)

# %% [markdown]
# Sort `coordinates` and `mapping` according to `mapping`

# %%
sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
mapping = mapping[sorted_idx]
coordinates = coordinates[sorted_idx]

# %% [markdown]
# Check size and dump

# %%
np.product(mapping.size()) * 64 / 8 / 1024 / 1024

# %%
np.product(coordinates.size()) * 32 / 8 / 1024 / 1024

# %%
fragments.mapping = mapping
fragments.coordinates = coordinates

# %% [markdown]
# ## Fragment distribution

# %%
import gzip

# %%
sizes = []
with gzip.GzipFile(folder_data_preproc / "atac_fragments.tsv.gz", "r") as fragment_file:
    i = 0
    for line in fragment_file:
        line = line.decode("utf-8")
        if line.startswith("#"):
            continue
        split = line.split("\t")
        sizes.append(int(split[2]) - int(split[1]))
        i += 1
        if i > 1000000:
            break

# %%
sizes = np.array(sizes)

# %%
np.isnan(sizes).sum()

# %%
fig, ax = plt.subplots()
ax.hist(sizes, range = (0, 1000), bins = 100)
ax.set_xlim(0, 1000)

# %%

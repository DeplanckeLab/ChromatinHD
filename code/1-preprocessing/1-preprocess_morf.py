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

# dataset_name = "FLI1_7"; organism = "hs"; genome = "GRCh38.107"
# dataset_name = "PAX2_7"; organism = "hs"; genome = "GRCh38.107"
# dataset_name = "NHLH1_7"; organism = "hs"; genome = "GRCh38.107"
# dataset_name = "CDX2_7"; organism = "hs"; genome = "GRCh38.107"
dataset_name = "CDX1_7"; organism = "hs"; genome = "GRCh38.107"
# dataset_name = "MSGN1_7"; organism = "hs"; genome = "GRCh38.107"
# dataset_name = "KLF4_7"; organism = "hs"; genome = "GRCh38.107"
# dataset_name = "KLF5_7"; organism = "hs"; genome = "GRCh38.107"
# dataset_name = "PTF1A_4"; organism = "hs"; genome = "GRCh38.107"
# dataset_name = "morf_20"; organism = "hs"; genome = "GRCh38.107"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

if organism == "mm":
    chromosomes = ["chr" + str(i) for i in range(20)] + ["chrX", "chrY"]
elif organism == "hs":
    chromosomes = ["chr" + str(i) for i in range(24)] + [
        "chrX", 
        # "chrY" # chromosome Y was removed for some reason in the original paper
    ]

# %% [markdown]
# ## Download

# %%
import pathlib

# %%
folder_dataset_remote = pathlib.Path("/home/wsaelens/projects/latenta_manuscript/data/individual/") / dataset_name

# %%
# !rsync -av wsaelens@updeplasrv6.epfl.ch:{folder_dataset_remote}/* {folder_data_preproc}/

# %%
# !zcat {folder_data_preproc}/fragments.tsv.gz | head -n 5

# %%
# to reuse from lymphoma
# !ln -s {folder_data_preproc}/../lymphoma/chromosome.sizes {folder_data_preproc}/chromosome.sizes
# !ln -s {folder_data_preproc}/../lymphoma/genes.gff.gz {folder_data_preproc}/genes.gff.gz
# !ln -s {folder_data_preproc}/../lymphoma/dna.fa.gz {folder_data_preproc}/dna.fa.gz
# !ln -s {folder_data_preproc}/../lymphoma/genome.pkl.gz {folder_data_preproc}/genome.pkl.gz

# %% [markdown]
# ### Genes

# %%
biomart_dataset_name = "hsapiens_gene_ensembl"

# %%
query = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query  virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >

    <Dataset name = "{biomart_dataset_name}" interface = "default" >
        <Filter name = "transcript_is_canonical" excluded = "0"/>
        <Filter name = "transcript_biotype" value = "protein_coding"/>
        <Attribute name = "ensembl_gene_id" />
        <Attribute name = "transcript_start" />
        <Attribute name = "transcript_end" />
        <Attribute name = "end_position" />
        <Attribute name = "start_position" />
        <Attribute name = "ensembl_transcript_id" />
        <Attribute name = "chromosome_name" />
        <Attribute name = "strand" />
        <Attribute name = "external_gene_name" />
    </Dataset>
</Query>"""
url = "http://www.ensembl.org/biomart/martservice?query=" + query.replace("\t", "").replace("\n", "")
from io import StringIO
import requests
session = requests.Session()
session.headers.update({'User-Agent': 'Custom user agent'})
r = session.get(url)
result = pd.read_table(StringIO(r.content.decode("utf-8")))

# %%
genes = result.rename(columns = {
    "Gene stable ID":"gene",
    "Transcript start (bp)":"start",
    "Transcript end (bp)":"end",
    "Chromosome/scaffold name":"chr",
    "Gene name":"symbol",
    "Strand":"strand"
})
genes["chr"] = "chr" + genes["chr"].astype(str)
genes = genes.groupby("gene").first()

# %%
genes = genes.loc[genes["chr"].isin(chromosomes)]

# %%
assert genes.groupby(level = 0).size().mean() == 1, "For each gene, there should only be one transcript"

# %%
genes.to_csv(folder_data_preproc / "genes.csv")

# %% [markdown]
# ## Create transcriptome

# %%
genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col = 0)

# %%
import peakfreeatac.data

# %%
adata = pickle.load((folder_data_preproc / "adata.pkl").open("rb"))

# %%
transcriptome = peakfreeatac.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
adata.var["symbol"] = adata.var.index

# %% [markdown]
# ### Read and process

# %%
matched_genes = adata.var.index.intersection(genes["symbol"])
len(matched_genes)
len(matched_genes) / len(adata.var.index)

# %%
adata = adata[:, matched_genes]
adata.var.index = genes.reset_index().set_index("symbol").loc[matched_genes].groupby(level = 0).first()["gene"].values
adata.var = adata.var.copy()

# %%
len(matched_genes)

# %%
cols = adata.var.columns.str.endswith("pvals_adj")

# %%
all_gene_ids = sorted(list(set(genes.loc[genes["chr"].isin(chromosomes)].index) & set(adata.var.index)))
# all_gene_ids = adata.var.loc[all_gene_ids].sort_values("pvals_adj").iloc[:1000].index
# all_gene_ids = adata.var.loc[all_gene_ids].query("differential").index
all_gene_ids = adata.var.loc[all_gene_ids, cols].min(1).sort_values().iloc[:1000].index

adata = adata[:, all_gene_ids].copy()

# %%
sc.pp.highly_variable_genes(adata)

# %%
adata.var["chr"] = genes["chr"]

# %%
transcriptome.adata = adata
transcriptome.var = adata.var
transcriptome.obs = adata.obs

# %%
transcriptome.create_X()

# %%
fig, ax = plt.subplots()
sns.scatterplot(
    adata.var,
    x = "means", y = "dispersions_norm"
)
ax.set_xscale("log")

# %%
# adata.var['mt'] = adata.var["symbol"].str.startswith('MT')  # annotate the group of mitochondrial genes as 'mt'
# sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# %%
genes_oi = adata.var.sort_values("dispersions_norm", ascending = False).index[:10]
# genes_oi = transcriptome.gene_id(["LEF1"])
sc.pl.umap(adata, color=genes_oi, title = transcriptome.symbol(genes_oi))

# %%
adata.var.query("dispersions_norm > 0.5").index.to_series().to_json((transcriptome.path / "variable_genes.json").open("w"))

# %% [markdown]
# ## Create windows

# %% [markdown]
# ### Creating promoters

# %%
import tabix
import pysam

# %%
fragments_tabix = tabix.open(str(folder_data_preproc / "fragments.tsv.gz"))

# %%
if not (folder_data_preproc / "fragments.tsv.gz.tbi").exists():
    pysam.tabix_index(str(folder_data_preproc / "fragments.tsv.gz"), seq_col = 0, start_col = 1, end_col = 2, force = True)

# %%
# zippedf =folder_data_preproc / "fragments.tsv.gz"
# def tabix_index(zippedf):
#     from subprocess import Popen,PIPE
#     import shlex
#     p = Popen(['tabix','-f', zippedf], stdout= PIPE)
#     # or : cmd = "tabix -f " + zippedf
#     # p = Popen(shlex.split(cmd), stdout=PIPE) 
#     #(shlex splits the cmd in spaces)
#     p.wait()

# %%
# tabix_index(zippedf)

# %% [markdown]
# #### Define promoters

# %%
promoter_name, (padding_negative, padding_positive) = "4k2k", (2000, 4000)
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)
# promoter_name, (padding_negative, padding_positive) = "20kpromoter", (10000, 0)
# promoter_name, (padding_negative, padding_positive) = "1k1k", (1000, 1000)

# %%
import pybedtools

# %%
all_gene_ids = transcriptome.var.index

# %%
promoters = pd.DataFrame(index = all_gene_ids)

# %%
promoters["tss"] = [genes_row["start"] if genes_row["strand"] == "+" else genes_row["end"] for _, genes_row in genes.loc[promoters.index].iterrows()]
promoters["strand"] = genes["strand"].astype(int)
promoters["positive_strand"] = (promoters["strand"] == 1).astype(int)
promoters["negative_strand"] = (promoters["strand"] == -1).astype(int)
promoters["chr"] = genes.loc[promoters.index, "chr"]

# %%
promoters["start"] = promoters["tss"] - padding_negative * promoters["positive_strand"] - padding_positive * promoters["negative_strand"]
promoters["end"] = promoters["tss"] + padding_negative * promoters["negative_strand"] + padding_positive * promoters["positive_strand"]

# %%
promoters = promoters.drop(columns = ["positive_strand", "negative_strand"], errors = "ignore")

# %%
promoters.to_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"))

# %% [markdown]
# #### Create fragments

# %%
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
transcriptome = peakfreeatac.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
import pathlib
import peakfreeatac.data
fragments = pfa.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
var = pd.DataFrame(index = promoters.index)
var["ix"] = np.arange(var.shape[0])

n_genes = var.shape[0]

# %%
obs = transcriptome.adata.obs[[]].copy()
obs.index.name = "cell"
obs["ix"] = np.arange(obs.shape[0])

cell_to_cell_ix = pd.Series(obs["ix"].values, obs["ix"].values.astype(str)).to_dict()

n_cells = obs.shape[0]

# %%
gene_to_fragments = [[] for i in var["ix"]]
cell_to_fragments = [[] for i in obs["ix"]]

# %%
coordinates_raw = []
mapping_raw = []

for i, (gene, promoter_info) in tqdm.tqdm(enumerate(promoters.iterrows()), total = promoters.shape[0]):
    gene_ix = var.loc[gene, "ix"]
    fragments_promoter = fragments_tabix.query(*promoter_info[["chr", "start", "end"]])
    
    for fragment in fragments_promoter:
        cell = fragment[3]
        
        # only store the fragment if the cell is actually of interest
        if cell in cell_to_cell_ix:
            # add raw data of fragment relative to tss
            coordinates_raw.append([
                (int(fragment[1]) - promoter_info["tss"]) * promoter_info["strand"],
                (int(fragment[2]) - promoter_info["tss"]) * promoter_info["strand"]
            ][::promoter_info["strand"]])

            # add mapping of cell/gene
            mapping_raw.append([
                cell_to_cell_ix[fragment[3]],
                gene_ix
            ])

# %%
fragments.var = var
fragments.obs = obs

# %% [markdown]
# Create fragments tensor

# %%
coordinates = torch.tensor(np.array(coordinates_raw, dtype = np.int64))
mapping = torch.tensor(np.array(mapping_raw), dtype = torch.int64)

# %% [markdown]
# Sort `coordinates` and `mapping` according to `mapping`

# %%
sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
mapping = mapping[sorted_idx]
coordinates = coordinates[sorted_idx]

# %% [markdown]
# Check size

# %%
np.product(mapping.size()) * 64 / 8 / 1024 / 1024

# %%
np.product(coordinates.size()) * 64 / 8 / 1024 / 1024

# %% [markdown]
# Store

# %%
fragments.mapping = mapping
fragments.coordinates = coordinates

# %% [markdown]
# Create cellxgene index pointers

# %%
fragments.create_cellxgene_indptr()

# %% [markdown]
# #### Create training folds

# %%
n_bins = 5

# %%
# train/test split
cells_all = np.arange(fragments.n_cells)
genes_all = np.arange(fragments.n_genes)

cell_bins = np.floor((np.arange(len(cells_all))/(len(cells_all)/n_bins)))

chromosome_gene_counts = transcriptome.var.groupby("chr").size().sort_values(ascending = False)
chromosome_bins = np.cumsum(((np.cumsum(chromosome_gene_counts) % (chromosome_gene_counts.sum() / n_bins + 1)).diff() < 0))

gene_bins = chromosome_bins[transcriptome.var["chr"]].values

n_folds = 5
folds = []
for i in range(n_folds):
    cells_train = cells_all[cell_bins != i]
    cells_validation = cells_all[cell_bins == i]

    chromosomes_train = chromosome_bins.index[~(chromosome_bins == i)]
    chromosomes_validation = chromosome_bins.index[chromosome_bins == i]
    genes_train = fragments.var["ix"][transcriptome.var.index[transcriptome.var["chr"].isin(chromosomes_train)]].values
    genes_validation = fragments.var["ix"][transcriptome.var.index[transcriptome.var["chr"].isin(chromosomes_validation)]].values
    
    folds.append({
        "cells_train":cells_train,
        "cells_validation":cells_validation,
        "genes_train":genes_train,
        "genes_validation":genes_validation
    })
pickle.dump(folds, (fragments.path / "folds.pkl").open("wb"))

# %% [markdown]
# ## Fragment distribution

# %%
import gzip

# %%
sizes = []
with gzip.GzipFile(folder_data_preproc / "fragments.tsv.gz", "r") as fragment_file:
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

# %%

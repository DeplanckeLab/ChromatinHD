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

dataset_name = "pbmc10k"; main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k"; genome = "GRCh38.107"; organism = "hs"
# dataset_name = "pbmc3k"; main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k"; genome = "GRCh38.107"; organism = "hs"
# dataset_name = "lymphoma"; main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/lymph_node_lymphoma_14k/lymph_node_lymphoma_14k"; genome = "GRCh38.107"; organism = "hs"
# dataset_name = "e18brain"; main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k";  genome = "mm10"; organism = "mm"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

if organism == "mm":
    chromosomes = ["chr" + str(i) for i in range(20)] + ["chrX", "chrY"]
elif organism == "hs":
    chromosomes = ["chr" + str(i) for i in range(24)] + ["chrX", "chrY"]

# %% [markdown]
# ## Download

# %% [markdown]
# For an overview on the output data format, see:
# https://support.10xgenomics.com/single-cell-atac/software/pipelines/latest/algorithms/overview

# %%
# ! echo mkdir {folder_data_preproc}
# ! echo mkdir {folder_data_preproc}/bam

# %%
# # ! wget {main_url}_atac_possorted_bam.bam -O {folder_data_preproc}/bam/atac_possorted_bam.bam
# # ! wget {main_url}_atac_possorted_bam.bam.bai -O {folder_data_preproc}/bam/atac_possorted_bam.bam.bai
# ! echo wget {main_url}_atac_possorted_bam.bam -O {folder_data_preproc}/bam/atac_possorted_bam.bam
# ! echo wget {main_url}_atac_possorted_bam.bam.bai -O {folder_data_preproc}/bam/atac_possorted_bam.bam.bai

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
# # !wget http://ftp.ensembl.org/pub/release-107/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_rm.primary_assembly.fa.gz -O {folder_data_preproc}/dna.fa.gz

# %%
if genome == "GRCh38.107":
    # !wget http://ftp.ensembl.org/pub/release-107/gff3/homo_sapiens/Homo_sapiens.GRCh38.107.gff3.gz -O {folder_data_preproc}/genes.gff.gz
    # !wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes -O  {folder_data_preproc}/chromosome.sizes
    # # !wget http://ftp.ensembl.org/pub/release-107/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz -O {folder_data_preproc}/dna.fa.gz
elif genome == "mm10":
    # !wget http://ftp.ensembl.org/pub/release-98/gff3/mus_musculus/Mus_musculus.GRCm38.98.gff3.gz -O {folder_data_preproc}/genes.gff.gz
    # !wget http://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes -O  {folder_data_preproc}/chromosome.sizes
    # # !wget http://ftp.ensembl.org/pub/release-98/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna_sm.toplevel.fa.gz -O {folder_data_preproc}/dna.fa.gz

# %%
# !zcat {folder_data_preproc}/genes.gff.gz | grep -vE "^#" | awk '$3 == "gene"' > {folder_data_preproc}/genes.gff

# %% [markdown]
# ### Genome

# %%
from operator import xor
y = lambda bp: 0b11 & xor((ord(bp) >> 2), (ord(bp) >> 1))

# %%
# >>> from operator import xor
# >>> y = lambda bp: 0b11 & xor((ord(bp) >> 2), (ord(bp) >> 1))

# %%
import gzip
genome = {}
chromosome = None
translate_table = {"A":0, "C":1, "G":2, "T":3, "N":4} # alphabetic order
for i, line in enumerate(gzip.GzipFile(folder_data_preproc / "dna.fa.gz")):
    line = str(line,'utf-8')
    if line.startswith(">"):
        if chromosome is not None:
            genome[chromosome] = np.array(genome_chromosome, dtype = np.int8)
        chromosome = "chr" + line[1:line.find(" ")]
        genome_chromosome = []
        
        print(chromosome)
        
        if chromosome not in chromosomes:
            break
    else:
        genome_chromosome += [translate_table[x] for x in line.strip("\n").upper()]

# %%
# !ln -s {folder_data_preproc}/../pbmc10k/genome.pkl.gz {folder_data_preproc}/

pickle.dump(genome, gzip.GzipFile((folder_data_preproc / "genome.pkl.gz"), "wb", compresslevel = 3))

# %%
# !ls -lh {folder_data_preproc}

# %% [markdown]
# ## Create genes

# %%
gff = pd.read_table(folder_data_preproc / "genes.gff", sep = "\t", names = ["chr", "type", "__", "start", "end", "dot", "strand", "dot2", "info"])
genes = gff.copy()
genes["chr"] = "chr" + genes["chr"]
genes["symbol"] = genes["info"].str.split(";").str[1].str[5:]
genes["gene"] = genes["info"].str.split(";").str[0].str[8:]
genes = genes.set_index("gene", drop = False)

# %%
genes.to_csv(folder_data_preproc / "genes.csv")

# %%
import pybedtools

# %%
genes.query("symbol == 'PCNA'")

# %%
# print(pd.read_table(folder_data_preproc / "peak_annot.tsv").query("gene == 'Neurod1'"))
# print(genes.query("symbol == 'Neurod1'"))

# %% [markdown]
# ## Create transcriptome

# %%
import peakfreeatac.transcriptome

# %%
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")

# %% [markdown]
# ### Read and process

# %%
adata = sc.read_10x_h5(folder_data_preproc / "filtered_feature_bc_matrix.h5")

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
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_counts = 1000)
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_genes= 200)
print(adata.obs.shape[0])
sc.pp.filter_genes(adata, min_cells=3)
print(adata.var.shape[0])

# %%
sc.external.pp.scrublet(adata)

# %%
adata.obs["doublet_score"].plot(kind = "hist")

# %%
adata.obs["doublet"] = (adata.obs["doublet_score"] > 0.1).astype("category")

print(adata.obs.shape[0])
adata = adata[~adata.obs["doublet"].astype(bool)]
print(adata.obs.shape[0])

# %%
sc.pp.normalize_per_cell(adata)

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
adata = adata[:, adata.var.query("n_cells > 100")["dispersions_norm"].sort_values(ascending = False)[:5000].index]
print(adata.var.shape[0])

all_gene_ids = adata.var.index

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
transcriptome.adata.var["chr"] = genes["chr"]

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
sc.pl.umap(adata, color=genes_oi, title = transcriptome.symbol(genes_oi))

# %%
# genes_oi = transcriptome.gene_id(["LEF1"])
sc.pl.umap(adata, color=genes_oi, title = transcriptome.symbol(genes_oi))

# %%
adata.var.query("dispersions_norm > 0.5").index.to_series().to_json((transcriptome.path / "variable_genes.json").open("w"))

# %% [markdown]
# ### Interpret PBMC10K

# %%
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, gene_symbols = "symbol")

# %%
import io

# %%
marker_annotation = pd.read_table(io.StringIO("""ix	symbols	celltype
0	IL7R, CD3D	CD4 T cells
1	CD14, LYZ	CD14+ Monocytes
2	MS4A1	B cells
3	CD8A, CD3D	CD8 T cells
4	GNLY, NKG7	NK cells
4	GNLY, NKG7, CD3D, CCL5	 NKT cells
5	FCGR3A, MS4A7	FCGR3A+ Monocytes
5	IGLC2, CD27	Plasma cells
6	TCF4	pDCs
6	FCER1A, CST3	cDCs
""")).set_index("celltype")
marker_annotation["symbols"] = marker_annotation["symbols"].str.split(", ")
# marker_annotation = marker_annotation.explode("symbols")

# %%
#Define cluster score for all markers
def evaluate_partition(anndata, marker_dict, gene_symbol_key=None, partition_key='louvain_r1'):
    # Inputs:
    #    anndata         - An AnnData object containing the data set and a partition
    #    marker_dict     - A dictionary with cell-type markers. The markers should be stores as anndata.var_names or 
    #                      an anndata.var field with the key given by the gene_symbol_key input
    #    gene_symbol_key - The key for the anndata.var field with gene IDs or names that correspond to the marker 
    #                      genes
    #    partition_key   - The key for the anndata.obs field where the cluster IDs are stored. The default is
    #                      'louvain_r1' 

    #Test inputs
    if partition_key not in anndata.obs.columns.values:
        print('KeyError: The partition key was not found in the passed AnnData object.')
        print('   Have you done the clustering? If so, please tell pass the cluster IDs with the AnnData object!')
        raise

    if (gene_symbol_key != None) and (gene_symbol_key not in anndata.var.columns.values):
        print('KeyError: The provided gene symbol key was not found in the passed AnnData object.')
        print('   Check that your cell type markers are given in a format that your anndata object knows!')
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
            marker_res[i,j] = z_scores.X[np.ix_(cluster_cells,marker_genes)].mean()
            j += 1
        i+=1

    variances = np.nanvar(marker_res, axis=0)
    if np.all(np.isnan(variances)):
        print("No variances could be computed, check if your cell markers are in the data set.")
        print("Maybe the cell marker IDs do not correspond to your gene_symbol_key input or the var_names")
        raise

    marker_res_df = pd.DataFrame(marker_res, columns=clusters, index=marker_dict.keys())

    #Return the median of the variances over the clusters
    return(marker_res_df)

# %%
cluster_celltypes = evaluate_partition(adata, marker_annotation["symbols"].to_dict(), "symbol", partition_key="leiden").idxmax()

# %%
adata.obs["celltype"] = cluster_celltypes[adata.obs["leiden"]].values
adata.obs["celltype"] = adata.obs["celltype"].astype(str)
adata.obs.loc[adata.obs["leiden"] == "4", "celltype"] = "NKT"

# %%
transcriptome.adata.obs["log_n_counts"] = np.log(transcriptome.adata.obs["n_counts"])

# %%
sc.pl.umap(
    adata,
    color = ["celltype", "log_n_counts", "leiden"]
)
sc.pl.umap(
    adata,
    color = transcriptome.gene_id(marker_annotation["symbols"].explode()),
    title = marker_annotation["symbols"].explode()
)

# %%
transcriptome.obs = adata.obs
transcriptome.adata = adata

# %% [markdown]
# ## Create cell type dataset

# %%
import peakfreeatac.transcriptome

# %%
dataset_name_original = "pbmc10k"
dataset_name = dataset_name_original + "_clustered"

# %%
folder_data_preproc_original = folder_data_preproc.parent / (dataset_name_original)

# %%
transcriptome_original = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc_original / "transcriptome")

# %%
folder_data_preproc = folder_data_preproc.parent / dataset_name
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")

# %%
sc.tl.leiden(transcriptome_original.adata, resolution = 100)

# %%
transcriptome_original.adata.obs.groupby("leiden").size().plot(kind = "hist")

# %%
# transcriptome_original.adata.obs["leiden"] = np.arange(transcriptome_original.adata.obs.shape[0])

# %%
groupby = transcriptome_original.adata.obs.reset_index().groupby("leiden")
obs = groupby["cell"].apply(list).to_frame()
obs.columns = ["cell_original"]
obs.index.name = "cell"
obs.index = obs.index.astype(str)

# %%
X = {}
for celltype, group in groupby:
    X[celltype] = np.array(transcriptome_original.adata[group.index].X.mean(0))[0]
X = pd.DataFrame(X).T.values

# %%
transcriptome.var = transcriptome_original.var
transcriptome.obs = obs

# %%
adata = sc.AnnData(X, obs = transcriptome.obs, var = transcriptome.var)

# %%
transcriptome.adata = adata

# %%
transcriptome.create_X()

# %%
# !ln -s {folder_data_preproc_original}/promoters_10k10k.csv {folder_data_preproc}/promoters_10k10k.csv

# %%
# !ln -s {folder_data_preproc_original}/genes.gff {folder_data_preproc}/genes.gff

# %% [markdown]
# ## Create windows

# %% [markdown]
# ### Creating promoters

# %%
import tabix

# %%
fragments_tabix = tabix.open(str(folder_data_preproc / "atac_fragments.tsv.gz"))

# %% [markdown]
# #### Define promoters

# %%
promoter_name, (padding_negative, padding_positive) = "4k2k", (2000, 4000)
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)
# promoter_name, (padding_negative, padding_positive) = "1k1k", (1000, 1000)

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
promoters["start"] = promoters["tss"] - padding_negative * promoters["positive_strand"] - padding_positive * promoters["negative_strand"]
promoters["end"] = promoters["tss"] + padding_negative * promoters["negative_strand"] + padding_positive * promoters["positive_strand"]

# %%
promoters = promoters.drop(columns = ["positive_strand", "negative_strand"], errors = "ignore")

# %%
promoters

# %%
promoters.to_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"))

# %% [markdown]
# #### Create fragments

# %%
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")

# %%
import pathlib
import peakfreeatac.fragments
fragments = pfa.fragments.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
var = pd.DataFrame(index = promoters.index)
var["ix"] = np.arange(var.shape[0])

n_genes = var.shape[0]

# %%
obs = transcriptome.adata.obs[[]].copy()
obs.index.name = "cell"
obs["ix"] = np.arange(obs.shape[0])

if "cell_original" in transcriptome.adata.obs.columns:
    cell_ix_to_cell = transcriptome.adata.obs["cell_original"].explode()
    cell_to_cell_ix = pd.Series(cell_ix_to_cell.index.astype(int), cell_ix_to_cell.values)
else:
    cell_to_cell_ix = obs["ix"].to_dict()

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
# we use int32 for smaller memory usage
# will have to be converted to floats anyway...
coordinates = torch.tensor(np.array(coordinates_raw, dtype = np.int32))

# %%
# int64 is needed for torch_scatter
mapping = torch.tensor(np.array(mapping_raw), dtype = torch.int64)

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

# %% [markdown]
# Store

# %%
fragments.mapping = mapping
fragments.coordinates = coordinates

# %% [markdown]
# Create cellxgene index pointers

# %%
cellxgene = fragments.mapping[:, 0] * fragments.n_genes + fragments.mapping[:, 1]
n_cellxgene = fragments.n_genes * fragments.n_cells

# %%
import torch_sparse

# %%
cellxgene_indptr = torch.ops.torch_sparse.ind2ptr(cellxgene, n_cellxgene)

# %%
assert fragments.coordinates.shape[0] == cellxgene_indptr[-1]

# %%
fragments.cellxgene_indptr = cellxgene_indptr

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

# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ###### Old

# %% [markdown]
# Splits up the fragments into "splits" for training, and precalculates several elements that a model can use:
# - `fragments_cellxgene_idx`, the cellxgene index locally, i.e. for the chosen cells and genes
# - `fragment_count_mapping`: a dictionary with k:# fragments and v:fragments for which the cellxgene contains k number of fragments

# %%
folds = pfa.fragments.Folds(fragments.n_cells, fragments.n_genes, 1000, 5000, n_folds = 5)
folds.populate(fragments)

# %% [markdown]
# Create mapping between # of fragments per cellxgene

# %%
# save
pfa.save(folds, open(fragments.path / "folds.pkl", "wb"))

# %%
# !ls -lh {fragments.path}

# %% [markdown]
# Split across genes too

# %%
folds = pfa.fragments.FoldsDouble(fragments.n_cells, fragments.n_genes, 200, n_folds = 1, perc_train = 4/5)
# folds = pfa.fragments.FoldsDouble(10000, 20, 200, n_folds = 1, perc_train = 4/5)
folds.populate(fragments)

# %%
folds[0].plot()
None

# %%
# save
pfa.save(folds, open(fragments.path / "folds2.pkl", "wb"))

# %%
for k, v in folds[0][0].__dict__.items():
    if torch.is_tensor(v):
        print(k, v.size())

# %%
# !ls -lh {fragments.path}/folds2.pkl

# %% [markdown]
# ### Create windows around genes

# %% [markdown]
# Old code used to create windows around genes and determine which peaks are in that window

# %%
genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col = 0)

# %%
genewindows = genes.copy()
genewindows["start"] = np.maximum(genewindows["start"] - 20000, 0)
genewindows["end"] = genewindows["end"] + 20000

# %%
genewindows_bed = pybedtools.BedTool.from_dataframe(genewindows.reset_index()[["chr", "start", "end", "strand", "gene"]])

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

# %%

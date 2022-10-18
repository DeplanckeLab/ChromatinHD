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
# # PBMC10K

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

# %%
import peakfreeatac as pfa

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"
folder_data_preproc = folder_data / "pbmc10kgran"
folder_data_preproc.mkdir(exist_ok = True, parents = True)

# %%
folder_accessibilities = folder_root / "accessibilities" / folder_data_preproc.name

# %%
import peakfreeatac.transcriptome

# %%
transcriptome = pfa.transcriptome.Transcriptome(folder_data_preproc)

# %%
import peakfreeatac.flow
peakfreeatac.flow.Flow(folder_accessibilities)

# %% [markdown]
# ## Create and count peaks

# %%
import peakfreeatac.accessibility

# %%
accessibility = pfa.accessibility.HalfPeak(
    path = folder_accessibilities / "halfpeak"
)

# accessibility = pfa.accessibility.FullPeak(
#     path = folder_accessibilities / "fullpeak"
# )

# accessibility = pfa.accessibility.FragmentPeak(
#     path = folder_accessibilities / "fragmentpeak"
# )

# %%
peak_annot = pd.read_table(folder_data_preproc / "peak_annot.tsv")
peak_annot.index = pd.Index(peak_annot.chrom + ":" + peak_annot.start.astype(str) + "-" + peak_annot.end.astype(str), name = "peak")
variable_genes = pd.read_json(folder_data_preproc / "variable_genes.json", typ="series")
cell_barcodes = transcriptome.obs.index

# %%
gene_peak_links = pd.read_csv(folder_data_preproc / "gene_peak_links_20k.csv", index_col = 0)

# %%
peak_annot_oi = peak_annot.loc[gene_peak_links.loc[gene_peak_links["gene"].isin(variable_genes)]["peak"].unique()]

# %%
accessibility.create_peaks(peak_annot_oi)

# %%
accessibility.count_peaks(folder_data_preproc / "atac_fragments.tsv.gz", cell_barcodes)

# %%
counts = accessibility.counts

# %%
fig, ax = plt.subplots()
ax.hist(np.log10(np.array(accessibility.counts.sum(0))[0]), range = (0, 10), bins = 100)
ax.set_xlim(0, 10)

# %%
accessibility.create_adata(transcriptome.adata)

# %% [markdown]
# ## Load peaks

# %%
# gene = "IGHE"
gene = "CD3D"
# gene = "IGHD"
# gene = "PTPRC"
# gene = "AL136456.1"
# gene = "RALGPS2"
# gene = "ANGPTL1"
# gene = "CD247"
# gene = "NVL"
# gene = "PRF1"
# gene = "IGHA1"

# %%
sc.pl.umap(transcriptome.adata, color = [gene])

# %%
peaks_oi = accessibility.var.query("gene == @gene").index
peaks_oi = accessibility.var.index[accessibility.var["original_peak"].isin(gene_peak_links.query("gene == @gene")["peak"].tolist())]

# %%
sc.pl.umap(accessibility.adata, color = peaks_oi[:10])
# sc.pl.umap(accessibility.adata, color = peak_ids[:10])

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
folder_data_preproc = folder_data / "lymphoma"
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
# accessibility = pfa.accessibility.HalfPeak(
accessibility = pfa.accessibility.FullPeak(
# accessibility = pfa.accessibility.FragmentPeak(
    folder = folder_accessibilities
)

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
transcriptome.adata.var.query("highly_variable")

# %%
# gene = "IGHE"
# gene = "CD3D"
# gene = "IGHD"
# gene = "PTPRC"
# gene = "AL136456.1"
# gene = "RALGPS2"
# gene = "ANGPTL1"

gene = "CD247"

gene = "CD4"
# gene = "PCNA"
# gene = "TBX21"

# gene = "PAX5"

# gene = "EBLN3P"
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

# %% [markdown]
# ## Visualize fragments around a peak

# %%
import subprocess as sp
import tqdm
import math

# %%
tabix_location = pfa.accessibility.tabix_location

# %%
fragments_location = folder_data_preproc / "atac_fragments.tsv.gz"

# %%
# window_info = gff.query("gene == @gene").iloc[0].copy()
# window_info = accessibility_full.var.query("original_peak == @best_peak_original_peak").iloc[0].copy()
window = accessibility.var.query("peak_type == 'promoter' & gene == @gene").iloc[0]
# window = accessibility.var.loc[accessibility.var["gene"] == gene].iloc[0]

desired_window_size = 200000
# desired_window_size = (window["end"] - window["start"])

padding = desired_window_size - (window["end"] - window["start"])

window["start"] = window["start"] - math.floor(padding/2)
window["end"] = window["end"] + math.ceil(padding/2)


# %%
def name_window(window_info):
    return window_info["chrom"] + ":" + str(window_info["start"]) + "-" + str(window_info["end"])

window_name = name_window(window)

# %%
process = sp.Popen([tabix_location, fragments_location, window_name], stdout=sp.PIPE, stderr=sp.PIPE)
# process.wait()
# out = process.stdout.read().decode("utf-8")
fragments = pd.read_table(process.stdout, names = ["chrom", "start", "end", "barcode", "i"])

# %%
x_expression = sc.get.obs_df(transcriptome.adata, gene)

# %%
X = np.zeros((x_expression.index.size, window["end"] - window["start"]))        

# %%
barcode_map = pd.Series(np.arange(x_expression.index.size), x_expression.index)

# %%
for _, fragment in fragments.iterrows():
    if fragment["barcode"] not in barcode_map:
        continue
    X[barcode_map[fragment["barcode"]], (fragment["start"] - window["start"]):(fragment["end"] - window["start"])] = 1.

# %%
cells_oi = np.random.choice(barcode_map.index, 10000, replace = False)
cells_oi = x_expression[cells_oi].sort_values().index
cells_oi_ix = barcode_map[cells_oi].values

# %%
window

# %%
fig, ax = plt.subplots(figsize = (20, 20))
ax.matshow(X[cells_oi_ix, ], aspect = 5.)

# %%
window_name

# %%
plt.plot(X[cells_oi_ix, ].sum(1))

# %%
window_name

# %%
plt.plot(np.arange(X.shape[1]), X.sum(0))

# %%
x_atac = fragments.groupby("barcode").size()
x_atac = x_atac.reindex(x_expression.index, fill_value = 0)

# %%
sns.ecdfplot(hue = x_atac > 1, x = x_expression)

# %%

# %%

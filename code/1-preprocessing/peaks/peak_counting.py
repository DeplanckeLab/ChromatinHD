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

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')

import pickle

import scanpy as sc
import pathlib

import torch

import tqdm.auto as tqdm

# %%
import peakfreeatac as pfa
import peakfreeatac.peakcounts
import peakfreeatac.data

# %%
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "alzheimer"
# dataset_name = "brain"

# dataset_name = "FLI1_7"
# dataset_name = "PAX2_7"
# dataset_name = "NHLH1_7"
# dataset_name = "CDX2_7"
# dataset_name = "CDX1_7"
# dataset_name = "MSGN1_7"
# dataset_name = "KLF4_7"
# dataset_name = "KLF5_7"
# dataset_name = "PTF1A_4"

# peaks_name = "cellranger"
peaks_name = "genrich"
# peaks_name = "macs2"
# peaks_name = "macs2_improved"
# peaks_name = "stack"
# peaks_name = "rolling_500"; window_size = 500
# peaks_name = "rolling_50"; window_size = 50

# %%
folder_data_preproc = pfa.get_output() / "data" / dataset_name
folder_root = pfa.get_output()

# %%
promoter_name, window = "10k10k", (-10000, 10000)
# promoter_name, window = "20kpromoter", (10000, 0)

promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
promoters

# %%
transcriptome = pfa.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = peakfreeatac.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
peakcounts = pfa.peakcounts.FullPeak(folder = pfa.get_output() / "peakcounts" / dataset_name / peaks_name)

# %% [markdown]
# ## Merge peaks with promoters

# %%
if peaks_name == "stack":
    peaks = promoters.reset_index()[["chr", "start", "end", "gene"]]
elif peaks_name.startswith("rolling"):
    peaks = []
    for gene, promoter in promoters.iterrows():
        starts = np.arange(promoter["start"], promoter["end"], step = window_size)
        ends = np.hstack([starts[1:], [promoter["end"]]])
        peaks.append(pd.DataFrame({"chrom":promoter["chr"], "start":starts, "ends":ends, "gene":gene}))
    peaks = pd.concat(peaks)
    
    peaks_folder = folder_root / "peaks" / dataset_name / peaks_name
    peaks_folder.mkdir(exist_ok = True, parents = True)
    peaks.to_csv(peaks_folder / "peaks.bed", index = False, header = False, sep = "\t")
else:
    peaks_folder = folder_root / "peaks" / dataset_name / peaks_name
    peaks = pd.read_table(peaks_folder / "peaks.bed", names = ["chrom", "start", "end"], usecols = [0, 1, 2])

    if peaks_name == "genrich":
        peaks["start"] += 1

# %%
import pybedtools
promoters_bed = pybedtools.BedTool.from_dataframe(promoters.reset_index()[["chr", "start", "end", "gene"]])
peaks_bed = pybedtools.BedTool.from_dataframe(peaks)

# %%
if peaks_name != "stack":
    intersect = promoters_bed.intersect(peaks_bed)
    intersect = intersect.to_dataframe()

    # peaks = intersect[["score", "strand", "thickStart", "name"]]
    peaks = intersect
peaks.columns = ["chrom", "start", "end", "gene"]
peaks = peaks.loc[peaks["start"] != -1]
peaks.index = pd.Index(peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str), name = "peak")

# %%
peaks["relative_begin"] = (peaks["start"] - promoters.loc[peaks["gene"], "start"].values + window[0])
peaks["relative_stop"] = (peaks["end"] - promoters.loc[peaks["gene"], "start"].values + window[0])

# %%
peaks["relative_start"] = np.where(promoters.loc[peaks["gene"], "strand"] == 1, peaks["relative_begin"], -peaks["relative_stop"])
peaks["relative_end"] = np.where(promoters.loc[peaks["gene"], "strand"] == -1, -peaks["relative_begin"], peaks["relative_stop"])

# %%
peaks["gene_ix"] = fragments.var["ix"][peaks["gene"]].values

# %%
peaks["peak"] = peaks.index

# %%
peaks.index = peaks.peak + "_" + peaks.gene
peaks.index.name = "peak_gene"

# %%
peakcounts.peaks = peaks

# %% [markdown]
# ## Count

# %%
transcriptome.adata.obs["ix"] = np.arange(transcriptome.adata.obs.shape[0])

# %%
peakcounts.count_peaks(folder_data_preproc / "atac_fragments.tsv.gz", transcriptome.adata.obs.index)
# peakcounts.count_peaks(folder_data_preproc / "fragments.tsv.gz", transcriptome.adata.obs["ix"].astype(str).values)

# %% [markdown]
# ## Visualize

# %%
adata_atac = sc.AnnData(peakcounts.counts.astype(np.float32), obs = transcriptome.obs)
sc.pp.normalize_total(adata_atac)
sc.pp.log1p(adata_atac)
sc.pp.highly_variable_genes(adata_atac)

# %%
import sklearn

# %%
adata_atac.var["highly_variable"].sum()

# %%
sc.pp.pca(adata_atac, use_highly_variable=False)
sc.pp.neighbors(adata_atac, use_rep = "X_pca")
sc.tl.umap(adata_atac)
sc.pl.umap(adata_atac)

# %%
adata_transcriptome = transcriptome.adata
sc.pp.neighbors(adata_transcriptome)
sc.tl.leiden(adata_transcriptome, resolution = 1.0)

# %%
sc.pl.umap(adata_transcriptome, color = ["leiden"])

# %%
adata_atac.obs["leiden_transcriptome"] = adata_transcriptome.obs["leiden"]
adata_atac.obs["n_fragments"] = np.log1p(torch.bincount(fragments.cellmapping, minlength = fragments.n_cells).cpu().numpy())

# %%
sc.pl.umap(adata_atac, color = ["leiden_transcriptome", "n_fragments"], legend_loc = "on data")

# %% [markdown]
# ----

# %%
fig, ax = plt.subplots()

ax.set_aspect(1)
ax.scatter(motifscores["logodds"], motifscores2["logodds"], color = "grey", s = 1.0)
ax.scatter(motifscores.loc[motifs_oi, "logodds"], motifscores2.loc[motifs_oi, "logodds"], color = "orange", s = 2)
ax.axline((0, 0), slope=1, color = "#555", zorder = 0, lw = 1)
ax.axline((0, 0), slope=1.5, color = "#888", zorder = 0, lw = 1)
ax.axline((0, 0), slope=1/1.5, color = "#888", zorder = 0, lw = 1)
ax.axline((0, 0), slope=2, color = "#aaa", zorder = 0, lw = 1)
ax.axline((0, 0), slope=0.5, color = "#aaa", zorder = 0, lw = 1)
ax.axline((0, 0), slope=3, color = "#ccc", zorder = 0, lw = 1)
ax.axline((0, 0), slope=1/3, color = "#ccc", zorder = 0, lw = 1)
ax.axline((0, 0), slope=6, color = "#ddd", zorder = 0, lw = 1)
ax.axline((0, 0), slope=1/6, color = "#ddd", zorder = 0, lw = 1)
ax.axline((0, -intercept), slope=1/slope, color = "red")

# %% [markdown]
# ## Predict (temporarily here 👷)

# %%
import peakfreeatac.prediction

# %%
peaks_names = [
    # "cellranger",
    # "macs2",
    "rolling_500"
]
design_peaks = pd.DataFrame({"peaks":peaks_names})
methods = [
    ["_xgboost", peakfreeatac.prediction.PeaksGene],
    # ["_linear", peakfreeatac.prediction.PeaksGeneLinear],
    # ["_polynomial", peakfreeatac.prediction.PeaksGenePolynomial],
    # ["_lasso", peakfreeatac.prediction.PeaksGeneLasso]
]
design_methods = pd.DataFrame(methods, columns = ["method_suffix", "method_class"])
dataset_names = [
    "pbmc10k",
    "lymphoma",
    "e18brain",
]
design_datasets = pd.DataFrame({"dataset":dataset_names})

# %%
design = pfa.utils.crossing(design_peaks, design_methods, design_datasets)

# %%
for _, design_row in design.iterrows():
    dataset_name = design_row["dataset"]
    peaks_name = design_row["peaks"]
    transcriptome = pfa.data.Transcriptome(pfa.get_output() / "data" / dataset_name / "transcriptome")
    peakcounts = pfa.peakcounts.FullPeak(folder = pfa.get_output() / "peakcounts" / dataset_name / peaks_name)
    
    peaks = peakcounts.peaks
    
    gene_peak_links = peaks.reset_index()
    gene_peak_links["gene"] = pd.Categorical(gene_peak_links["gene"], categories = transcriptome.adata.var.index)
    
    fragments = peakfreeatac.data.Fragments(pfa.get_output() / "data" / dataset_name / "fragments" / promoter_name)
    folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
    
    method_class = design_row["method_class"]
    method_suffix = design_row["method_suffix"]
    prediction = method_class(
        pfa.get_output() / "prediction_positional" / dataset_name / promoter_name / (peaks_name + method_suffix),
        transcriptome,
        peakcounts
    )
    
    prediction.score(gene_peak_links, folds)
    
    prediction.scores = prediction.scores
    # prediction.models = prediction.models

# %%
prediction.scores

# %%
prediction.scores["cor"].unstack().T.sort_values("validation").plot()

# %%

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
# dataset_name = "pbmc10k"
dataset_name = "e18brain"

peaks_name = "cellranger"
# peaks_name = "genrich"
# peaks_name = "macs2"
# peaks_name = "stack"
# peaks_name = "rolling_500"; window_size = 500

# %%
folder_data_preproc = pfa.get_output() / "data" / dataset_name
folder_root = pfa.get_output()

# %%
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)
# promoter_name, (padding_negative, padding_positive) = "20kpromoter", (10000, 0)

promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

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
peakcounts.peaks = peaks

# %% [markdown]
# ## Count

# %%
peakcounts.count_peaks(folder_data_preproc / "atac_fragments.tsv.gz", transcriptome.obs.index)

# %%
adata = sc.AnnData(peakcounts.counts, obs = transcriptome.obs)
sc.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)

# %%
adata.var["highly_variable"].sum()

# %%
sc.pp.pca(adata, use_highly_variable=False)
sc.pp.neighbors(adata, use_rep = "X_pca")

# %%
sc.tl.umap(adata)

# %%
sc.pl.umap(adata)

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

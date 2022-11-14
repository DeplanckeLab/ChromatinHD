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
import peakfreeatac.transcriptome

# %%
dataset_name = "lymphoma"
# dataset_name = "pbmc10k"

peaks_name = "cellranger"
# peaks_name = "genrich"
# peaks_name = "macs2"
# peaks_name = "stack"
peaks_name = "rolling_200"; window_size = 200

# %%
folder_data_preproc = pfa.get_output() / "data" / dataset_name
folder_root = pfa.get_output()

# %%
transcriptome = pfa.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")

# %%
promoters = pd.read_csv(folder_data_preproc / "promoters.csv", index_col = 0)

# %%
peakcounts = pfa.peakcounts.FullPeak(folder = pfa.get_output() / "peakcounts" / dataset_name / peaks_name)

# %%
promoter = promoters.iloc[0]

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

# %% [markdown]
# ## Predict (temporarily here 👷)

# %%
peaks = peakcounts.peaks

# %%
import peakfreeatac.prediction

# %%
# method_suffix = ""; prediction_class = peakfreeatac.prediction.PeaksGene
method_suffix = "_linear"; prediction_class = peakfreeatac.prediction.PeaksGeneLinear

# %%
prediction = prediction_class(
    pfa.get_output() / "prediction_promoter" / dataset_name / (peaks_name + method_suffix),
    transcriptome,
    peakcounts
)

# %%
import pybedtools

# %% [markdown]
# Link peaks and genes (promoters)

# %%
gene_peak_links = peaks.reset_index()
gene_peak_links["gene"] = pd.Categorical(gene_peak_links["gene"], categories = transcriptome.adata.var.index)

# %% [markdown]
# Split train/validation

# %%
test_cell_cutoff = int((transcriptome.obs.shape[0] / 3)*2) # split train/test cells

# %%
cells_train = transcriptome.obs.index[:test_cell_cutoff]
cells_validation = transcriptome.obs.index[test_cell_cutoff:]

# %%
prediction.score(gene_peak_links, cells_train, cells_validation)

# %%
prediction.scores["mse_diff"] = prediction.scores["mse"] - prediction.scores["mse_dummy"]

# %%
# prediction.scores["label"] = transcriptome.symbol(prediction.scores["gene"]).values

# %%
prediction.scores = prediction.scores

# %%
# !ls {prediction.path}

# %%

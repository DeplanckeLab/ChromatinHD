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
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc
import pathlib

import tqdm.auto as tqdm

# %%
import peakfreeatac as pfa
import peakfreeatac.fragments
import peakfreeatac.transcriptome

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
folder_data_preproc = folder_data / dataset_name


# %%
class Prediction(pfa.flow.Flow):
    pass
prediction = Prediction(pfa.get_output() / "prediction_promoter" / dataset_name / "nn")

# %%
gene_aggrscores = pd.read_csv(prediction.path / "gene_aggscores.csv", index_col = ["phase", "gene"])

# %%
method_names = [
    "nn",
    "cellranger",
    "macs2",
    "macs2_linear",
    "cellranger_linear",
    "stack_linear"
]

# %%
scores = {}
for method_name in method_names:
    prediction = Prediction(pfa.get_output() / "prediction_promoter" / dataset_name / method_name)
    if method_name == "nn":
        gene_aggscores = pd.read_csv(prediction.path / "gene_aggscores.csv", index_col = ["phase", "gene"])
    else:
        gene_aggscores = pd.read_table(prediction.path / "scores.tsv", index_col = ["phase", "gene"])
        
    scores[method_name] = gene_aggscores
scores = pd.concat(scores, names = ["method", *gene_aggscores.index.names])

# %%
genes_oi = scores.query("phase == 'validation'").groupby("gene")["mse_diff"].min() < -1e-2
genes_oi = genes_oi.index[genes_oi]
genes_oi.shape[0]

# %%
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")

# %%
method_info = pd.DataFrame([
    ["nn", "NN (ours)"],
    ["macs2_linear", "MACS2 + Linear"],
    ["macs2", "MACS2 + XGBoost"],
    ["cellranger_linear", "cellranger + Linear"],
    ["cellranger", "cellranger + XGBoost"],
    ["stack_linear", "Stacked + Linear"],
], columns = ["method", "label"]).set_index("method")

# %%
fig, (ax_all, ax_able) = plt.subplots(1, 2, figsize = (3, 3), sharey = True)
scores.groupby(["method", "phase"])["mse_diff"].mean().unstack().plot(kind = "barh", ax = ax_all, legend = False)
scores.query("gene in @genes_oi").groupby(["method", "phase"])["mse_diff"].mean().unstack().plot(kind = "barh", ax = ax_able, legend = False)
ax_all.set_xlim(ax_all.get_xlim()[::-1])
ax_able.set_xlim(ax_able.get_xlim()[::-1])
ax_all.set_xlabel("MSE difference")
ax_all.set_yticklabels(method_info.loc[[tick._text for tick in ax_all.get_yticklabels()]]["label"])

# %%
fig, (ax_all, ax_able) = plt.subplots(1, 2, figsize = (3, 3), sharey = True)
scores.groupby(["method", "phase"])["cor"].mean().unstack().plot(kind = "barh", ax = ax_all, legend = False)
scores.query("gene in @genes_oi").groupby(["method", "phase"])["cor"].mean().unstack().plot(kind = "barh", ax = ax_able, legend = False)
ax_all.set_xlabel("Correlation")
ax_all.set_yticklabels(method_info.loc[[tick._text for tick in ax_all.get_yticklabels()]]["label"])

# %%
scores.groupby(["method", "phase"])["mse_diff"].mean().unstack().T.plot()
scores.query("gene in @genes_oi").groupby(["method", "phase"])["mse_diff"].mean().unstack().T.plot()

# %%
scores.groupby(["method", "phase"])["cor"].mean().unstack().T.plot()

# %%
scores.query("gene in @genes_oi").groupby(["method", "phase"])["cor"].mean().unstack().T.plot()

# %%

# %%

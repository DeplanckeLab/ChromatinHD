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

dataset_name = "lymphoma"
# dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# %%
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)


# %%
class Prediction(pfa.flow.Flow):
    pass

model_name = "v7"
prediction = Prediction(pfa.get_output() / "prediction_promoter" / dataset_name / promoter_name / model_name)

# %%
# !ls {prediction.path}

# %%
gene_scores = pd.read_pickle(prediction.path / "scoring" / "overall" / "gene_scores.pkl")

# %%
method_names = [
    # "v3",
    "v5",
    # "cellranger",
    # "macs2",
    # "macs2_linear",
    # "cellranger_linear",
    "cellranger_polynomial",
    # "stack_linear",
    # "rolling_200",
    # "rolling_200_linear"
]

# %%
scores = {}
for method_name in method_names:
    prediction = Prediction(pfa.get_output() / "prediction_promoter" / dataset_name / promoter_name / method_name)
    if method_name in ["v5"]:
        gene_aggscores = pd.read_pickle(prediction.path / "scoring" / "overall" / "gene_scores.pkl")
    else:
        gene_aggscores = pd.read_table(prediction.path / "scores.tsv", index_col = ["phase", "gene"])
        
    scores[method_name] = gene_aggscores
scores = pd.concat(scores, names = ["method", *gene_aggscores.index.names])

# %%
genes_oi = scores.query("phase == 'validation'").groupby("gene")["mse_diff"].min() < -1e-3
genes_oi = genes_oi.index[genes_oi]
genes_oi.shape[0]

# %%
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")

# %%
method_info = pd.DataFrame([
    ["v5", "V5 (ours)"],
    ["macs2_linear", "MACS2 + Linear"],
    ["macs2", "MACS2 + XGBoost"],
    ["cellranger_linear", "cellranger + Linear"],
    ["cellranger_polynomial", "cellranger + Polynomial"],
    ["cellranger", "cellranger + XGBoost"],
    ["stack_linear", "Stacked + Linear"],
    ["rolling_200_linear", "Rolling window + Linear"],
    ["rolling_200", "Rolling window + XGBoost"],
], columns = ["method", "label"]).set_index("method")

# %%
fig, (ax_all, ax_able) = plt.subplots(1, 2, figsize = (3, 3), sharey = True)

plotdata_all = scores.groupby(["method", "phase"])["mse_diff"].mean().to_frame().reset_index()
plotdata_able = scores.query("gene in @genes_oi").groupby(["method", "phase"])["mse_diff"].mean().to_frame().reset_index()

sns.barplot(plotdata_all, y = "method", hue = "phase", x = "mse_diff", ax = ax_all)
sns.barplot(plotdata_able, y = "method", hue = "phase", x = "mse_diff", ax = ax_able)

ax_all.legend([],[], frameon=False)
ax_able.legend([],[], frameon=False)
ax_all.set_xlim(ax_all.get_xlim()[::-1])
ax_able.set_xlim(ax_able.get_xlim()[::-1])
# ax_all.set_xlim(
ax_all.set_xlabel("MSE difference")
# ax_all.set_yticklabels(method_info.loc[[tick._text for tick in ax_all.get_yticklabels()]]["label"])

# %%
fig, (ax_all, ax_able) = plt.subplots(1, 2, figsize = (3, 3), sharey = True)

plotdata_all = scores.groupby(["method", "phase"])["mse_diff"].mean().to_frame().reset_index()#scores.groupby(["method", "phase"])["mse_diff"].mean().unstack()
plotdata_able = scores.query("gene in @genes_oi").groupby(["method", "phase"])["mse_diff"].mean().to_frame().reset_index()#scores.query("gene in @genes_oi").groupby(["method", "phase"])["mse_diff"].mean().unstack()

sns.barplot(plotdata_all, y = "method", hue = "phase", x = "mse_diff", ax = ax_all)
sns.barplot(plotdata_able, y = "method", hue = "phase", x = "mse_diff", ax = ax_able)

ax_all.legend([],[], frameon=False)
ax_able.legend([],[], frameon=False)
ax_all.set_xlim(ax_all.get_xlim()[::-1])
ax_able.set_xlim(ax_able.get_xlim()[::-1])
# ax_all.set_xlim(
ax_all.set_xlabel("MSE difference")
# ax_all.set_yticklabels(method_info.loc[[tick._text for tick in ax_all.get_yticklabels()]]["label"])

# %%
scores.groupby(["method", "phase"])["mse_diff"].mean().unstack().T.plot()
scores.query("gene in @genes_oi").groupby(["method", "phase"])["mse_diff"].mean().unstack().T.plot()

# %%
fig, ax = plt.subplots()
scores.groupby(["method", "phase"])["cor"].mean().unstack().T.plot(ax = ax)
scores.query("gene in @genes_oi").groupby(["method", "phase"])["cor"].mean().unstack().T.plot(ax = ax)
ax.set_ylim(0)

# %%

# %%

# %%

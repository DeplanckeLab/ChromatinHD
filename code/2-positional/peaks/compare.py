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

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

dataset_names = [
    "pbmc10k",
    # "pbmc10k_gran",
    "lymphoma",
    "e18brain",
    "pbmc3k-pbmc10k",
    "lymphoma-pbmc10k",
    "pbmc10k_gran-pbmc10k"
]
# dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_names[0]
transcriptome = pfa.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
promoter_name, (padding_negative, padding_positive) = "10k10k", (-10000, 10000)
# promoter_name, (padding_negative, padding_positive) = "20kpromoter", (-10000, 0)

# %% tags=[]
method_names = [
    "counter",
    "counter_binary",
    "macs2_linear",
    "macs2_polynomial",
    "macs2_xgboost",
    "cellranger_linear",
    "cellranger_polynomial",
    "cellranger_xgboost",
    "rolling_500_linear",
    "rolling_500_xgboost",
    "v14_50freq_sum_sigmoid_initdefault",
]


# %%
class Prediction(pfa.flow.Flow):
    pass


# %%
scores = {}
for dataset_name in dataset_names:
    for method_name in method_names:
        prediction = Prediction(pfa.get_output() / "prediction_positional" / dataset_name / promoter_name / method_name)
        if (prediction.path / "scoring" / "overall" / "genescores.pkl").exists():
            genescores = pd.read_pickle(prediction.path / "scoring" / "overall" / "genescores.pkl")
            
            if method_name.startswith("v14") or (method_name == "counter"):
                pass
            else:
                genescores["mse"] = (genescores["mse"] ** 2) * 100
            
            scores[(dataset_name, method_name)] = genescores
scores = pd.concat(scores, names = ["dataset", "method", *genescores.index.names])

# %%
genes_oi = scores.query("phase == 'validation'").groupby("gene")["cor"].min() > 0.01
genes_oi = genes_oi.index[genes_oi]
genes_oi.shape[0]

# %%
method_info= pd.DataFrame([
    ["counter", "Counter (baseline)", "baseline"],
    ["counter_binary", "Counter binary", "baseline"],
    ["v9", "Ours", "ours"],
    ["v11", "Ours", "ours"],
    ["v14_50freq_sum_sigmoid_initdefault", "Positional NN", "ours"],
    ["macs2_linear", "Linear", "peak"],
    ["macs2_polynomial", "Quadratic", "peak"],
    ["macs2_xgboost", "XGBoost", "peak"],
    ["cellranger_linear", "Linear", "peak"],
    ["cellranger_polynomial", "Quadratic", "peak"],
    ["cellranger_xgboost", "XGBoost", "peak"],
    ["rolling_500_linear", "Linear", "rolling"],
    ["rolling_500_xgboost", "XGBoost", "rolling"]
], columns = ["method", "label", "type"]).set_index("method")
missing_methods = pd.Index(method_names).difference(method_info.index).tolist()
missing_method_info = pd.DataFrame({"method":missing_methods, "label":missing_methods}).set_index("method")
method_info = pd.concat([method_info, missing_method_info])
method_info = method_info.loc[method_names]
method_info["ix"] = -np.arange(method_info.shape[0])

method_info.loc[method_info["type"] == 'baseline', "color"] = "grey"
method_info.loc[method_info["type"] == 'ours', "color"] = "#0074D9"
method_info.loc[method_info["type"] == 'rolling', "color"] = "#FF851B"
method_info.loc[method_info["type"] == 'peak', "color"] = "#FF4136"
method_info.loc[pd.isnull(method_info["color"]), "color"] = "black"

method_info["opacity"] = "88"
method_info.loc[method_info.index[-1], "opacity"] = "FF"

# %%
dummy_method = "counter"

# %%
(scores["cor"] - scores.xs(dummy_method, level = "method")["cor"])

# %%
try:
    scores = scores.drop(columns = ["cor_diff"])
except:
    pass
scores = scores.join(pd.DataFrame({"cor_diff":(scores["cor"] - scores.xs(dummy_method, level = "method")["cor"])}))

# %%
cutoff = 0.005
metric = "mse_diff";metric_multiplier = -1;metric_limits = (-0.1, 0.1); metric_label = "$\Delta$ mse"
metric = "cor_diff";metric_multiplier = 1;metric_limits = (-0.015, 0.015); metric_label = "$\Delta$ cor"
# metric = "cor";metric_multiplier = 1;metric_limits = (0, 0.1); metric_label = "cor"

# %%
scores_all = scores.groupby(["dataset", "method", "phase"])[["cor", "mse"]].mean()
scores_able = scores.query("gene in @genes_oi").groupby(["method", "phase"])[[metric]].mean()

meanscores = scores.groupby(["dataset","method", "phase"])[["cor", "mse"]].mean()

diffscores = (meanscores - meanscores.xs(dummy_method, level = "method"))
diffscores.columns = diffscores.columns + "_diff"
diffscores["mse_diff"] = -diffscores["mse_diff"]

score_relative_all = meanscores.join(diffscores)

score_relative_all["better"] = ((scores - scores.xs(dummy_method, level = "method"))[metric] * metric_multiplier > cutoff).groupby(["dataset", "method", "phase"]).mean()
score_relative_all["worse"] = ((scores - scores.xs(dummy_method, level = "method"))[metric] * metric_multiplier < -cutoff).groupby(["dataset", "method", "phase"]).mean()
score_relative_all["same"] = ((scores - scores.xs(dummy_method, level = "method"))[metric].abs() < cutoff).groupby(["dataset", "method", "phase"]).mean()

# mean_scores = scores.query("gene in @genes_oi").groupby(["dataset", "method", "phase"])[["cor"]].mean()
# scores_relative_able = (mean_scores - mean_scores.xs(dummy_method, level = "method"))

# scores_relative_able["better"] = ((scores - scores.xs(dummy_method, level = "method")).query("gene in @genes_oi")["cor"] > cutoff).groupby(["dataset", "phase", "method"]).mean()
# scores_relative_able["worse"] = ((scores - scores.xs(dummy_method, level = "method")).query("gene in @genes_oi")["cor"] < -cutoff).groupby(["dataset", "phase", "method"]).mean()
# scores_relative_able["same"] = ((scores - scores.xs(dummy_method, level = "method").query("gene in @genes_oi"))["cor"].abs() < cutoff).groupby(["dataset", "phase", "method"]).mean()

# %%
import textwrap

# %%
fig, axes = plt.subplots(2, len(dataset_names), figsize = (len(dataset_names) * 6 / 4, 2 * len(method_names) / 4), sharey = True, gridspec_kw = {"wspace":0.05, "hspace":0.05}, squeeze=False)

for axes_dataset, dataset_name in zip(axes.T, dataset_names):
    axes_dataset = axes_dataset.tolist()
    ax = axes_dataset.pop(0)
    plotdata = score_relative_all.loc[dataset_name].reset_index()
    plotdata = plotdata.query("phase in ['validation', 'test']")
    plotdata["ratio"] = plotdata["better"] / (plotdata["worse"] + plotdata["better"])
    plotdata.loc[pd.isnull(plotdata["ratio"]), "ratio"] = 0.5
    
    ax.barh(
        width = plotdata[metric],
        y = method_info.loc[plotdata["method"]]["ix"],
        color = method_info.loc[plotdata["method"]]["color"],
        # color = "#FF4136" + method_info.loc[plotdata["method"]]["opacity"],
        lw = 0
    )
    ax.axvline(0., dashes = (2, 2), color = "#333333", zorder = -10)
    ax.set_yticks(method_info["ix"])
    ax.set_yticklabels(method_info["label"])
    ax.xaxis.tick_top()
    ax.set_xlim(*metric_limits)
    ax = axes_dataset.pop(0)
    ax.scatter(
        x = plotdata["ratio"],
        y = method_info.loc[plotdata["method"]]["ix"],
        c = method_info.loc[plotdata["method"]]["color"].values,
        # color = "#2ECC40" + method_info.loc[plotdata["method"]]["opacity"],
        # alpha = method_info.loc[plotdata["method"]]["opacity"]
    )
    ax.axvline(0.5, dashes = (2, 2), color = "#333333", zorder = -10)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
    
#     for _, method in method_info.iterrows():
#         ax.axhspan(
# sns.barplot(plotdata_able, y = "method", hue = "phase", x = "cor", ax = ax_cor_able)
# sns.barplot(plotdata_relative_all, y = "method", hue = "phase", x = "cor", ax = ax_rcor_all)
# sns.barplot(plotdata_relative_able, y = "method", hue = "phase", x = "cor", ax = ax_rcor_able)

for ax, dataset_name in zip(axes[0], dataset_names):
    ax.set_title("-\n".join(dataset_name.split("-")))

for i, ax in enumerate(axes[0]):
    ax.tick_params(axis = "y", length  = 0)
    ax.tick_params(axis = "x", labelsize=8)
    if i == 0:
        ax.set_xlabel(metric_label, fontsize = 8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    else:
        ax.set_xlabel("  \n  ", fontsize = 8)
        ax.set_xticks([])
    ax.xaxis.set_label_position('top')
    
for i, ax in enumerate(axes[1]):
    ax.tick_params(length  = 0)
    ax.tick_params(axis = "x", labelsize=8)
    if i == 0:
        ax.set_xlabel("# improved genes\n over # changing genes", fontsize = 8)
    else:
        ax.set_xlabel("  \n  ", fontsize = 8)
        ax.set_xticks([])
    ax.xaxis.set_label_position('bottom')

for ax in fig.axes:
    ax.legend([],[], frameon=False)
# ax_all.set_xlim(
# ax_cor_all.set_yticklabels(method_info.loc[[tick._text for tick in ax_cor_all.get_yticklabels()]]["label"])

# %%
datasets = pd.DataFrame({"dataset":dataset_names}).set_index("dataset")
datasets["color"] = sns.color_palette("Set1", datasets.shape[0])

# %%
plotdata = pd.DataFrame({
    "cor_total":scores.xs("v14_50freq_sum_sigmoid_initdefault", level = "method").xs("validation", level = "phase")["cor"],
    "cor_a":scores.xs("v14_50freq_sum_sigmoid_initdefault", level = "method").xs("validation", level = "phase")["cor_diff"],
    "cor_b":scores.xs("cellranger_linear", level = "method").xs("validation", level = "phase")["cor_diff"],
    "dataset":scores.xs("cellranger_linear", level = "method").xs("validation", level = "phase").index.get_level_values("dataset")
})
plotdata = plotdata.query("cor_total > 0.05")
plotdata["diff"] = plotdata["cor_a"] - plotdata["cor_b"]
plotdata = plotdata.sample(n = plotdata.shape[0])

# %%
fig, ax = plt.subplots(figsize = (3, 3))
ax.axline((0, 0), slope=1, dashes = (2, 2), zorder = 1, color = "#333")
ax.axvline(0, dashes = (1, 1), zorder = 1, color = "#333")
ax.axhline(0, dashes = (1, 1), zorder = 1, color = "#333")
plt.scatter(plotdata["cor_b"], plotdata["cor_a"], c = datasets.loc[plotdata["dataset"], "color"], alpha = 0.5, s = 1)
ax.set_xlabel("$\Delta$ cor Cellranger linear")
ax.set_ylabel("$\Delta$ cor Positional NN", rotation = 0, ha = "right", va = "center")

# %%
plotdata.sort_values("diff", ascending = False).head(20)

# %%

# %%

# %%

# %%

# %%

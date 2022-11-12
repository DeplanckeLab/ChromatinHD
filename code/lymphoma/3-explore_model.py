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

import torch_scatter
import torch

import tqdm.auto as tqdm

# %%
import peakfreeatac as pfa
import peakfreeatac.fragments
import peakfreeatac.transcriptome

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"
folder_data_preproc = folder_data / "lymphoma"
folder_data_preproc.mkdir(exist_ok = True, parents = True)

# %%
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")
fragments = peakfreeatac.fragments.Fragments(folder_data_preproc / "fragments")

# %%
splits = pickle.load(open("splits.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# %% [markdown]
# ## Overall performace

# %%
splits = [split.to("cuda") for split in splits]
transcriptome_X = transcriptome.X.to("cuda")

loss = torch.nn.MSELoss()

# %%
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (8, 4))
# ax1.scatter(
#     y = expression_predicted.detach().cpu().numpy().mean(0),
#     x = transcriptome_subset.detach().cpu().numpy().mean(0),
#     lw = 0,
#     s = 1
# )
# ax1.set_xlabel("Observed mean")
# ax1.set_ylabel("Predicted mean")
# ax2.scatter(
#     y = expression_predicted.detach().cpu().numpy().flatten(),
#     x = transcriptome_subset.detach().cpu().numpy().flatten(),
#     lw = 0,
#     s = 1
# )
# ax2.set_xlabel("Observed")
# ax2.set_ylabel("Predicted")

# %%
scores = []
expression_prediction = pd.DataFrame(0., index = transcriptome.obs.index, columns = transcriptome.var.index)
expression_observed = pd.DataFrame(transcriptome.X.dense().cpu().detach().numpy(), index = transcriptome.obs.index, columns = transcriptome.var.index)

for split in tqdm.tqdm(splits):
    expression_predicted = model(
        split.fragments_coordinates,
        split.fragment_cellxgene_idx,
        split.cell_n,
        split.gene_n,
        split.gene_idx
    )

    transcriptome_subset = transcriptome_X.dense_subset(split.cell_idx)[:, split.gene_idx]

    mse = loss(expression_predicted, transcriptome_subset)
    
    expression_predicted_dummy = transcriptome_subset.mean(0, keepdim = True).expand(transcriptome_subset.shape)
    mse_dummy = loss(expression_predicted_dummy, transcriptome_subset)
    
    transcriptome.obs.loc[transcriptome.obs.index[split.cell_idx], "phase"] = split.phase
    
    genescores = pd.DataFrame({
        "mse":((expression_predicted - transcriptome_subset)**2).mean(0).detach().cpu().numpy(),
        "gene":transcriptome.var.index[np.arange(split.gene_idx.start, split.gene_idx.stop)],
        "mse_dummy":((expression_predicted_dummy - transcriptome_subset)**2).mean(0).detach().cpu().numpy(),
    })

    scores.append({
        "mse":float(mse.detach().cpu().numpy()),
        "phase":split.phase,
        "genescores":genescores,
        "mse_dummy":float(mse_dummy.detach().cpu().numpy()),
    })
    
    expression_prediction.values[split.cell_idx, split.gene_idx] = expression_predicted.cpu().detach().numpy()
scores = pd.DataFrame(scores)

# %%
cells_oi = np.random.choice(expression_prediction.index, size = 100, replace = False)
plotdata = pd.DataFrame({"prediction":expression_prediction.loc[cells_oi].stack()})
plotdata["observed"] = expression_observed.loc[cells_oi].stack().values
plotdata = plotdata.loc[(plotdata["observed"] > 0) & (plotdata["prediction"] > 0)]

# %%
plotdata["phase"] = transcriptome.obs["phase"].loc[plotdata.index.get_level_values("cell")].values

# %%
sns.scatterplot(x = "prediction", y = "observed", hue = "phase", data = plotdata, s = 1)

# %% [markdown]
# ### Global view

# %%
scores["phase"] = scores["phase"].astype("category")

# %%
scores.groupby("phase").mean().assign(mse_diff = lambda x:x.mse_dummy - x.mse)

# %%
scores_models = scores.set_index(["phase"])[["mse", "mse_dummy"]]
scores_models.columns.name = "model"
scores_models = scores_models.unstack().rename("mse").reset_index()

# %%
sns.boxplot(x = "mse", y = "phase", hue = "model", data = scores_models)
# sns.boxplot(x = scores["mse_dummy"], y = scores["phase"])

# %% [markdown]
# ### Gene-specific view

# %%
def explode_dataframe(df, column):
    df2 = pd.concat([
        y[column].assign(**y[[col for col in y.index if (col != column) and (col not in y[column].columns)]]) for _, y in df.iterrows()
    ])
    return df2


# %%
gene_scores = explode_dataframe(scores, "genescores").groupby(["phase", "gene"]).mean().reset_index()

# %%
gene_scores_grouped = gene_scores.groupby(["phase", "gene"]).mean()
gene_scores_grouped["mse_diff"] = gene_scores_grouped["mse"] - gene_scores_grouped["mse_dummy"]
gene_scores_grouped["mse_diff"].unstack().T.sort_values("test").plot()

# %%
gene_scores_grouped["symbol"] = transcriptome.symbol(gene_scores_grouped.index.get_level_values("gene")).values

# %%
gene_scores_grouped.loc["test"].sort_values("mse_diff").head(20)

# %%
gene_mse = gene_scores_grouped["mse"]

# %%
sns.boxplot(x = gene_scores["phase"], y = gene_scores["mse"])

# %% [markdown]
# ## Performance when masking a window

# %%
splits = [split.to("cuda") for split in splits]

# %%
padding_positive = 2000
padding_negative = 4000


# %%
def select_window(coordinates, window_start, window_end):
    return ~((coordinates[:, 0] < window_end) & (coordinates[:, 1] > window_start))
assert (select_window(np.array([[-100, 200], [-300, -100]]), -50, 50) == np.array([False, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -310, -309) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), 201, 202) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -200, 20) == np.array([False, False])).all()

# %%
window_size = 100
cuts = np.arange(-padding_negative, padding_positive, step = window_size)

# %%
scores_windows = []
for split in tqdm.tqdm(splits):
    for window_idx, (window_start, window_end) in enumerate(zip(cuts[:-1], cuts[1:])):
        # take fragments within the window
        fragments_oi = select_window(split.fragments_coordinates, window_start, window_end)
        
        # calculate how much is retained overall
        perc_retained = fragments_oi.float().mean().detach().item()
        
        # calculate how much is retained per gene
        # scatter is needed here because the values are not sorted by gene (but by cellxgene)
        perc_retained_gene = torch_scatter.scatter_mean(fragments_oi.float().to("cpu"), split.local_gene_idx.to("cpu"), dim_size = split.gene_n)
        
        expression_predicted = model(
            split.fragments_coordinates[fragments_oi],
            split.fragment_cellxgene_idx[fragments_oi],
            split.cell_n,
            split.gene_n,
            split.gene_idx
        )

        transcriptome_subset = transcriptome_X.dense_subset(split.cell_idx)[:, split.gene_idx]

        mse = loss(expression_predicted, transcriptome_subset)

        expression_predicted_dummy = transcriptome_subset.mean(0, keepdim = True).expand(transcriptome_subset.shape)
        mse_dummy = loss(expression_predicted_dummy, transcriptome_subset)
        
        expression_predicted_mean = expression_predicted.mean()
        
        genescores = pd.DataFrame({
            "mse":((expression_predicted - transcriptome_subset)**2).mean(0).detach().cpu().numpy(),
            "mse_dummy":((expression_predicted_dummy - transcriptome_subset)**2).mean(0).detach().cpu().numpy(),
            "gene": transcriptome.var.index[np.arange(split.gene_idx.start, split.gene_idx.stop)],
            "perc_retained":perc_retained_gene.detach().cpu().numpy(),
            "expression_predicted":expression_predicted_mean.detach().cpu().numpy()
        })

        scores_windows.append({
            "mse":mse.detach().cpu().numpy(),
            "mse_dummy":mse_dummy.detach().cpu().numpy(),
            "phase":split.phase,
            "window_mid":window_start + (window_end - window_start)/2,
            "perc_retained":perc_retained,
            "genescores":genescores
        })
scores_windows = pd.DataFrame(scores_windows)

# %% [markdown]
# ### Global view

# %%
perc_retained_windows = scores_windows.groupby(["window_mid", "phase"])["perc_retained"].mean().unstack()

perc_retained_windows["train"].plot()

# %%
mse = scores.groupby("phase")["mse"].mean()
mse_windows = scores_windows.groupby(["window_mid", "phase"])["mse"].mean().unstack()
mse_dummy_windows = scores_windows.groupby(["window_mid", "phase"])["mse_dummy"].mean().unstack()

# %%
fig, ax_mse = plt.subplots()
patch_train = ax_mse.plot(mse_windows.index, mse_windows["train"], color = "blue", label = "train")
ax_mse.plot(mse_dummy_windows.index, mse_dummy_windows["train"], color = "blue", alpha = 0.1)
ax_mse.axhline(mse["train"], dashes = (2, 2), color = "blue")

ax_mse2 = ax_mse.twinx()

patch_test = ax_mse2.plot(mse_windows.index, mse_windows["test"], color = "red", label = "test")
ax_mse2.plot(mse_dummy_windows.index, mse_dummy_windows["test"], color = "red", alpha = 0.1)
ax_mse2.axhline(mse["test"], color = "red", dashes = (2, 2))

ax_mse.set_ylabel("MSE train", rotation = 0, ha = "right", color = "blue")
ax_mse2.set_ylabel("MSE test", rotation = 0, ha = "left", color = "red")
ax_mse.axvline(0, color = "#33333366", lw = 1)

plt.legend([patch_train[0], patch_test[0]], ['train', 'test'])

# %% [markdown]
# ### Gene-specific view

# %%
gene_scores_windows = explode_dataframe(scores_windows, "genescores").groupby(["gene", "phase", "window_mid"]).mean().reset_index()

# %%
gene_mse_windows = gene_scores_windows.groupby(["phase", "gene", "window_mid"])["mse"].first().unstack()
gene_perc_retained_windows = gene_scores_windows.groupby(["phase", "gene", "window_mid"])["perc_retained"].first().unstack()

# %%
gene_mse_dummy_windows = gene_scores_windows.groupby(["phase", "gene", "window_mid"])["mse_dummy"].first().unstack()

# %%
fig, ax = plt.subplots()
ax.hist(gene_mse_windows.loc["train"].idxmax(1), bins = cuts, histtype = "stepfilled", alpha = 0.5, color = "blue")
ax.hist(gene_mse_windows.loc["test"].idxmax(1), bins = cuts, histtype = "stepfilled", alpha = 0.8, color = "red")
fig.suptitle("Most important window across genes")
# .plot(kind = "hist", bins = len(cuts)-1, alpha = 0.5, zorder = 10)
# gene_mse_windows.loc["test"].idxmax(1).plot(kind = "hist", bins = len(cuts)-1)
None

# %%
gene_mse_windows_notss = gene_mse_windows.loc[:, (cuts[:-1] < -1000) | (cuts[:-1] > 1000)]

# %%
fig, ax = plt.subplots()
ax.hist(gene_mse_windows_notss.loc["train"].idxmax(1), bins = cuts, histtype = "stepfilled", alpha = 0.5, color = "blue")
ax.hist(gene_mse_windows_notss.loc["test"].idxmax(1), bins = cuts, histtype = "stepfilled", alpha = 0.8, color = "red")
fig.suptitle("Most important window across genes outside of TSS")

# %%
gene_mse_windows_norm = (gene_mse_windows - gene_mse_windows.values.min(1, keepdims = True)) / (gene_mse_windows.values.max(1, keepdims = True) - gene_mse_windows.values.min(1, keepdims = True))

# %%
sns.heatmap(gene_mse_windows_norm)

# %%
gene_id = transcriptome.gene_id("LYN")

# %% [markdown]
# Extract promoter info of gene

# %%
promoters = pd.read_csv(folder_data_preproc / "promoters.csv", index_col = 0)

# %%
promoter = promoters.loc[gene_id]

# %%
import pybedtools
promoter_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame(promoter).T[["chr", "start", "end"]])
peaks_bed = pybedtools.BedTool(folder_data_preproc / "atac_peaks.bed")

# %%
peaks_oi = promoter_bed.intersect(peaks_bed).to_dataframe()
if peaks_oi.shape[0] == 0:
    peaks_oi = pd.DataFrame(columns = ["start", "end"])
else:
    peaks_oi[["start", "end"]] = [
        [
            (peak["start"] - promoter["tss"]) * promoter["strand"],
            (peak["end"] - promoter["tss"]) * promoter["strand"]
        ][::promoter["strand"]]

        for _, peak in peaks_oi.iterrows()
    ]

# %% [markdown]
# Extract bigwig info of gene

# %%
import pyBigWig
bw = pyBigWig.open(str(folder_data_preproc / "atac_cut_sites.bigwig"))

# %%
fig, (ax_mse, ax_perc, ax_peak, ax_bw) = plt.subplots(4, 1, height_ratios = [1, 0.5, 0.2, 0.2], sharex=True)
ax_mse2 = ax_mse.twinx()

# mse annot
ax_mse.set_ylabel("MSE train", rotation = 0, ha = "right", color = "blue")
ax_mse2.set_ylabel("MSE test", rotation = 0, ha = "left", color = "red")
ax_mse.axvline(0, color = "#33333366", lw = 1)

# dummy mse
show_dummy = False
if show_dummy:
    plotdata = gene_mse_dummy_windows.loc["train"].loc[gene_id]
    ax_mse.plot(plotdata.index, plotdata, color = "blue", alpha = 0.1)
    plotdata = gene_mse_dummy_windows.loc["test"].loc[gene_id]
    ax_mse2.plot(plotdata.index, plotdata, color = "red", alpha = 0.1)

# unperturbed mse
ax_mse.axhline(gene_mse["train"][gene_id], dashes = (2, 2), color = "blue")
ax_mse2.axhline(gene_mse["test"][gene_id], dashes = (2, 2), color = "red")

# mse
plotdata = gene_mse_windows.loc["train"].loc[gene_id]
patch_train = ax_mse.plot(plotdata.index, plotdata, color = "blue", label = "train")
plotdata = gene_mse_windows.loc["test"].loc[gene_id]
patch_test = ax_mse2.plot(plotdata.index, plotdata, color = "red", label = "train")

# perc_retained
plotdata = gene_perc_retained_windows.loc["train"].loc[gene_id]
ax_perc.plot(plotdata.index, plotdata, color = "blue", label = "train")
plotdata = gene_perc_retained_windows.loc["test"].loc[gene_id]
ax_perc.plot(plotdata.index, plotdata, color = "red", label = "train")

ax_perc.axvline(0, color = "#33333366", lw = 1)

ax_perc.set_ylabel("Fragments\nretained", rotation = 0, ha = "right")
ax_perc.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax_perc.set_ylim([1, ax_perc.get_ylim()[0]])

# peaks
for _, peak in peaks_oi.iterrows():
    rect = mpl.patches.Rectangle((peak["start"], 0), peak["end"] - peak["start"], 1)
    ax_peak.add_patch(rect)
ax_peak.set_yticks([])
ax_peak.set_ylabel("Peaks", rotation = 0, ha = "right", va = "center")

# bw
ax_bw.plot(
    np.arange(promoter["start"] - promoter["tss"], promoter["end"] - promoter["tss"]) * promoter["strand"],
    bw.values(promoter["chr"], promoter["start"], promoter["end"])
)
ax_bw.set_ylabel("Smoothed\nfragments", rotation = 0, ha = "right", va = "center")
ax_bw.set_ylim(0)

# legend
ax_mse.legend([patch_train[0], patch_test[0]], ['train', 'test'])
# ax_mse.set_xlim(*ax_mse.get_xlim()[::-1])
fig.suptitle(transcriptome.symbol(gene_id) + " promoter")

# %%
# if you want to explore this window in igv
import IPython.display
IPython.display.HTML("<textarea>" + pfa.utils.name_window(promoters.loc[gene_id]) + "</textarea>")

# %% [markdown]
# ## Performance when masking a pairs of windows

# %%
splits = [split.to("cuda") for split in splits]

# %%
padding_positive = 2000
padding_negative = 4000


# %%
def select_window(coordinates, window_start, window_end):
    return ~((coordinates[:, 0] < window_end) & (coordinates[:, 1] > window_start))
assert (select_window(np.array([[-100, 200], [-300, -100]]), -50, 50) == np.array([False, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -310, -309) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), 201, 202) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -200, 20) == np.array([False, False])).all()

# %%
window_size = 1000
cuts = np.arange(-padding_negative, padding_positive, step = window_size)

# %%
scores_windowpairs = []
for split in tqdm.tqdm(splits):
    for window_idx, (window_start, window_end) in enumerate(zip(cuts[:-1], cuts[1:])):
        for window_idx, (window_start2, window_end2) in enumerate(zip(cuts[:-1], cuts[1:])):
            # take fragments within the window
            fragments_oi1 = select_window(split.fragments_coordinates, window_start, window_end)
            fragments_oi2 = select_window(split.fragments_coordinates, window_start2, window_end2)
            
            fragments_oi = fragments_oi1 & fragments_oi2

            # calculate how much is retained overall
            perc_retained = fragments_oi.float().mean().detach().item()

            # calculate how much is retained per gene
            # scatter is needed here because the values are not sorted by gene (but by cellxgene)
            perc_retained_gene = torch_scatter.scatter_mean(fragments_oi.float().to("cpu"), split.local_gene_idx.to("cpu"), dim_size = split.gene_n)

            expression_predicted = model(
                split.fragments_coordinates[fragments_oi],
                split.fragment_cellxgene_idx[fragments_oi],
                split.cell_n,
                split.gene_n,
                split.gene_idx
            )

            ##

            transcriptome_subset = transcriptome_X.dense_subset(split.cell_idx)[:, split.gene_idx]

            ##

            mse = loss(expression_predicted, transcriptome_subset)

            expression_predicted_dummy = transcriptome_subset.mean(0, keepdim = True).expand(transcriptome_subset.shape)
            mse_dummy = loss(expression_predicted_dummy, transcriptome_subset)
            
            ##
            
            exp_predicted = expression_predicted.mean()

            ##
            genescores = pd.DataFrame({
                "mse":((expression_predicted - transcriptome_subset)**2).mean(0).detach().cpu().numpy(),
                "mse_dummy":((expression_predicted_dummy - transcriptome_subset)**2).mean(0).detach().cpu().numpy(),
                "gene": transcriptome.var.index[np.arange(split.gene_idx.start, split.gene_idx.stop)],
                "perc_retained":perc_retained_gene.detach().cpu().numpy(),
                "exp_predicted":exp_predicted.detach().cpu().numpy()
            })
            
            score = {
                "mse":mse.detach().cpu().numpy(),
                "mse_dummy":mse_dummy.detach().cpu().numpy(),
                "phase":split.phase,
                "window_mid1":window_start + (window_end - window_start)/2,
                "window_mid2":window_start2 + (window_end2 - window_start2)/2,
                "perc_retained":perc_retained,
                "genescores":genescores
            }
            
            scores_windowpairs.append(score)
scores_windowpairs = pd.DataFrame(scores_windowpairs)

# %% [markdown]
# ### Global view

# %%
perc_retained_windowpairs = scores_windowpairs.groupby(["window_mid1", "window_mid2", "phase"])["perc_retained"].mean().unstack()

perc_retained_windowpairs["train"].plot()

# %%
mse = scores.groupby("phase")["mse"].mean()
mse_windowpairs = scores_windowpairs.groupby(["window_mid1", "window_mid2", "phase"])["mse"].mean().unstack()
mse_dummy_windowpairs = scores_windowpairs.groupby(["window_mid1", "window_mid2", "phase"])["mse_dummy"].mean().unstack()

# %%
sns.heatmap(mse_windowpairs["test"].unstack())

# %% [markdown]
# ### Gene-specific view

# %%
gene_scores_windowpairs = explode_dataframe(scores_windowpairs, "genescores").groupby(["gene", "phase", "window_mid1", "window_mid2"]).mean().reset_index()

# %%
gene_mse_windowpairs = gene_scores_windowpairs.groupby(["phase", "gene", "window_mid1", "window_mid2"])["mse"].first().unstack()
gene_perc_retained_windowpairs = gene_scores_windowpairs.groupby(["phase", "gene", "window_mid1", "window_mid2"])["perc_retained"].first().unstack()

# %%
gene_mse_dummy_windowpairs = gene_scores_windowpairs.groupby(["phase", "gene", "window_mid1", "window_mid2"])["mse_dummy"].first().unstack()

# %%
gene_id = transcriptome.gene_id("FOXP1")

# %%
plotdata = gene_mse_windowpairs.loc["test"].loc[gene_id]

# %%
sns.heatmap(gene_mse_windowpairs.loc["test"].loc[gene_id], vmin = gene_mse.loc["test"].loc[gene_id])

# %% [markdown]
# ## Performance when masking peaks

# %% [markdown]
# ## Performance when removing fragment lengths

# %%
splits = [split.to("cuda") for split in splits]

# %%
# cuts = [200, 400, 600]
# cuts = list(np.arange(0, 200, 10)) + list(np.arange(200, 1000, 50))
cuts = list(np.arange(0, 1000, 10))
windows = [[cut0, cut1] for cut0, cut1 in zip(cuts, cuts[1:] + [9999999])]

# %%
window_size = 1000

scores_lengths = []
for split in tqdm.tqdm(splits):
    for window_idx, (window_start, window_end) in enumerate(windows):
        # take fragments within the window
        fragment_lengths = (split.fragments_coordinates[:,1] - split.fragments_coordinates[:,0])
        fragments_oi = ~((fragment_lengths >= window_start) & (fragment_lengths < window_end))
        
        # calculate how much is retained overall
        perc_retained = fragments_oi.float().mean().detach().item()
        
        # calculate how much is retained per gene
        # scatter is needed here because the values are not sorted by gene (but by cellxgene)
        perc_retained_gene = torch_scatter.scatter_mean(fragments_oi.float().to("cpu"), split.local_gene_idx.to("cpu"), dim_size = split.gene_n)
        
        expression_predicted = model(
            split.fragments_coordinates[fragments_oi],
            split.fragment_cellxgene_idx[fragments_oi],
            split.cell_n,
            split.gene_n,
            split.gene_idx
        )

        ##

        transcriptome_subset = transcriptome_X.dense_subset(split.cell_idx)[:, split.gene_idx]

        ##

        mse = loss(expression_predicted, transcriptome_subset)
        
        expression_predicted_dummy = transcriptome_subset.mean(0, keepdim = True).expand(transcriptome_subset.shape)
        mse_dummy = loss(expression_predicted_dummy, transcriptome_subset)
        
        ##
        genescores = pd.DataFrame({
            "mse":((expression_predicted - transcriptome_subset)**2).mean(0).detach().cpu().numpy(),
            "mse_dummy":((expression_predicted_dummy - transcriptome_subset)**2).mean(0).detach().cpu().numpy(),
            "gene": transcriptome.var.index[np.arange(split.gene_idx.start, split.gene_idx.stop)],
            "perc_retained":perc_retained_gene.detach().cpu().numpy()
        })

        scores_lengths.append({
            "mse":mse.detach().cpu().numpy(),
            "mse_dummy":mse_dummy.detach().cpu().numpy(),
            "phase":split.phase,
            "window_start":window_start,
            "perc_retained":perc_retained,
            "genescores":genescores
        })
scores_lengths = pd.DataFrame(scores_lengths)

# %% [markdown]
# ### Global view

# %%
mse = scores.groupby("phase")["mse"].mean()

mse_lengths = scores_lengths.groupby(["window_start", "phase"])["mse"].mean().unstack()
mse_dummy_lengths = scores_lengths.groupby(["window_start", "phase"])["mse_dummy"].mean().unstack()

# %%
perc_retained_lengths = scores_lengths.groupby(["window_start", "phase"])["perc_retained"].mean().unstack()

# %%
fig, ax_perc = plt.subplots()
ax_mse = ax_perc.twinx()
ax_mse.plot(mse_lengths.index, mse_lengths["test"])
ax_mse.axhline(mse["test"], color = "red", dashes = (2, 2))
ax_mse.set_ylabel("MSE", rotation = 0, ha = "left", va = "center", color = "blue")
# ax_mse.plot(mse_dummy_lengths.index, mse_dummy_lengths["test"]) 

ax_perc.plot(perc_retained_lengths.index, perc_retained_lengths["test"], color = "#33333344")
ax_perc.set_ylim(ax_perc.set_ylim()[::-1])
ax_perc.set_ylabel("Fragments\nretained", rotation = 0, ha = "right", va = "center", color = "#33333344")
ax_perc.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))

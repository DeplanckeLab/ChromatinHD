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

dataset_name = "lymphoma"
# dataset_name = "pbmc10k"
folder_data_preproc = folder_data / dataset_name

# %%
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")
fragments = peakfreeatac.fragments.Fragments(folder_data_preproc / "fragments")


# %%
class Prediction(pfa.flow.Flow):
    pass
prediction = Prediction(pfa.get_output() / "prediction_promoter" / dataset_name / "nn")

# %%
splits = pickle.load(open(prediction.path / "splits.pkl", "rb"))
model = pickle.load(open(prediction.path / "model.pkl", "rb"))

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

for split_ix, split in enumerate(tqdm.tqdm(splits)):
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
        "split_ix":split_ix
    })

    scores.append({
        "mse":float(mse.detach().cpu().numpy()),
        "phase":split.phase,
        "genescores":genescores,
        "mse_dummy":float(mse_dummy.detach().cpu().numpy()),
        "split_ix":split_ix
    })
    
    expression_prediction.values[split.cell_idx, split.gene_idx] = expression_predicted.cpu().detach().numpy()
scores = pd.DataFrame(scores)

# %% [markdown]
# ## Global visual check

# %%
cells_oi = np.random.choice(expression_prediction.index, size = 100, replace = False) # only subselect 100 cells
plotdata = pd.DataFrame({"prediction":expression_prediction.loc[cells_oi].stack()})
plotdata["observed"] = expression_observed.loc[cells_oi].stack().values

print(np.corrcoef(plotdata["prediction"], plotdata["observed"])[0, 1])

plotdata = plotdata.loc[(plotdata["observed"] > 0) & (plotdata["prediction"] > 0)] # only take cells with both values > 0

print(np.corrcoef(plotdata["prediction"], plotdata["observed"])[0, 1])

# %%
plotdata["phase"] = transcriptome.obs["phase"].loc[plotdata.index.get_level_values("cell")].values

# %%
fig, (ax_train, ax_validation) = plt.subplots(1, 2)

plotdata_train = plotdata.query("phase == 'train'")
sns.scatterplot(x = "prediction", y = "observed", color = "blue", data = plotdata_train, s = 1, ax = ax_train)
correlation = np.corrcoef(plotdata_train["prediction"], plotdata_train["observed"])[0, 1]
ax_train.annotate(f"{correlation:.2f}", (0.9, 0.9), xycoords = "axes fraction", ha = "right", va = "top")
ax_train.set_title("Training")

plotdata_validation = plotdata.query("phase == 'validation'")
sns.scatterplot(x = "prediction", y = "observed", color = "red", data = plotdata_validation, s = 1, ax = ax_validation)
correlation = np.corrcoef(plotdata_validation["prediction"], plotdata_validation["observed"])[0, 1]
ax_validation.annotate(f"{correlation:.2f}", (0.9, 0.9), xycoords = "axes fraction", ha = "right", va = "top")
ax_validation.set_title("Validation")

# %% [markdown]
# ### Global view

# %%
scores["phase"] = scores["phase"].astype("category")

# %%
splitinfo = pd.DataFrame({"split_ix":i, "n_cells":split.cell_n, "phase":split.phase} for i, split in enumerate(splits))
splitinfo["weight"] = splitinfo["n_cells"] / splitinfo.groupby("phase")["n_cells"].sum()[splitinfo["phase"]].values

# %%
col = "mse"


# %%
def aggregate_splits(df, columns = ("mse", "mse_dummy")):
    """
    Calculates the weighted mean of certain columns according to the size (# cells) of each split
    Requires a grouped dataframe as input
    """
    assert "split_ix" in df.obj.columns
    assert isinstance(df, pd.core.groupby.generic.DataFrameGroupBy)
    for col in columns:
        df.obj[col + "_weighted"] = df.obj[col] * splitinfo.loc[df.obj["split_ix"], "weight"].values
    
    df2 = df[[col + "_weighted" for col in columns]].sum()
    df2.columns = columns
    return df2


# %%
# aggscores = scores.groupby("phase").apply(aggregate_split)
aggscores = aggregate_splits(scores.groupby("phase"))

# %%
scores_models = scores.set_index(["phase"])[["mse", "mse_dummy"]]
scores_models.columns.name = "model"
scores_models = scores_models.unstack().rename("mse").reset_index()

# %%
sns.boxplot(x = "mse", y = "phase", hue = "model", data = scores_models)
# sns.boxplot(x = scores["mse_dummy"], y = scores["phase"])

# %%
aggscores.to_csv(prediction.path / "aggscores.csv")


# %% [markdown]
# ### Gene-specific view

# %%
def explode_dataframe(df, column):
    df2 = pd.concat([
        y[column].assign(**y[[col for col in y.index if (col != column) and (col not in y[column].columns)]]) for _, y in df.iterrows()
    ])
    return df2


# %%
gene_aggscores = aggregate_splits(explode_dataframe(scores, "genescores").groupby(["phase", "gene"]))
gene_aggscores["mse_diff"] = gene_aggscores["mse"]  - gene_aggscores["mse_dummy"]

# %%
gene_aggscores["mse_diff"].unstack().T.sort_values("validation").plot()

# %%
gene_aggscores["symbol"] = transcriptome.symbol(gene_aggscores.index.get_level_values("gene")).values

# %%
gene_aggscores.loc["validation"].sort_values("mse_diff", ascending = True).head(20)

# %%
gene_aggscores.to_csv(prediction.path / "gene_aggscores.csv")

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
for split_ix, split in enumerate(tqdm.tqdm(splits)):
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
            "expression_predicted":expression_predicted_mean.detach().cpu().numpy(),
            "split_ix":split_ix
        })

        scores_windows.append({
            "mse":mse.detach().cpu().numpy().item(),
            "mse_dummy":mse_dummy.detach().cpu().numpy().item(),
            "phase":split.phase,
            "window_mid":window_start + (window_end - window_start)/2,
            "perc_retained":perc_retained,
            "genescores":genescores,
            "split_ix":split_ix
        })
scores_windows = pd.DataFrame(scores_windows)

# %% [markdown]
# ### Global view

# %%
aggrscores_windows = aggregate_splits(scores_windows.groupby(["phase", "window_mid"]), columns = ["perc_retained", "mse", "mse_dummy"])

# %%
aggrscores_windows.loc["train"]["perc_retained"].plot()
aggrscores_windows.loc["validation"]["perc_retained"].plot()

# %%
mse_windows = aggrscores_windows["mse"].unstack().T
mse_dummy_windows = aggrscores_windows["mse_dummy"].unstack().T

# %%
fig, ax_mse = plt.subplots()
patch_train = ax_mse.plot(mse_windows.index, mse_windows["train"], color = "blue", label = "train")
ax_mse.plot(mse_dummy_windows.index, mse_dummy_windows["train"], color = "blue", alpha = 0.1)
ax_mse.axhline(aggscores.loc["train", "mse"], dashes = (2, 2), color = "blue")

ax_mse2 = ax_mse.twinx()

patch_validation = ax_mse2.plot(mse_windows.index, mse_windows["validation"], color = "red", label = "validation")
ax_mse2.plot(mse_dummy_windows.index, mse_dummy_windows["validation"], color = "red", alpha = 0.1)
ax_mse2.axhline(aggscores.loc["validation", "mse"], color = "red", dashes = (2, 2))

ax_mse.set_ylabel("MSE train", rotation = 0, ha = "right", color = "blue")
ax_mse2.set_ylabel("MSE validation", rotation = 0, ha = "left", color = "red")
ax_mse.axvline(0, color = "#33333366", lw = 1)

plt.legend([patch_train[0], patch_validation[0]], ['train', 'validation'])

# %% [markdown]
# ### Gene-specific view

# %%
gene_aggscores_windows = scores_windows.pipe(explode_dataframe, "genescores").groupby(["phase", "gene", "window_mid"]).pipe(aggregate_splits, columns = ["perc_retained", "mse", "mse_dummy"])

# %%
gene_mse_windows = gene_aggscores_windows["mse"].unstack()
gene_perc_retained_windows = gene_aggscores_windows["perc_retained"].unstack()
gene_mse_dummy_windows = gene_aggscores_windows["mse_dummy"].unstack()

# %%
fig, ax = plt.subplots()
ax.hist(gene_mse_windows.loc["train"].idxmax(1), bins = cuts, histtype = "stepfilled", alpha = 0.5, color = "blue")
ax.hist(gene_mse_windows.loc["validation"].idxmax(1), bins = cuts, histtype = "stepfilled", alpha = 0.8, color = "red")
fig.suptitle("Most important window across genes")
# .plot(kind = "hist", bins = len(cuts)-1, alpha = 0.5, zorder = 10)
# gene_mse_windows.loc["validation"].idxmax(1).plot(kind = "hist", bins = len(cuts)-1)
None

# %%
gene_mse_windows_notss = gene_mse_windows.loc[:, (cuts[:-1] < -1000) | (cuts[:-1] > 1000)]

# %%
fig, ax = plt.subplots()
ax.hist(gene_mse_windows_notss.loc["train"].idxmax(1), bins = cuts, histtype = "stepfilled", alpha = 0.5, color = "blue")
ax.hist(gene_mse_windows_notss.loc["validation"].idxmax(1), bins = cuts, histtype = "stepfilled", alpha = 0.8, color = "red")
fig.suptitle("Most important window across genes outside of TSS")

# %%
gene_mse_windows_norm = (gene_mse_windows - gene_mse_windows.values.min(1, keepdims = True)) / (gene_mse_windows.values.max(1, keepdims = True) - gene_mse_windows.values.min(1, keepdims = True))

# %%
sns.heatmap(gene_mse_windows_norm)

# %%
gene_id = transcriptome.gene_id("PAX5")

# %% [markdown]
# Extract promoter info of gene

# %%
promoters = pd.read_csv(folder_data_preproc / "promoters.csv", index_col = 0)

# %%
promoter = promoters.loc[gene_id]


# %%
def center_peaks(peaks, promoter):
    if peaks.shape[0] == 0:
        peaks = pd.DataFrame(columns = ["start", "end", "method"])
    else:
        peaks[["start", "end"]] = [
            [
                (peak["start"] - promoter["tss"]) * promoter["strand"],
                (peak["end"] - promoter["tss"]) * promoter["strand"]
            ][::promoter["strand"]]

            for _, peak in peaks.iterrows()
        ]
    return peaks


# %%
# # !grep -v "KI270728.1" {folder_data_preproc}/atac_peaks.bed > {folder_data_preproc}/atac_peaks.bed

# %%
import pybedtools
promoter_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame(promoter).T[["chr", "start", "end"]])
peaks_bed = pybedtools.BedTool(folder_data_preproc / "atac_peaks.bed")
peaks_cellranger = promoter_bed.intersect(peaks_bed).to_dataframe()
peaks_cellranger["method"] = "cellranger"
peaks_cellranger = center_peaks(peaks_cellranger, promoter)

# %%
peaks_bed = pybedtools.BedTool(pfa.get_output() / "peaks" / "lymphoma" / "macs2" / "peaks.bed")
peaks_macs2 = promoter_bed.intersect(peaks_bed).to_dataframe()
peaks_macs2["method"] = "macs2"
peaks_macs2 = center_peaks(peaks_macs2, promoter)

# %%
peaks_bed = pybedtools.BedTool(pfa.get_output() / "peaks" / "lymphoma" / "genrich" / "peaks.bed")
peaks_genrich = promoter_bed.intersect(peaks_bed).to_dataframe()
peaks_genrich["method"] = "genrich"
peaks_genrich = center_peaks(peaks_genrich, promoter)

# %%
peaks = pd.concat([peaks_cellranger, peaks_macs2, peaks_genrich])

# %%
peak_methods = pd.DataFrame({"method":["macs2", "cellranger", "genrich"]}).set_index("method")
peak_methods["ix"] = np.arange(peak_methods.shape[0])

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
ax_mse2.set_ylabel("MSE validation", rotation = 0, ha = "left", color = "red")
ax_mse.axvline(0, color = "#33333366", lw = 1)
ax_mse.set_xlim(gene_mse_dummy_windows.columns[0], gene_mse_dummy_windows.columns[-1])

# dummy mse
show_dummy = False
if show_dummy:
    plotdata = gene_mse_dummy_windows.loc["train"].loc[gene_id]
    ax_mse.plot(plotdata.index, plotdata, color = "blue", alpha = 0.1)
    plotdata = gene_mse_dummy_windows.loc["validation"].loc[gene_id]
    ax_mse2.plot(plotdata.index, plotdata, color = "red", alpha = 0.1)

# unperturbed mse
ax_mse.axhline(gene_aggscores.loc[("train", gene_id), "mse"], dashes = (2, 2), color = "blue")
ax_mse2.axhline(gene_aggscores.loc[("validation", gene_id), "mse"], dashes = (2, 2), color = "red")

# mse
plotdata = gene_mse_windows.loc["train"].loc[gene_id]
patch_train = ax_mse.plot(plotdata.index, plotdata, color = "blue", label = "train")
plotdata = gene_mse_windows.loc["validation"].loc[gene_id]
patch_validation = ax_mse2.plot(plotdata.index, plotdata, color = "red", label = "train")

# perc_retained
plotdata = gene_perc_retained_windows.loc["train"].loc[gene_id]
ax_perc.plot(plotdata.index, plotdata, color = "blue", label = "train")
plotdata = gene_perc_retained_windows.loc["validation"].loc[gene_id]
ax_perc.plot(plotdata.index, plotdata, color = "red", label = "train")

ax_perc.axvline(0, color = "#33333366", lw = 1)

ax_perc.set_ylabel("Fragments\nretained", rotation = 0, ha = "right")
ax_perc.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax_perc.set_ylim([1, ax_perc.get_ylim()[0]])

# peaks
for _, peak in peaks.iterrows():
    y = peak_methods.loc[peak["method"], "ix"]
    rect = mpl.patches.Rectangle((peak["start"], y), peak["end"] - peak["start"], 1)
    ax_peak.add_patch(rect)
ax_peak.set_ylim(0, peak_methods["ix"].max() + 1)
ax_peak.set_yticks(peak_methods["ix"] + 0.5)
ax_peak.set_yticklabels(peak_methods.index)
ax_peak.set_ylabel("Peaks", rotation = 0, ha = "right", va = "center")

# bw
ax_bw.plot(
    np.arange(promoter["start"] - promoter["tss"], promoter["end"] - promoter["tss"]) * promoter["strand"],
    bw.values(promoter["chr"], promoter["start"], promoter["end"])
)
ax_bw.set_ylabel("Smoothed\nfragments", rotation = 0, ha = "right", va = "center")
ax_bw.set_ylim(0)

# legend
ax_mse.legend([patch_train[0], patch_validation[0]], ['train', 'validation'])
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
for split_ix, split in enumerate(tqdm.tqdm(splits)):
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
                "exp_predicted":exp_predicted.detach().cpu().numpy(),
                "split_ix":split_ix
            })
            
            score = {
                "mse":mse.detach().cpu().numpy().item(),
                "mse_dummy":mse_dummy.detach().cpu().numpy().item(),
                "phase":split.phase,
                "window_mid1":window_start + (window_end - window_start)/2,
                "window_mid2":window_start2 + (window_end2 - window_start2)/2,
                "perc_retained":perc_retained,
                "genescores":genescores,
                "split_ix":split_ix
            }
            
            scores_windowpairs.append(score)
scores_windowpairs = pd.DataFrame(scores_windowpairs)

# %% [markdown]
# ### Global view

# %%
aggscores_windowpairs = scores_windowpairs.groupby(["phase", "window_mid1", "window_mid2"]).pipe(aggregate_splits, columns = ["perc_retained", "mse", "mse_dummy"])

aggscores_windowpairs.loc["train", "perc_retained"].plot()

# %%
mse_windowpairs = aggscores_windowpairs["mse"].unstack()
mse_dummy_windowpairs = aggscores_windowpairs["mse_dummy"].unstack()

# %%
sns.heatmap(mse_windowpairs.loc["validation"])

# %% [markdown]
# ### Gene-specific view

# %%
gene_aggscores_windowpairs = (
    scores_windowpairs
        .pipe(explode_dataframe, "genescores")
        .groupby(["phase", "gene", "window_mid2", "window_mid1"])
        .pipe(aggregate_splits, columns = ["perc_retained", "mse", "mse_dummy"])
)

# %%
gene_mse_windowpairs = gene_aggscores_windowpairs["mse"].unstack()
gene_mse_dummy_windowpairs = gene_aggscores_windowpairs["mse_dummy"].unstack()
gene_perc_retained_windowpairs = gene_aggscores_windowpairs["perc_retained"].unstack()

# %%
gene_id = transcriptome.gene_id("FOXP1")

# %%
plotdata = gene_mse_windowpairs.loc["validation"].loc[gene_id]

# %%
sns.heatmap(gene_mse_windowpairs.loc["validation"].loc[gene_id], vmin = gene_mse.loc["validation"].loc[gene_id])

# %% [markdown]
# ## Performance when masking peaks

# %% [markdown]
# ## Performance when removing fragment lengths

# %%
splits = [split.to("cuda") for split in splits]

# %%
# cuts = [200, 400, 600]
# cuts = list(np.arange(0, 200, 10)) + list(np.arange(200, 1000, 50))
cuts = list(np.arange(0, 1000, 25))
windows = [[cut0, cut1] for cut0, cut1 in zip(cuts, cuts[1:] + [9999999])]

# %%
scores_lengths = []
for split_ix, split in enumerate(tqdm.tqdm(splits)):
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
            "perc_retained":perc_retained_gene.detach().cpu().numpy(),
            "split_ix":split_ix
        })

        scores_lengths.append({
            "mse":mse.detach().cpu().numpy().item(),
            "mse_dummy":mse_dummy.detach().cpu().numpy().item(),
            "phase":split.phase,
            "window_start":window_start,
            "perc_retained":perc_retained,
            "genescores":genescores,
            "split_ix":split_ix
        })
scores_lengths = pd.DataFrame(scores_lengths)

# %% [markdown]
# ### Global view

# %%
aggrscores_lengths = aggregate_splits(scores_lengths.groupby(["phase", "window_start"]), columns = ["perc_retained", "mse", "mse_dummy"])

# %%
mse_lengths = aggrscores_lengths["mse"].unstack().T
mse_dummy_lengths = aggrscores_lengths["mse_dummy"].unstack().T
perc_retained_lengths = aggrscores_lengths["perc_retained"].unstack().T

# %%
fig, ax_perc = plt.subplots()
ax_mse = ax_perc.twinx()
ax_mse.plot(mse_lengths.index, mse_lengths["validation"])
ax_mse.axhline(aggscores.loc["validation", "mse"], color = "red", dashes = (2, 2))
ax_mse.set_ylabel("MSE", rotation = 0, ha = "left", va = "center", color = "blue")
# ax_mse.plot(mse_dummy_lengths.index, mse_dummy_lengths["validation"]) 

ax_perc.plot(perc_retained_lengths.index, perc_retained_lengths["validation"], color = "#33333344")
ax_perc.set_ylim(ax_perc.set_ylim()[::-1])
ax_perc.set_ylabel("Fragments\nretained", rotation = 0, ha = "right", va = "center", color = "#33333344")
ax_perc.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))

# %% [markdown]
# ## Visualize a gene fragments

# %%
padding_positive = 2000
padding_negative = 4000
lim = (-padding_negative, padding_positive)

# %%
gene_id = transcriptome.gene_id("IGKC")

# %%
sc.pl.umap(transcriptome.adata, color = [gene_id])

# %%
transcriptome.var.head(20)

# %%
gene_ix = fragments.var.loc[gene_id]["ix"]

# %%
coordinates = fragments.coordinates[fragments.mapping[:, 1] == gene_ix].numpy()
mapping = fragments.mapping[fragments.mapping[:, 1] == gene_ix].numpy()

# %%
n_cells = fragments.n_cells

cell_order = np.argsort(sc.get.obs_df(transcriptome.adata, gene_id))
obs = fragments.obs.copy().iloc[cell_order]
obs["y"] = np.arange(obs.shape[0])
obs = obs.set_index("ix")

# %%
fig, ax = plt.subplots(figsize = (4, n_cells/300))
ax.set_xlim(lim)
ax.set_ylim(0, n_cells)

for (start, end, cell_ix) in zip(coordinates[:, 0], coordinates[:, 1], mapping[:, 0]):
    rect = mpl.patches.Rectangle((start, obs.loc[cell_ix, "y"]), end - start, 10, fc = "black", ec = None)
    ax.add_patch(rect)

# %%

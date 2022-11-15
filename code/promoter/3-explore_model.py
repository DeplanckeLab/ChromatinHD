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

device = "cuda:0"

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
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")
fragments = peakfreeatac.fragments.Fragments(folder_data_preproc / "fragments")


# %%
class Prediction(pfa.flow.Flow):
    pass
prediction = Prediction(pfa.get_output() / "prediction_promoter" / dataset_name / "nn")

# %%
splits = pickle.load(open(prediction.path / "splits.pkl", "rb"))
model = pickle.load(open(prediction.path / "model.pkl", "rb")).to(device)

# %% [markdown]
# ## Overall performace

# %%
splits = [split.to(device) for split in splits]
transcriptome_X = transcriptome.X.to(device)


# %%
def aggregate_splits(df, columns = ("mse", "mse_dummy"), splitinfo = None):
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

def explode_dataframe(df, column):
    df2 = pd.concat([
        y[column].assign(**y[[col for col in y.index if (col != column) and (col not in y[column].columns)]]) for _, y in df.iterrows()
    ])
    return df2


# %%
transcriptome_pd = pd.DataFrame(transcriptome_X.dense().cpu().numpy(), index = transcriptome.obs.index, columns = transcriptome.var.index)


# %%
def score_fragments(
    splits,
    fragments_oi,
    expression_prediction_full = None,
    return_expression_prediction = False
):
    """
    Scores a set of fragments_oi using a set of splits
    """
    
    splitinfo = pd.DataFrame({"split_ix":i, "n_cells":split.cell_n, "phase":split.phase} for i, split in enumerate(splits))
    splitinfo["weight"] = splitinfo["n_cells"] / splitinfo.groupby("phase")["n_cells"].sum()[splitinfo["phase"]].values
    
    scores = []
    expression_prediction = pd.DataFrame(0., index = transcriptome.obs.index, columns = transcriptome.var.index)
    
    # run all the splits
    for split_ix, split in enumerate(splits):
        fragments_oi_split = fragments_oi[split.fragments_selected]
        
        # calculate how much is retained overall
        perc_retained = fragments_oi_split.float().mean().detach().item()
        
        # calculate how much is retained per gene
        # scatter is needed here because the values are not sorted by gene (but by cellxgene)
        perc_retained_gene = torch_scatter.scatter_mean(fragments_oi_split.float().to("cpu"), split.local_gene_idx.to("cpu"), dim_size = split.gene_n)
        
        # run the model and calculate mse
        with torch.no_grad():
            expression_predicted = model(
                split.fragments_coordinates[fragments_oi_split],
                split.fragment_cellxgene_idx[fragments_oi_split],
                split.cell_n,
                split.gene_n,
                split.gene_idx
            )#.to("cpu")

        transcriptome_subset = transcriptome_X.dense_subset(split.cell_idx)[:, split.gene_idx].to(device)

        mse = ((expression_predicted - transcriptome_subset)**2).mean()

        expression_predicted_dummy = transcriptome_subset.mean(0, keepdim = True).expand(transcriptome_subset.shape)
        mse_dummy = ((expression_predicted_dummy - transcriptome_subset)**2).mean()

        transcriptome.obs.loc[transcriptome.obs.index[split.cell_idx], "phase"] = split.phase

        genescores = pd.DataFrame({
            "mse":((expression_predicted - transcriptome_subset)**2).mean(0).detach().cpu().numpy(),
            "gene":transcriptome.var.index[np.arange(split.gene_idx.start, split.gene_idx.stop)],
            "mse_dummy":((expression_predicted_dummy - transcriptome_subset)**2).mean(0).detach().cpu().numpy(),
            "perc_retained":perc_retained_gene.detach().cpu().numpy(),
            "split_ix":split_ix
        })

        scores.append({
            "mse":float(mse.detach().cpu().numpy()),
            "phase":split.phase,
            "genescores":genescores,
            "mse_dummy":float(mse_dummy.detach().cpu().numpy()),
            "perc_retained":perc_retained,
            "split_ix":split_ix
        })

        expression_prediction.values[split.cell_idx, split.gene_idx] = expression_predicted.cpu().detach().numpy()
        
    # aggregate overall scores
    scores = pd.DataFrame(scores)
    scores["phase"] = scores["phase"].astype("category")
    aggscores = aggregate_splits(
        scores.groupby("phase"),
        splitinfo = splitinfo,
        columns = ["perc_retained", "mse", "mse_dummy"]
    )
    aggscores["mse_diff"] = aggscores["mse"] - aggscores["mse_dummy"]
    
    # aggregate gene scores
    gene_aggscores = (
        scores
            .pipe(explode_dataframe, "genescores")
            .groupby(["phase", "gene"])
            .pipe(aggregate_splits, columns = ["perc_retained", "mse", "mse_dummy"], splitinfo = splitinfo)
    )
    gene_aggscores["mse_diff"] = gene_aggscores["mse_dummy"] - gene_aggscores["mse"]
    
    # calculate summary statistics on the predicted expression
    # first extract train and validation cells
    cells_train = transcriptome.obs.index[list(set([cell_idx for split in splits for cell_idx in split.cell_idxs if split.phase == "train"]))]
    cells_validation = transcriptome.obs.index[list(set([cell_idx for split in splits for cell_idx in split.cell_idxs if split.phase == "validation"]))]
    
    # calculate effect
    if expression_prediction_full is not None:
        effect_train = expression_prediction.loc[cells_train].mean() - expression_prediction_full.loc[cells_train].mean()
        effect_validation = expression_prediction.loc[cells_validation].mean() - expression_prediction_full.loc[cells_validation].mean()
        
        aggscores["effect"] = pd.Series({"train":effect_train.mean(), "validation":effect_validation.mean()})
        gene_aggscores["effect"] = pd.concat({"train":effect_train, "validation":effect_validation}, names = ["phase", "gene"])
    
    # calculate correlation
    cor_train = pfa.utils.paircor(expression_prediction.loc[cells_train], transcriptome_pd.loc[cells_train])
    cor_validation = pfa.utils.paircor(expression_prediction.loc[cells_validation], transcriptome_pd.loc[cells_validation])
    
    gene_aggscores["cor"] = pd.concat({"train":cor_train, "validation":cor_validation}, names = ["phase", "gene"])
    aggscores["cor"] = pd.Series({"train":cor_train.mean(), "validation":cor_validation.mean()})
    
    if return_expression_prediction:
        return aggscores, gene_aggscores, expression_prediction
    return aggscores, gene_aggscores


# %%
fragments_oi = torch.tensor([True] * fragments.coordinates.shape[0], device = device)
aggscores, gene_aggscores, expression_prediction = score_fragments(splits, fragments_oi, return_expression_prediction = True)

# %% [markdown]
# ### Global visual check

# %%
cells_oi = np.random.choice(expression_prediction.index, size = 100, replace = False) # only subselect 100 cells
plotdata = pd.DataFrame({"prediction":expression_prediction.loc[cells_oi].stack()})
plotdata["observed"] = transcriptome_pd.loc[cells_oi].stack().values

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
aggscores.style.bar()

# %%
aggscores.to_csv(prediction.path / "aggscores.csv")

# %% [markdown]
# ### Gene-specific view

# %%
gene_aggscores["symbol"] = transcriptome.symbol(gene_aggscores.index.get_level_values("gene")).values

# %%
gene_aggscores.loc["validation"].sort_values("mse_diff", ascending = False).head(20).style.bar(subset = ["mse_diff", "cor"])

# %%
gene_aggscores["mse_diff"].unstack().T.sort_values("validation").plot()

# %%
gene_aggscores["cor"].unstack().T.sort_values("validation").plot()

# %%
gene_aggscores.to_csv(prediction.path / "gene_aggscores.csv")

# %% [markdown]
# ## Performance when masking a window

# %% [markdown]
# Hypothesis: are fragments from certain regions more predictive than others?

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
aggscores_windows = []
gene_aggscores_windows = []
for window_idx, (window_start, window_end) in tqdm.tqdm(enumerate(zip(cuts[:-1], cuts[1:])), total = len(cuts)-1):
    # take fragments within the window
    fragments_oi = select_window(fragments.coordinates, window_start, window_end)
    
    aggscores_window, gene_aggscores_window = score_fragments(splits, fragments_oi, expression_prediction_full=expression_prediction)
    
    window_mid = window_start + (window_end - window_start)/2
    aggscores_window["window_mid"] = window_mid
    gene_aggscores_window["window_mid"] = window_mid
    
    aggscores_windows.append(aggscores_window)
    gene_aggscores_windows.append(gene_aggscores_window)
    
aggscores_windows = pd.concat(aggscores_windows)
aggscores_windows = aggscores_windows.set_index("window_mid", append = True)

gene_aggscores_windows = pd.concat(gene_aggscores_windows)
gene_aggscores_windows = gene_aggscores_windows.set_index("window_mid", append = True)

# %% [markdown]
# ### Global view

# %%
aggscores_windows.loc["train"]["perc_retained"].plot()
aggscores_windows.loc["validation"]["perc_retained"].plot()

# %%
aggscores_windows.loc["train"]["effect"].plot()
aggscores_windows.loc["validation"]["effect"].plot()

# %%
mse_windows = aggscores_windows["mse"].unstack().T
mse_dummy_windows = aggscores_windows["mse_dummy"].unstack().T

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
gene_mse_windows = gene_aggscores_windows["mse"].unstack()
gene_perc_retained_windows = gene_aggscores_windows["perc_retained"].unstack()
gene_mse_dummy_windows = gene_aggscores_windows["mse_dummy"].unstack()
gene_effect_windows = gene_aggscores_windows["effect"].unstack()

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

# %% [markdown]
# #### Plot a single gene

# %%
gene_id = transcriptome.gene_id("PLXDC2")
gene_id = transcriptome.gene_id("FOSB")
gene_id = transcriptome.gene_id("C9orf72")
gene_id = transcriptome.gene_id("WARS")

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
peaks = []

import pybedtools
promoter_bed = pybedtools.BedTool.from_dataframe(pd.DataFrame(promoter).T[["chr", "start", "end"]])

peaks_bed = pybedtools.BedTool(folder_data_preproc / "atac_peaks.bed")
peaks_cellranger = promoter_bed.intersect(peaks_bed).to_dataframe()
peaks_cellranger["method"] = "cellranger"
peaks_cellranger = center_peaks(peaks_cellranger, promoter)
peaks.append(peaks_cellranger)

peaks_bed = pybedtools.BedTool(pfa.get_output() / "peaks" / dataset_name / "macs2" / "peaks.bed")
peaks_macs2 = promoter_bed.intersect(peaks_bed).to_dataframe()
peaks_macs2["method"] = "macs2"
peaks_macs2 = center_peaks(peaks_macs2, promoter)
peaks.append(peaks_macs2)

# peaks_bed = pybedtools.BedTool(pfa.get_output() / "peaks" / dataset_name / "genrich" / "peaks.bed")
# peaks_genrich = promoter_bed.intersect(peaks_bed).to_dataframe()
# peaks_genrich["method"] = "genrich"
# peaks_genrich = center_peaks(peaks_genrich, promoter)
# peaks.append(peaks_genrich)

peaks = pd.concat(peaks)

peak_methods = pd.DataFrame({"method":peaks["method"].unique()}).set_index("method")
peak_methods["ix"] = np.arange(peak_methods.shape[0])

# %%
gene_effect_windows.loc["validation"].max(1).sort_values()

# %% [markdown]
# Extract bigwig info of gene

# %%
import pyBigWig
bw = pyBigWig.open(str(folder_data_preproc / "atac_cut_sites.bigwig"))

# %%
fig, (ax_mse, ax_effect, ax_perc, ax_peak, ax_bw) = plt.subplots(5, 1, height_ratios = [1, 0.5, 0.5, 0.2, 0.2], sharex=True)
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

# effect
plotdata = gene_effect_windows.loc["train"].loc[gene_id]
patch_train = ax_effect.plot(plotdata.index, plotdata, color = "blue", label = "train")
plotdata = gene_effect_windows.loc["validation"].loc[gene_id]
patch_validation = ax_effect.plot(plotdata.index, plotdata, color = "red", label = "train")

ax_effect.axhline(0, color = "#333333")
ax_effect.set_ylim(ax_effect.get_ylim()[::-1])
ax_effect.set_ylabel("Effect", rotation = 0, ha = "right")

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
# ### Are opposing effects put into the same peak?

# %% [markdown]
# ## Performance when masking a pairs of windows

# %% [markdown]
# Hypothesis: are fragments from pairs of regions co-predictive, i.e. fragments from both regions together provide a non-linear prediction?

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
window_size = 500
cuts = np.arange(-padding_negative, padding_positive, step = window_size)

# %%
aggscores_windowpairs = []
gene_aggscores_windowpairs = []
for window_idx, (window_start1, window_end1) in tqdm.tqdm(enumerate(zip(cuts[:-1], cuts[1:]))):
    for window_idx, (window_start2, window_end2) in enumerate(zip(cuts[:-1], cuts[1:])):
        # take fragments within the window
        fragments_oi1 = select_window(fragments.coordinates, window_start1, window_end1)
        fragments_oi2 = select_window(fragments.coordinates, window_start2, window_end2)

        fragments_oi = fragments_oi1 & fragments_oi2

        aggscores_window, gene_aggscores_window, _ = score_fragments(splits, fragments_oi)

        window_mid1 = window_start1 + (window_end1 - window_start1)/2
        aggscores_window["window_mid1"] = window_mid1
        gene_aggscores_window["window_mid1"] = window_mid1
        window_mid2 = window_start2 + (window_end2 - window_start2)/2
        aggscores_window["window_mid2"] = window_mid2
        gene_aggscores_window["window_mid2"] = window_mid2

        aggscores_windowpairs.append(aggscores_window)
        gene_aggscores_windowpairs.append(gene_aggscores_window)
    
aggscores_windowpairs = pd.concat(aggscores_windowpairs)
aggscores_windowpairs = aggscores_windowpairs.set_index(["window_mid1", "window_mid2"], append = True)

gene_aggscores_windowpairs = pd.concat(gene_aggscores_windowpairs)
gene_aggscores_windowpairs = gene_aggscores_windowpairs.set_index(["window_mid1", "window_mid2"], append = True)

# %% [markdown]
# ### Global view

# %%
aggscores_windowpairs.loc["train", "perc_retained"].plot()

# %%
mse_windowpairs = aggscores_windowpairs["mse"].unstack()
mse_dummy_windowpairs = aggscores_windowpairs["mse_dummy"].unstack()

# %%
sns.heatmap(mse_windowpairs.loc["validation"])

# %% [markdown]
# #### Calculate interaction

# %% [markdown]
# Try to calculate whether an interactions occurs, i.e. if removing both windows make things worse or better than removing the windows individually

# %%
aggscores_windowpairs["mse_loss"] = aggscores["mse"] - aggscores_windowpairs["mse"]
aggscores_windowpairs["perc_loss"] = 1- aggscores_windowpairs["perc_retained"]

# %%
# determine what the reference (single) mse values are
# in this case, we can simply use the diagonal
reference_idx = aggscores_windowpairs.index.get_level_values("window_mid2") == aggscores_windowpairs.index.get_level_values("window_mid1")
reference = aggscores_windowpairs.loc[reference_idx]
reference1 = reference.droplevel("window_mid2")
reference2 = reference.droplevel("window_mid1")

# %%
cols = ["mse_loss", "perc_loss"]

# %%
aggscores_windowpairs_test = aggscores_windowpairs.join(reference1[cols], rsuffix = "1").join(reference2[cols], rsuffix = "2")

# %% [markdown]
# Fragments can be removed by both perturbations at the same time, e.g. if two windows are adjacent a lot of fragments will be shared.
# We corrected for this bias using the `perc_loss_bias` column.

# %%
# calculate the bias
aggscores_windowpairs_test["perc_loss12"] = (aggscores_windowpairs_test["perc_loss1"] + aggscores_windowpairs_test["perc_loss2"])
aggscores_windowpairs_test["perc_loss_bias"] = (
    aggscores_windowpairs_test["perc_loss"] /
    aggscores_windowpairs_test["perc_loss12"]
)

# %%
# calculate the (corrected) expected additive values
for col in cols:
    aggscores_windowpairs_test[col + "12"] = (
        (aggscores_windowpairs_test[col+"1"] + aggscores_windowpairs_test[col+"2"]) * 
        (aggscores_windowpairs_test["perc_loss_bias"])
    )

# calculate the interaction
for col in cols:
    aggscores_windowpairs_test[col + "_interaction"] = (
        aggscores_windowpairs_test[col] -
        aggscores_windowpairs_test[f"{col}12"]
    )

# %% [markdown]
# We can check that the bias correction worked correctly by checking the interaction of perc_loss, which should be 0.

# %%
plotdata = aggscores_windowpairs_test.loc[("validation")]["perc_loss_interaction"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.RdBu_r, center = 0., vmin = -1e-5, vmax = 1e-5)

# %% [markdown]
# #### Plot interaction

# %%
plotdata = aggscores_windowpairs_test.loc[("validation")]["mse_loss_interaction"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.RdBu, center = 0.)

# %% [markdown]
# ### Gene-specific view

# %% [markdown]
# #### Calculate interaction effects

# %% [markdown]
# Test whether:
# $$
# \text{effect}_1 + \text{effect}_2 \neq \text{effect}_{1,2}
# $$
#
# given:
# $$
# \text{effect} = \text{MSE}_\text{unperturbed} - \text{MSE}_\text{perturbed}
# $$

# %%
gene_aggscores_windowpairs["mse_loss"] = gene_aggscores["mse"] - gene_aggscores_windowpairs["mse"]
gene_aggscores_windowpairs["cor_loss"] = -gene_aggscores["cor"] + gene_aggscores_windowpairs["cor"]
gene_aggscores_windowpairs["perc_loss"] = 1- gene_aggscores_windowpairs["perc_retained"]

# %%
# determine what the reference mse values are for a single perturbation
# in this case, we can simply use the "diagonal", i.e. where window_mid1 == window_mid2, because in that case only one fragment is removed
# in other perturbation, you will probably have to use some other technique
# you could for example include a "dummy" effect
reference_idx = gene_aggscores_windowpairs.index.get_level_values("window_mid2") == gene_aggscores_windowpairs.index.get_level_values("window_mid1")
reference = gene_aggscores_windowpairs.loc[reference_idx]
reference1 = reference.droplevel("window_mid2")
reference2 = reference.droplevel("window_mid1")

# %%
cols = ["mse_loss", "perc_loss", "cor_loss"]

# %%
gene_aggscores_windowpairs_test = gene_aggscores_windowpairs.join(reference1[cols], rsuffix = "1").join(reference2[cols], rsuffix = "2")

# %%
# calculate the bias
gene_aggscores_windowpairs_test["perc_loss12"] = (gene_aggscores_windowpairs_test["perc_loss1"] + gene_aggscores_windowpairs_test["perc_loss2"])
eps = 1e-8
gene_aggscores_windowpairs_test["perc_loss_bias"] = (
    gene_aggscores_windowpairs_test["perc_loss"] /
    gene_aggscores_windowpairs_test["perc_loss12"]
).fillna(1)

# %%
# calculate the (corrected) expected additive values
for col in cols:
    gene_aggscores_windowpairs_test[col + "12"] = (
        (gene_aggscores_windowpairs_test[col+"1"] + gene_aggscores_windowpairs_test[col+"2"]) * 
        (gene_aggscores_windowpairs_test["perc_loss_bias"])
    )

# calculate the interaction
for col in cols:
    gene_aggscores_windowpairs_test[col + "_interaction"] = (
        gene_aggscores_windowpairs_test[col] -
        gene_aggscores_windowpairs_test[f"{col}12"]
    )

# %% [markdown]
# #### Plot for particular gene

# %%
col = "perc_loss"
plotdata_all = gene_aggscores_windowpairs_test.loc[("validation", gene_id)]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4), sharey = True, sharex = True)

norm = mpl.colors.Normalize(vmin = 0)

plotdata = plotdata_all[f"{col}"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.Reds, ax = ax1, norm = norm)

plotdata = plotdata_all[f"{col}12"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.Reds, ax = ax2, norm = norm)

norm = mpl.colors.CenteredNorm(0, halfrange = 1e-5)

plotdata = plotdata_all[f"{col}_interaction"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.RdBu, norm = norm, ax = ax3)

# %%
gene_aggscores_windowpairs_test["mse_loss_interaction"].loc["validation"].sort_values(ascending = True).head(10)

# %%
# gene_id = transcriptome.gene_id("LYN")
gene_id = transcriptome.gene_id("PLXDC2")
# gene_id = transcriptome.gene_id("TNFAIP2")
gene_id = transcriptome.gene_id("HLA-DRA")
gene_id = transcriptome.gene_id("NKG7")

# %%
col = "mse_loss"

# %%
plotdata_all = gene_aggscores_windowpairs_test.loc[("validation", gene_id)]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4), sharey = True, sharex = True)

norm = mpl.colors.Normalize(vmax = 0)

plotdata = plotdata_all[f"{col}"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.Reds_r, ax = ax1, norm = norm)

plotdata = plotdata_all[f"{col}12"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.Reds_r, ax = ax2, norm = norm)

norm = mpl.colors.CenteredNorm(0)

plotdata = plotdata_all[f"{col}_interaction"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.RdBu, norm = norm, ax = ax3)

# %% [markdown]
# ## Performance when masking peaks

# %% [markdown]
# Hypothesis: are fragments outside of peaks also predictive

# %% [markdown]
# ## Performance when removing fragment lengths

# %% [markdown]
# Hypothesis: **do fragments of certain lengths provide more or less predictive power?**

# %%
splits = [split.to("cuda") for split in splits]

# %%
# cuts = [200, 400, 600]
# cuts = list(np.arange(0, 200, 10)) + list(np.arange(200, 1000, 50))
cuts = list(np.arange(0, 1000, 25))
windows = [[cut0, cut1] for cut0, cut1 in zip(cuts, cuts[1:] + [9999999])]

# %%
aggscores_lengths = []
gene_aggscores_lengths = []
for window_idx, (window_start, window_end) in tqdm.tqdm(enumerate(windows)):
    # take fragments within the window
    fragment_lengths = (fragments.coordinates[:,1] - fragments.coordinates[:,0])
    fragments_oi = ~((fragment_lengths >= window_start) & (fragment_lengths < window_end))

    aggscores_window, gene_aggscores_window, _ = score_fragments(splits, fragments_oi)

    window_mid1 = window_start1 + (window_end1 - window_start1)/2
    aggscores_window["window_start"] =  window_start
    gene_aggscores_window["window_start"] =  window_start

    aggscores_lengths.append(aggscores_window)
    gene_aggscores_lengths.append(gene_aggscores_window)
    
aggscores_lengths = pd.concat(aggscores_lengths)
aggscores_lengths = aggscores_lengths.set_index("window_start", append = True)

gene_aggscores_lengths = pd.concat(gene_aggscores_lengths)
gene_aggscores_lengths = gene_aggscores_lengths.set_index("window_start", append = True)

# %% [markdown]
# ### Global view

# %%
mse_lengths = aggscores_lengths["mse"].unstack().T
mse_dummy_lengths = aggscores_lengths["mse_dummy"].unstack().T
perc_retained_lengths = aggscores_lengths["perc_retained"].unstack().T

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

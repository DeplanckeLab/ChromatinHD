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
# # Explore model

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
transcriptome_pd = pd.DataFrame(transcriptome.X.dense().cpu().numpy(), index = transcriptome.obs.index, columns = transcriptome.var.index)


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
        
        effect_train = (expression_prediction.loc[cells_train] - expression_prediction_full.loc[cells_train]).std()
        effect_validation = (expression_prediction.loc[cells_validation] - expression_prediction_full.loc[cells_validation]).std()
    
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

splits = [split.to(device) for split in splits]
transcriptome_X = transcriptome.X.to(device)

aggscores, gene_aggscores, expression_prediction = score_fragments(splits, fragments_oi, return_expression_prediction = True)

splits = [split.to("cpu") for split in splits]
transcriptome_X = transcriptome.X.to("cpu")
torch.cuda.empty_cache()

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
# ## Performance when removing fragment lengths

# %% [markdown]
# Hypothesis: **do fragments of certain lengths provide more or less predictive power?**

# %%
# cuts = [200, 400, 600]
# cuts = list(np.arange(0, 200, 10)) + list(np.arange(200, 1000, 50))
cuts = list(np.arange(0, 1000, 25))
windows = [[cut0, cut1] for cut0, cut1 in zip(cuts, cuts[1:] + [9999999])]

# %%
splits = [split.to(device) for split in splits]
transcriptome_X = transcriptome.X.to(device)

aggscores_lengths = []
gene_aggscores_lengths = []
for window_start, window_end in tqdm.tqdm(windows):
    # take fragments within the window
    fragment_lengths = (fragments.coordinates[:,1] - fragments.coordinates[:,0])
    fragments_oi = ~((fragment_lengths >= window_start) & (fragment_lengths < window_end))

    aggscores_window, gene_aggscores_window = score_fragments(splits, fragments_oi, expression_prediction_full=expression_prediction)

    aggscores_window["window_start"] =  window_start
    gene_aggscores_window["window_start"] =  window_start

    aggscores_lengths.append(aggscores_window)
    gene_aggscores_lengths.append(gene_aggscores_window)
    
splits = [split.to("cpu") for split in splits]
transcriptome_X = transcriptome.X.to("cpu")
torch.cuda.empty_cache()
    
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
effect_lengths = aggscores_lengths["effect"].unstack().T


# %%
def zscore(x):
    return (x - x.mean())/x.std()
def minmax(x):
    return (x - x.min())/(x.max() - x.min())


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

# %%
fig, ax = plt.subplots()
(mse_lengths.std() * (zscore(mse_lengths) - zscore(1-perc_retained_lengths)))["validation"].plot()
ax.set_ylabel("Relative MSE", rotation = 0, ha = "right", va = "center")

# %% [markdown]
# ## Performance when masking a window

# %% [markdown]
# Hypothesis: **are fragments from certain regions more predictive than others?**

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
windows = np.array(list(zip(cuts[:-1], cuts[1:], cuts[1:] + (cuts[:-1] - cuts[1:])/2)))

# %%
splits = [split.to(device) for split in splits]
transcriptome_X = transcriptome.X.to(device)

aggscores_windows = []
gene_aggscores_windows = []
for window_start, window_end, window_mid in tqdm.tqdm(windows):
    # take fragments within the window
    fragments_oi = select_window(fragments.coordinates, window_start, window_end)
    
    aggscores_window, gene_aggscores_window = score_fragments(splits, fragments_oi, expression_prediction_full=expression_prediction)
    
    aggscores_window["window_mid"] = window_mid
    gene_aggscores_window["window_mid"] = window_mid
    
    aggscores_windows.append(aggscores_window)
    gene_aggscores_windows.append(gene_aggscores_window)
    
splits = [split.to("cpu") for split in splits]
transcriptome_X = transcriptome.X.to("cpu")
torch.cuda.empty_cache()
    
aggscores_windows = pd.concat(aggscores_windows)
aggscores_windows = aggscores_windows.set_index("window_mid", append = True)

gene_aggscores_windows = pd.concat(gene_aggscores_windows)
gene_aggscores_windows = gene_aggscores_windows.set_index("window_mid", append = True)

aggscores_windows["mse_loss"] = aggscores["mse"] - aggscores_windows["mse"]
gene_aggscores_windows["mse_loss"] = gene_aggscores["mse"] - gene_aggscores_windows["mse"]

# %% [markdown]
# ### Global view

# %%
aggscores_windows.loc["train"]["perc_retained"].plot()
aggscores_windows.loc["validation"]["perc_retained"].plot()

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

# %%
import sklearn.linear_model

# %%
lm = sklearn.linear_model.LinearRegression()
lm.fit(1-aggscores_windows.loc["validation"][["perc_retained"]], aggscores_windows.loc["validation"]["mse"])
mse_residual = aggscores_windows.loc["validation"]["mse"] - lm.predict(1-aggscores_windows.loc["validation"][["perc_retained"]])

# %%
mse_residual.plot()

# %%
fig, ax = plt.subplots()
ax.set_ylabel("Relative MSE", rotation = 0, ha = "right", va = "center")
ax.plot(
    aggscores_windows.loc["train"].index,
    zscore(aggscores_windows.loc["train"]["mse"]) - zscore(1-aggscores_windows.loc["train"]["perc_retained"]),
    color = "blue"
)
ax.plot(
    aggscores_windows.loc["validation"].index,
    zscore(aggscores_windows.loc["validation"]["mse"]) - zscore(1-aggscores_windows.loc["validation"]["perc_retained"]),
    color = "red"
)

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
# if you want genes with a high "negative effect" somewhere
gene_effect_windows.loc["validation"].max(1).sort_values(ascending = False).head(8)

# if you want genes with the highest mse diff
gene_aggscores.loc["validation"].sort_values("mse_diff", ascending = False).head(8)

# %%
# gene_id = transcriptome.gene_id("HLA-B")
# gene_id = transcriptome.gene_id("PTPRC")
# gene_id = transcriptome.gene_id("SIPA1L1")
gene_id = transcriptome.gene_id("IL1B")
gene_id = transcriptome.gene_id("FLT3LG")

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
ax_effect.set_ylabel("Effect", rotation = 0, ha = "right", va = "center")

# perc_retained
plotdata = gene_perc_retained_windows.loc["train"].loc[gene_id]
ax_perc.plot(plotdata.index, plotdata, color = "blue", label = "train")
plotdata = gene_perc_retained_windows.loc["validation"].loc[gene_id]
ax_perc.plot(plotdata.index, plotdata, color = "red", label = "train")

ax_perc.axvline(0, color = "#33333366", lw = 1)

ax_perc.set_ylabel("Fragments\nretained", rotation = 0, ha = "right", va = "center")
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
# ### Is promoter opening purely positive for gene expression, negative, or a mix?

# %% [markdown]
# - "up" = gene expression goes up if we remove a window, i.e. if there are fragments in this window the gene expression goes down
# - "down" = gene expression goes down if we remove a window, i.e. if there are fragments in this window the gene expression goes up

# %%
promoter_updown = pd.DataFrame({
    "max_increase":gene_effect_windows.loc["validation"].max(1),
    "max_decrease":gene_effect_windows.loc["validation"].min(1)
})

cutoff = np.abs(np.quantile(promoter_updown["max_decrease"], 0.01))

promoter_updown["down"] = promoter_updown["max_decrease"] < -cutoff
promoter_updown["up"] = promoter_updown["max_increase"] > cutoff

# %%
promoter_updown = pd.DataFrame({
    "max_increase":gene_effect_windows.loc["validation"].max(1),
    "max_decrease":gene_effect_windows.loc["validation"].min(1)
})
promoter_updown["label"] = transcriptome.symbol(promoter_updown.index)

cutoff = np.abs(np.quantile(promoter_updown["max_decrease"], 0.1))

promoter_updown["down"] = promoter_updown["max_decrease"] < -cutoff
promoter_updown["up"] = promoter_updown["max_increase"] > cutoff
promoter_updown["type"] = (pd.Series(["nothing", "down"])[promoter_updown["down"].values.astype(int)].reset_index(drop = True) + "_" + pd.Series(["nothing", "up"])[promoter_updown["up"].values.astype(int)].reset_index(drop = True)).values

fig, ax = plt.subplots()
type_info = pd.DataFrame([
    ["down_nothing", "green"],
    ["nothing_up", "blue"],
    ["down_up", "red"],
    ["nothing_nothing", "grey"],
])

sns.scatterplot(x = promoter_updown["max_increase"], y = promoter_updown["max_decrease"], hue = promoter_updown["type"])

# %%
promoter_updown.groupby("type").size()/promoter_updown.shape[0]

# %%
# if you're interested in the genes with the strongest increase effect
promoter_updown.query("type == 'nothing_up'").sort_values("max_increase", ascending = False)

# %% [markdown]
# ### Is information content of a window associated with the number of fragments?

# %%
sns.scatterplot(x = gene_aggscores_windows["mse_loss"], y = gene_aggscores_windows["perc_retained"], s = 1)

# %%
gene_mse_correlations = gene_aggscores_windows.groupby(["phase", "gene"]).apply(lambda x: x["perc_retained"].corr(x["mse"]))

# %%
gene_mse_correlations = gene_aggscores_windows.query("perc_retained < 0.98").groupby(["phase", "gene"]).apply(lambda x: x["perc_retained"].corr(x["mse"]))
gene_mse_correlations = gene_mse_correlations.to_frame("cor")
gene_mse_correlations = gene_mse_correlations[~pd.isnull(gene_mse_correlations["cor"])]
gene_mse_correlations["label"] = transcriptome.symbol(gene_mse_correlations.index.get_level_values("gene")).values

# %%
genes_oi = gene_aggscores.loc["validation"].query("mse_diff > 1e-3").index

# %%
fig, ax = plt.subplots()
ax.hist(gene_mse_correlations.loc["validation"].loc[:, "cor"], range = (-1, 1))
ax.hist(gene_mse_correlations.loc["validation"].loc[genes_oi, "cor"], range = (-1, 1))

# %%
gene_mse_correlations.loc["validation"].loc[genes_oi].sort_values("cor", ascending = False).head(10)

# %% [markdown]
# IL1B in pbmc10k is interesting here

# %%
gene_mse_correlations.loc["validation"].loc[genes_oi].sort_values("cor", ascending = True).iloc[400:]

# %% [markdown]
# ## Comparing peaks and windows

# %% [markdown]
# ### Linking peaks to windows

# %% [markdown]
# Create a `peak_window_matches` dataframe that contains peak - window - gene in long format

# %%
promoters = pd.read_csv(folder_data_preproc / "promoters.csv", index_col = 0)

# %%
# peaks_name = "cellranger"
peaks_name = "macs2"

# %%
peaks_folder = folder_root / "peaks" / dataset_name / peaks_name
peaks = pd.read_table(peaks_folder / "peaks.bed", names = ["chrom", "start", "end"], usecols = [0, 1, 2])

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
def center_peaks(peaks, promoters):
    promoter = promoters.loc[peaks["gene"]]
    
    peaks2 = peaks.copy()
    
    peaks2["start"] = np.where(
        promoter["strand"].values == 1,
        (peaks["start"] - promoter["tss"].values) * promoter["strand"].values,
        (peaks["end"] - promoter["tss"].values) * promoter["strand"].values
    )
    peaks2["end"] = np.where(
        promoter["strand"].values == 1,
        (peaks["end"] - promoter["tss"].values) * promoter["strand"].values,
        (peaks["start"] - promoter["tss"].values) * promoter["strand"].values,
    )
    return peaks2


# %%
localpeaks = center_peaks(peaks, promoters)

# %%
matched_peaks, matched_windows = np.where(((localpeaks["start"].values[:, None] < np.array(windows)[:, 1][None, :]) & (localpeaks["end"].values[:, None] > np.array(windows)[:, 0][None, :])))

# %%
peak_window_matches = pd.DataFrame({"peak":localpeaks.index[matched_peaks], "window_mid":windows[matched_windows, 2]}).set_index("peak").join(peaks[["gene"]]).reset_index()

# %% [markdown]
# ### Is the most predictive window inside a peak?

# %%
gene_best_windows = gene_aggscores_windows.loc["validation"].loc[gene_aggscores_windows.loc["validation"].groupby(["gene"])["mse"].idxmax()]

# %%
genes_oi = gene_aggscores.loc["validation"].query("mse_diff > 1e-3").index

# %%
gene_best_windows = gene_best_windows.join(peak_window_matches.set_index(["gene", "window_mid"])).reset_index(level = "window_mid")

# %%
gene_best_windows["matched"] = ~pd.isnull(gene_best_windows["peak"])

# %%
gene_best_windows = gene_best_windows.sort_values("mse_diff", ascending = False)
gene_best_windows["ix"] = np.arange(1, gene_best_windows.shape[0] + 1)
gene_best_windows["cum_matched"] = (np.cumsum(gene_best_windows["matched"]) / gene_best_windows["ix"])
gene_best_windows["perc"] = gene_best_windows["ix"] / gene_best_windows.shape[0]

# %% [markdown]
# Of the top 5% most predictive genes, how many are inside a peak?

# %%
top_cutoff = 0.05
perc_within_a_peak = gene_best_windows["cum_matched"].iloc[int(gene_best_windows.shape[0] * top_cutoff)]
print(perc_within_a_peak)
print(f"Perhaps the most predictive window in the promoter is not inside of a peak?\nIndeed, for {1-perc_within_a_peak:.2%} of the {top_cutoff:.0%} best predicted genes, the most predictive window does not lie within a peak.")

# %%
fig, ax = plt.subplots()
ax.plot(
    gene_best_windows["perc"],
    gene_best_windows["cum_matched"]
)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.set_xlabel("Top genes (acording to mse_loss)")
ax.set_ylabel("% of genes for which\nthe top window is\ncontained in a peak", rotation = 0, ha = "right", va = "center")

# %%
gene_best_windows["label"] = transcriptome.symbol(gene_best_windows.index)

# %%
gene_best_windows.query("~matched")

# %% [markdown]
# ### Are all predictive windows within a peak?

# %%
gene_aggscores_windows_matched = gene_aggscores_windows.loc["validation"].join(peak_window_matches.set_index(["gene", "window_mid"])).groupby(["gene", "window_mid"]).first().reset_index(level = "window_mid")

# %%
gene_aggscores_windows_matched["matched"] = ~pd.isnull(gene_aggscores_windows_matched["peak"])
gene_aggscores_windows_matched = gene_aggscores_windows_matched.sort_values("mse_loss")

# %%
gene_aggscores_windows_matched["ix"] = np.arange(1, gene_aggscores_windows_matched.shape[0] + 1)
gene_aggscores_windows_matched["cum_matched"] = (np.cumsum(gene_aggscores_windows_matched["matched"]) / gene_aggscores_windows_matched["ix"])
gene_aggscores_windows_matched["perc"] = gene_aggscores_windows_matched["ix"] / gene_aggscores_windows_matched.shape[0]

# %% [markdown]
# Of the top 5% most predictive sites, how many are inside a peak?

# %%
top_cutoff = 0.05
perc_within_a_peak = gene_aggscores_windows_matched["cum_matched"].iloc[int(gene_aggscores_windows_matched.shape[0] * top_cutoff)]
print(perc_within_a_peak)
print(f"Perhaps there are many windows that are predictive, but are not contained in any peak?\nIndeed, {1-perc_within_a_peak:.2%} of the top {top_cutoff:.0%} predictive windows does not lie within a peak.")

# %%
fig, ax = plt.subplots()
ax.plot(
    gene_aggscores_windows_matched["perc"],
    gene_aggscores_windows_matched["cum_matched"]
)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.set_xlabel("Top windows (acording to mse_loss of the best performing window)")
ax.set_ylabel("% of most predictive windows\ncontained in a peak", rotation = 0, ha = "right", va = "center")

# %% [markdown]
# ### Are opposing effects put into the same peak?

# %%
gene_peak_scores = pd.DataFrame({
    "effect_min":gene_aggscores_windows_matched.query("matched").groupby(["gene", "peak"])["effect"].min(),
    "effect_max":gene_aggscores_windows_matched.query("matched").groupby(["gene", "peak"])["effect"].max(),
    "mse_loss_max":gene_aggscores_windows_matched.query("matched").groupby(["gene", "peak"])["mse_loss"].max()
})

gene_peak_scores["label"] = transcriptome.symbol(gene_peak_scores.index.get_level_values("gene")).values

# %%
gene_peak_scores["effect_highest"] = np.maximum(np.abs(gene_peak_scores["effect_min"]), np.abs(gene_peak_scores["effect_max"]))
gene_peak_scores["effect_highest_cutoff"] = gene_peak_scores["effect_highest"]/4 # we put the cutoff at 1/4 of the highest effect

# %%
gene_peak_scores["up"] = (gene_peak_scores["effect_max"] > gene_peak_scores["effect_highest_cutoff"])
gene_peak_scores["down"] = (gene_peak_scores["effect_min"] < -gene_peak_scores["effect_highest_cutoff"])
gene_peak_scores["updown"] = gene_peak_scores["up"] & gene_peak_scores["down"]

# %%
gene_peak_scores = gene_peak_scores.sort_values("mse_loss_max", ascending = False)

# %%
gene_peak_scores["ix"] = np.arange(1, gene_peak_scores.shape[0] + 1)
gene_peak_scores["cum_updown"] = (np.cumsum(gene_peak_scores["updown"]) / gene_peak_scores["ix"])
gene_peak_scores["perc"] = gene_peak_scores["ix"] / gene_peak_scores.shape[0]

# %% [markdown]
# Of the top 5% most predictive peaks, how many have a single effect?

# %%
top_cutoff = 0.05
perc_updown = gene_peak_scores["cum_updown"].iloc[int(gene_peak_scores.shape[0] * top_cutoff)]
print(perc_updown)
print(f"Perhaps within a peak there may be both windows that are positively and negatively correlated with gene expression?\nIndeed, {perc_updown:.2%} of the top {top_cutoff:.0%} predictive peaks contains both positive and negative effects.")

# %%
fig, ax = plt.subplots()
ax.plot(
    gene_peak_scores["perc"],
    1-gene_peak_scores["cum_updown"]
)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax = 1))
ax.set_xlabel("Top peaks (acording to max mse_loss within peak)")
ax.set_ylabel("% of peaks with\nonly one effect", rotation = 0, ha = "right", va = "center")

# %%
gene_peak_scores.query("updown")

# %% [markdown]
# ### How much information do the non-peak regions contain?

# %%
gene_aggscores_windows_matched = gene_aggscores_windows.loc["validation"].join(peak_window_matches.set_index(["gene", "window_mid"]))

# %%
gene_aggscores_windows_matched["in_peak"] = (~pd.isnull(gene_aggscores_windows_matched["peak"])).astype("category")

# %%
gene_aggscores_windows_matched.groupby(["gene", "in_peak"])["mse_loss"].sum().unstack().T.mean(1)

# %%
plotdata = gene_aggscores_windows_matched.groupby(["gene", "in_peak"]).sum().reset_index()

# %%
sns.boxplot(x = "in_peak", y = "mse_loss", data = plotdata)

# %% [markdown]
# ## Performance when masking cuts within a window

# %% [markdown]
# Hypothesis: **do cuts give differrent informationt han fragments?**

# %%
padding_positive = 2000
padding_negative = 4000


# %%
def select_cutwindow(coordinates, window_start, window_end):
    return ~(((coordinates[:, 0] < window_end) & (coordinates[:, 0] > window_start)) | ((coordinates[:, 1] < window_end) & (coordinates[:, 1] > window_start)))
assert (select_window(np.array([[-100, 200], [-300, -100]]), -50, 50) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -310, -309) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), 201, 202) == np.array([True, True])).all()
assert (select_window(np.array([[-100, 200], [-300, -100]]), -200, 20) == np.array([False, False])).all()

# %%
window_size = 100
cuts = np.arange(-padding_negative, padding_positive, step = window_size)
windows = np.array(list(zip(cuts[:-1], cuts[1:], cuts[1:] + (cuts[:-1] - cuts[1:])/2)))

# %%
splits = [split.to(device) for split in splits]
transcriptome_X = transcriptome.X.to(device)

aggscores_cutwindows = []
gene_aggscores_cutwindows = []
for window_start, window_end, window_mid in tqdm.tqdm(windows):
    # take fragments within the window
    fragments_oi = select_cutwindow(fragments.coordinates, window_start, window_end)
    
    aggscores_cutwindow, gene_aggscores_cutwindow = score_fragments(splits, fragments_oi, expression_prediction_full=expression_prediction)
    
    aggscores_cutwindow["window_mid"] = window_mid
    gene_aggscores_cutwindow["window_mid"] = window_mid
    
    aggscores_cutwindows.append(aggscores_cutwindow)
    gene_aggscores_cutwindows.append(gene_aggscores_cutwindow)
    
splits = [split.to("cpu") for split in splits]
transcriptome_X = transcriptome.X.to("cpu")
torch.cuda.empty_cache()
    
aggscores_cutwindows = pd.concat(aggscores_cutwindows)
aggscores_cutwindows = aggscores_cutwindows.set_index("window_mid", append = True)

gene_aggscores_cutwindows = pd.concat(gene_aggscores_cutwindows)
gene_aggscores_cutwindows = gene_aggscores_cutwindows.set_index("window_mid", append = True)

# %% [markdown]
# ### Global view

# %%
aggscores_cutwindows.loc["train"]["perc_retained"].plot()
aggscores_cutwindows.loc["validation"]["perc_retained"].plot()

# %%
mse_cutwindows = aggscores_cutwindows["mse"].unstack().T
mse_dummy_cutwindows = aggscores_cutwindows["mse_dummy"].unstack().T

# %%
aggscores_windows["mse_loss"] = aggscores["mse"] - aggscores_windows["mse"]
aggscores_cutwindows["mse_loss"] = aggscores["mse"] - aggscores_cutwindows["mse"]

# %%
plotdata_frag = aggscores_windows.loc["validation"]
plotdata_cut = aggscores_cutwindows.loc["validation"]

# %%
plotdata_cut["mse"].plot()
plotdata_frag["mse"].plot()

# %%
plotdata_cut["perc_retained"].plot()
plotdata_frag["perc_retained"].plot()

# %%
lm = sklearn.linear_model.LinearRegression().fit(plotdata_frag[["perc_retained"]],plotdata_frag["mse_loss"])
frag_residual = plotdata_frag["mse_loss"] - lm.predict(plotdata_frag[["perc_retained"]])

# lm = sklearn.linear_model.LinearRegression().fit(plotdata_cut[["perc_retained"]],plotdata_cut["mse_loss"])
cut_residual = plotdata_cut["mse_loss"] - lm.predict(plotdata_cut[["perc_retained"]])

(frag_residual).plot(label = "cut")
(cut_residual).plot(label = "fragment")
plt.legend()

# %% [markdown]
# ## Performance when masking cuts around promoter

# %%
padding_positive = 200
padding_negative = 200

# %%
window_size = 10
cuts = np.arange(-padding_negative, padding_positive, step = window_size)
windows = np.array(list(zip(cuts[:-1], cuts[1:], cuts[1:] + (cuts[:-1] - cuts[1:])/2)))

# %%
splits = [split.to(device) for split in splits]
transcriptome_X = transcriptome.X.to(device)

aggscores_cutwindows = []
gene_aggscores_cutwindows = []
for window_start, window_end, window_mid in tqdm.tqdm(windows):
    # take fragments within the window
    fragments_oi = select_window(fragments.coordinates, window_start, window_end)
    
    aggscores_cutwindow, gene_aggscores_cutwindow = score_fragments(splits, fragments_oi, expression_prediction_full=expression_prediction)
    
    aggscores_cutwindow["window_mid"] = window_mid
    gene_aggscores_cutwindow["window_mid"] = window_mid
    
    aggscores_cutwindows.append(aggscores_cutwindow)
    gene_aggscores_cutwindows.append(gene_aggscores_cutwindow)
    
splits = [split.to("cpu") for split in splits]
transcriptome_X = transcriptome.X.to("cpu")
torch.cuda.empty_cache()
    
aggscores_cutwindows = pd.concat(aggscores_cutwindows)
aggscores_cutwindows = aggscores_cutwindows.set_index("window_mid", append = True)

gene_aggscores_cutwindows = pd.concat(gene_aggscores_cutwindows)
gene_aggscores_cutwindows = gene_aggscores_cutwindows.set_index("window_mid", append = True)

# %% [markdown]
# ### Global view

# %%
aggscores_cutwindows.loc["train"]["perc_retained"].plot()
aggscores_cutwindows.loc["validation"]["perc_retained"].plot()

# %%
mse_cutwindows = aggscores_cutwindows["mse"].unstack().T
mse_dummy_cutwindows = aggscores_cutwindows["mse_dummy"].unstack().T

# %%
aggscores_cutwindows["mse_loss"] = aggscores["mse"] - aggscores_cutwindows["mse"]

# %%
aggscores_cutwindows.loc["validation"]["mse_loss"].plot()

# %%
plotdata_cut["perc_retained"].plot()

# %%
aggscores_cutwindows.loc["validation"]["effect"].plot()

# %%
lm = sklearn.linear_model.LinearRegression().fit(aggscores_cutwindows[["perc_retained"]],aggscores_cutwindows["mse_loss"])
cut_residual = aggscores_cutwindows["mse_loss"] - lm.predict(aggscores_cutwindows[["perc_retained"]])

(cut_residual.loc["validation"]).plot(label = "validation")
(cut_residual.loc["train"]).plot(label = "train")
plt.legend()

# %% [markdown]
# ## Performance when masking pairs of windows

# %% [markdown]
# Hypothesis: **are fragments from pairs of regions co-predictive, i.e. provide a non-linear prediction?**

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
# determine what the reference (single-perturbation) mse values are
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

# %%
col = "perc_loss"
plotdata_all = gene_aggscores_windowpairs_test.loc[("validation", transcriptome.gene_id("MERTK"))]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4), sharey = True, sharex = True)

norm = mpl.colors.Normalize(vmin = 0)

plotdata = plotdata_all[f"{col}"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.Reds, ax = ax1, norm = norm)
ax1.set_title(f"actual {col}")

plotdata = plotdata_all[f"{col}12"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.Reds, ax = ax2, norm = norm)
ax2.set_title(f"expected {col}")

norm = mpl.colors.CenteredNorm(0, halfrange = 1e-5)

plotdata = plotdata_all[f"{col}_interaction"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.RdBu, norm = norm, ax = ax3)
ax3.set_title(f"interation {col}")

# %% [markdown]
# #### Plot for particular gene

# %%
# if you're interested in genes with a strong interaction
gene_aggscores_windowpairs_test.loc["validation"].sort_values("mse_loss_interaction", ascending = True).head(10)

# %%
# gene_id = transcriptome.gene_id("LYN")
gene_id = transcriptome.gene_id("PLXDC2")
# gene_id = transcriptome.gene_id("TNFAIP2")
gene_id = transcriptome.gene_id("HLA-DRA")
# gene_id = transcriptome.gene_id("NKG7")

# %%
col = "mse_loss"

# %%
plotdata_all = gene_aggscores_windowpairs_test.loc[("validation", gene_id)]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12, 4), sharey = True, sharex = True)

norm = mpl.colors.Normalize(vmax = 0)

plotdata = plotdata_all[f"{col}"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.Reds_r, ax = ax1, norm = norm)
ax1.set_title(f"actual {col}")

plotdata = plotdata_all[f"{col}12"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.Reds_r, ax = ax2, norm = norm)
ax2.set_title(f"expected {col}")

norm = mpl.colors.CenteredNorm(0)

plotdata = plotdata_all[f"{col}_interaction"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.RdBu, norm = norm, ax = ax3)
ax3.set_title(f"interation {col}")

# %% [markdown]
# ## Performance when masking peaks

# %% [markdown]
# Hypothesis: **are fragments outside of peaks also predictive?**

# %% [markdown]
# ## Visualize a gene fragments

# %%
padding_positive = 2000
padding_negative = 4000
lim = (-padding_negative, padding_positive)

# %%
gene_id = transcriptome.gene_id("FLT3LG")

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
obs = fragments.obs.copy()
obs["gex"] = sc.get.obs_df(transcriptome.adata, gene_id)
obs = obs.iloc[cell_order]
obs["y"] = np.arange(obs.shape[0])
obs = obs.set_index("ix")

# %%
fig, (ax_fragments, ax_gex) = plt.subplots(1, 2, figsize = (4, n_cells/300), sharey = True)
ax_fragments.set_xlim(lim)
ax_fragments.set_ylim(0, n_cells)

for (start, end, cell_ix) in zip(coordinates[:, 0], coordinates[:, 1], mapping[:, 0]):
    rect = mpl.patches.Rectangle((start, obs.loc[cell_ix, "y"]), end - start, 10, fc = "black", ec = None)
    ax_fragments.add_patch(rect)
ax_gex.plot(obs["gex"], obs["y"])

# %%

# %%

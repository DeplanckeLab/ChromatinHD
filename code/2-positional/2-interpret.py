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
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])

# %%
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")
fragments = peakfreeatac.fragments.Fragments(folder_data_preproc / "fragments" / promoter_name)


# %%
class Prediction(pfa.flow.Flow):
    pass

model_name = "v12"
prediction = Prediction(pfa.get_output() / "prediction_promoter" / dataset_name / promoter_name / model_name)

# %%
folds = pickle.load(open(fragments.path / "folds.pkl", "rb"))
models = pickle.load(open(prediction.path / "models.pkl", "rb"))[:1]

# %%
n_frequencies = 10
rates = torch.linspace(1, 10, n_frequencies)
dist = torch.distributions.Exponential(rates)

x = torch.linspace(-10000, 10000, 100)
sns.heatmap(torch.sign(x[:, None]) * dist.log_prob(torch.abs(x[:, None])/10000))


# %% [markdown]
# ## Overall performace

# %%
def aggregate_splits(df, columns = ("mse", "mse_dummy"), splitinfo = None):
    """
    Calculates the weighted mean of certain columns according to the size (# cells) of each split
    Requires a grouped dataframe as input
    """
    assert "split" in df.obj.columns
    assert isinstance(df, pd.core.groupby.generic.DataFrameGroupBy)
    for col in columns:
        df.obj[col + "_weighted"] = df.obj[col] * splitinfo.loc[df.obj["split"], "weight"].values
    
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
def score_fold(
    coordinates,
    mapping,
    fold,
    model,
    fragments_oi = None,
    expression_prediction_full = None,
    return_expression_prediction = False,
    use_all_train_splits = False,
    calculate_cor = False
):
    scores = []
    
    if fragments_oi is None:
        fragments_oi = torch.tensor([True] * coordinates.shape[0], device = device)
    
    # extract splits
    if use_all_train_splits:
        splits_oi = [split for split in fold if split.phase == "train"] + [split for split in fold if split.phase == "validation"]
    else:
        splits_oi = [fold[0]] + [split for split in fold if split.phase == "validation"]
    
    splitinfo = pd.DataFrame({"split":split_ix, "n_cells":split.cell_n, "phase":split.phase} for split_ix, split in enumerate(splits_oi))
    splitinfo["weight"] = splitinfo["n_cells"] / splitinfo.groupby(["phase"])["n_cells"].sum()[splitinfo["phase"]].values
    
    expression_prediction = pd.DataFrame(0., index = transcriptome.obs.index, columns = transcriptome.var.index)
    
    model = model.to(device)
    model = model.train(False)

    # run all the splits
    for split_ix, split in enumerate(splits_oi):
        try:
            split = split.to(device)

            fragments_oi_split = fragments_oi[split.fragments_selected]

            # calculate how much is retained overall
            perc_retained = fragments_oi_split.float().mean().detach().item()

            # calculate how much is retained per gene
            # scatter is needed here because the values are not sorted by gene (but by cellxgene)
            perc_retained_gene = torch_scatter.scatter_mean(fragments_oi_split.float().to("cpu"), split.local_gene_ix.to("cpu"), dim_size = split.gene_n)

            # run the model and calculate mse
            with torch.no_grad():
                expression_predicted = model.forward2(
                    split,
                    coordinates,
                    mapping,
                    fragments_oi = fragments_oi_split
                )

        finally:
            split = split.to("cpu")

        transcriptome_subset = transcriptome_X.dense_subset(split.cell_ix)[:, split.gene_ix].to(device)

        mse = ((expression_predicted - transcriptome_subset)**2).mean()

        expression_predicted_dummy = transcriptome_subset.mean(0, keepdim = True).expand(transcriptome_subset.shape)
        mse_dummy = ((expression_predicted_dummy - transcriptome_subset)**2).mean()

        transcriptome.obs.loc[transcriptome.obs.index[split.cell_ix], "phase"] = split.phase

        genescores = pd.DataFrame({
            "mse":((expression_predicted - transcriptome_subset)**2).mean(0).detach().cpu().numpy(),
            "gene":transcriptome.var.index[np.arange(split.gene_ix.start, split.gene_ix.stop)],
            "mse_dummy":((expression_predicted_dummy - transcriptome_subset)**2).mean(0).detach().cpu().numpy(),
            "perc_retained":perc_retained_gene.detach().cpu().numpy(),
            "split":split_ix
        })

        scores.append({
            "mse":float(mse.detach().cpu().numpy()),
            "phase":split.phase,
            "genescores":genescores,
            "mse_dummy":float(mse_dummy.detach().cpu().numpy()),
            "perc_retained":perc_retained,
            "split":split_ix
        })

        expression_prediction.values[split.cell_ix, split.gene_ix] = expression_predicted.cpu().detach().numpy()
        
    model = model.to("cpu")

    # aggregate overall scores
    scores = pd.DataFrame(scores)
    scores["phase"] = scores["phase"].astype("category")
    aggscores = aggregate_splits(
        scores.groupby(["phase"]),
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
    gene_aggscores["mse_diff"] = gene_aggscores["mse"] - gene_aggscores["mse_dummy"]

    # calculate summary statistics on the predicted expression
    # first extract train/test splits
    cells_train = transcriptome.obs.index[list(set([cell_ix for split in splits_oi for cell_ix in split.cell_ixs if split.phase == "train"]))].unique()
    cells_validation = transcriptome.obs.index[list(set([cell_ix for split in fold for cell_ix in split.cell_ixs if split.phase == "validation"]))].unique()
    
    # calculate effect
    if expression_prediction_full is not None:
        effect_train = expression_prediction.loc[cells_train].mean() - expression_prediction_full.loc[cells_train].mean()
        effect_validation = expression_prediction.loc[cells_validation].mean() - expression_prediction_full.loc[cells_validation].mean()
        aggscores["effect"] = pd.Series({"train":effect_train.mean(), "validation":effect_validation.mean()})
        gene_aggscores["effect"] = pd.concat({"train":effect_train, "validation":effect_validation}, names = ["phase", "gene"])

    # calculate correlation
    if calculate_cor:
        cor_train = pfa.utils.paircor(expression_prediction.loc[cells_train], transcriptome_pd.loc[cells_train])
        cor_validation = pfa.utils.paircor(expression_prediction.loc[cells_validation], transcriptome_pd.loc[cells_validation])

        gene_aggscores["cor"] = pd.concat({"train":cor_train, "validation":cor_validation}, names = ["phase", "gene"])
        aggscores["cor"] = pd.Series({"train":cor_train.mean(), "validation":cor_validation.mean()})
    
    if return_expression_prediction:
        return aggscores, gene_aggscores, expression_prediction
    return aggscores, gene_aggscores, None


# %%
coordinates = fragments.coordinates.to("cuda")
mapping = fragments.mapping.to("cuda")
transcriptome_X = transcriptome.X.to(device)

score_fold(coordinates, mapping, folds[0], models[0])


# %%
def score_folds(
    coordinates,
    mapping,
    folds,
    fold_runs,
    fragments_oi = None,
    expression_prediction_full = None,
    return_expression_prediction = False,
    **kwargs
):
    """
    Scores a set of fragments_oi using a set of splits
    """
    
    scores = []
    fold_ids = pd.Series(range(len(fold_runs)), name = "fold")
    expression_prediction = pd.DataFrame(0., index = pd.MultiIndex.from_product([fold_ids, transcriptome.obs.index]), columns = transcriptome.var.index)
    
    splitinfo = pd.DataFrame({"split":i, "n_cells":split.cell_n, "phase":split.phase, "fold":fold_ix} for fold_ix, fold in enumerate(folds) for i, split in enumerate(fold))
    splitinfo["weight"] = splitinfo["n_cells"] / splitinfo.groupby(["fold", "phase"])["n_cells"].sum()[pd.MultiIndex.from_frame(splitinfo[["fold", "phase"]])].values
    
    aggscores = {}
    gene_aggscores = {}
    expression_prediction = {}
    
    for model, (fold_ix, fold) in zip(fold_runs, enumerate(folds)):
        aggscores_, gene_aggscores_, expression_prediction_ = score_fold(
            coordinates,
            mapping,
            fold,
            model,
            fragments_oi = fragments_oi,
            return_expression_prediction = return_expression_prediction,
            expression_prediction_full = expression_prediction_full.loc[fold_ix] if expression_prediction_full is not None else None,
            **kwargs
        )
        aggscores[fold_ix] = aggscores_
        gene_aggscores[fold_ix] = gene_aggscores_
        expression_prediction[fold_ix] = expression_prediction_
        
    aggscores = pd.concat(aggscores, names = ["fold", "phase"])
    gene_aggscores = pd.concat(gene_aggscores, names = ["fold", "phase", "gene"])
    if return_expression_prediction:
        expression_prediction = pd.concat(expression_prediction, names = ["fold", "cell"])
    
    if return_expression_prediction:
        return aggscores, gene_aggscores, expression_prediction
    return aggscores, gene_aggscores, None


# %%
coordinates = fragments.coordinates.to(device)

aggscores, gene_aggscores, expression_prediction = score_folds(
    coordinates,
    mapping,
    folds, 
    models,
    return_expression_prediction = True,
    use_all_train_splits = True,
    calculate_cor = True
)

scores = aggscores.groupby(["phase"]).mean()
gene_scores = gene_aggscores.groupby(["phase", "gene"]).mean()

transcriptome_X = transcriptome.X.to("cpu")
torch.cuda.empty_cache()

# %% [markdown]
# ### Global visual check

# %%
cells_oi = np.random.choice(expression_prediction.loc[0].index, size = 100, replace = False) # only subselect 100 cells
plotdata = pd.DataFrame({"prediction":expression_prediction.loc[0].loc[cells_oi].stack()})
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

# %%
scores_dir = (prediction.path / "scoring" / "overall")
scores_dir.mkdir(parents = True, exist_ok = True)

scores.to_pickle(scores_dir / "scores.pkl")
gene_scores.to_pickle(scores_dir / "gene_scores.pkl")

# %% [markdown]
# ### Global view

# %%
# v11 new
aggscores.style.bar()

# %% [markdown]
# ### Gene-specific view

# %%
gene_scores["label"] = transcriptome.symbol(gene_scores.index.get_level_values("gene")).values

# %%
plt.scatter(gene_scores["mse_dummy"], gene_scores["mse_diff"])

# %%
gene_scores.sort_values("mse_diff", ascending = True).head(20).style.bar(subset = ["mse_diff", "cor", "mse_dummy"])

# %%
gene_scores["mse_diff"].unstack().T.sort_values("validation").plot()

# %%
transcriptome.adata.var["mse_diff"] = gene_scores.loc["validation"]["mse_diff"]
transcriptome.adata.var["log_mse_diff"] = np.clip(transcriptome.adata.var["mse_diff"], transcriptome.adata.var["mse_diff"].quantile(0.05), transcriptome.adata.var["mse_diff"].quantile(0.95))
transcriptome.adata.var["log_n_cells"] = np.log10(np.array((transcriptome.adata.X > 0).sum(0))[0] + 1)
transcriptome.adata.var["log_fragments"] = np.log10(pd.Series(np.bincount(mapping[:, 1].cpu().numpy(), minlength = transcriptome.var.shape[0]), fragments.var.index) + 1)
transcriptome.adata.var["log_means"] = np.log10(transcriptome.adata.var["means"] + 1)
transcriptome.adata.var["log_dispersions_norm"] = np.log10(transcriptome.adata.var["dispersions_norm"])

# %%
# fig, ax = plt.subplots()
sns.pairplot(transcriptome.adata.var, vars = ["log_means", "log_n_cells", "log_mse_diff", "log_fragments", "log_dispersions_norm"], diag_kind = None)
# ax.set_xscale("log")

# %% [markdown]
# ## Performance when removing fragment lengths

# %% [markdown]
# Hypothesis: **do fragments of certain lengths provide more or less predictive power?**

# %%
# cuts = [200, 400, 600]
# cuts = list(np.arange(0, 200, 10)) + list(np.arange(200, 1000, 50))
cuts = list(np.arange(0, 1000, 25))

windows = []
for window_start, window_end in zip(cuts, cuts[1:] + [9999999]):  
    windows.append({
        "window_start":window_start,
        "window_end":window_end
    })
windows = pd.DataFrame(windows).set_index("window_start", drop = False)
windows.index.name = "window"

# %%
aggscores_lengths = []
gene_aggscores_lengths = []
for window_id, (window_start, window_end) in tqdm.tqdm(windows.iterrows(), total = windows.shape[0]):
    # take fragments within the window
    fragment_lengths = (fragments.coordinates[:,1] - fragments.coordinates[:,0])
    fragments_oi = ~((fragment_lengths >= window_start) & (fragment_lengths < window_end))
    
    aggscores_window, gene_aggscores_window, _ = score_folds(coordinates, mapping, folds, models, fragments_oi, expression_prediction_full = expression_prediction)

    aggscores_window["window"] =  window_id
    gene_aggscores_window["window"] =  window_id

    aggscores_lengths.append(aggscores_window)
    gene_aggscores_lengths.append(gene_aggscores_window)
    
torch.cuda.empty_cache()
    
aggscores_lengths = pd.concat(aggscores_lengths)
aggscores_lengths = aggscores_lengths.set_index("window", append = True)

scores_lengths = aggscores_lengths.groupby(["phase", "window"]).mean()

gene_aggscores_lengths = pd.concat(gene_aggscores_lengths)
gene_aggscores_lengths = gene_aggscores_lengths.set_index("window", append = True)

gene_scores_lengths = gene_aggscores_lengths.groupby(["phase",  "gene", "window"]).mean()

# %%
scores_dir = (prediction.path / "scoring" / "lengths")
scores_dir.mkdir(parents = True, exist_ok = True)

scores_lengths.to_pickle(scores_dir / "scores.pkl")
gene_scores_lengths.to_pickle(scores_dir / "gene_scores.pkl")

# %% [markdown]
# ### Global view

# %%
mse_lengths = scores_lengths["mse"].unstack().T
mse_dummy_lengths = scores_lengths["mse_dummy"].unstack().T
perc_retained_lengths = scores_lengths["perc_retained"].unstack().T
effect_lengths = scores_lengths["effect"].unstack().T


# %%
def zscore(x, dim = 0):
    return (x - x.values.mean(dim, keepdims = True))/x.values.std(dim, keepdims = True)
def minmax(x, dim = 0):
    return (x - x.values.min(dim, keepdims = True))/(x.max(dim, keepdims = True) - x.min(dim, keepdims = True))


# %%
fig, ax_perc = plt.subplots()
ax_mse = ax_perc.twinx()
ax_mse.plot(mse_lengths.index, mse_lengths["validation"])
ax_mse.axhline(scores.loc["validation", "mse"], color = "blue", dashes = (2, 2))
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
# Do (mono/di/...)"nucleosome" fragments still have an overall positive or negative effect on gene expression?

# %%
fig, ax = plt.subplots()
ax.plot(effect_lengths["validation"].index, effect_lengths["validation"])

# %% [markdown]
# ### Gene-specific view

# %%
gene_scores_lengths["label"] = pd.Categorical(transcriptome.symbol(gene_scores_lengths.index.get_level_values("gene")).values)
gene_scores_lengths["perc_lost"] = 1 - gene_scores_lengths["perc_retained"]

# %%
gene_normmse_lengths = (gene_scores_lengths.loc["validation"]["mse"].unstack().pipe(zscore, dim = 1) - gene_scores_lengths.loc["validation"]["perc_lost"].unstack().pipe(zscore, dim = 1))

# %%
transcriptome.var["mean_expression"] = transcriptome.X.dense().mean(0).cpu().numpy()

# %%
gene_order = gene_scores.loc["validation"].sort_values("mse_diff", ascending = False).index
# gene_order = transcriptome.var.sort_values("mean_expression").index
sns.heatmap(gene_normmse_lengths.loc[gene_order])


# %% [markdown]
# ## Performance when masking a window

# %% [markdown]
# Hypothesis: **are fragments from certain regions more predictive than others?**

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

windows = []
for window_start, window_end in zip(cuts[:-1], cuts[1:]):  
    windows.append({
        "window_start":window_start,
        "window_end":window_end,
        "window":window_start + (window_end - window_start)/2
    })
windows = pd.DataFrame(windows).set_index("window")

# %%
# def run():
#     score_folds(coordinates, folds, models, fragments_oi, expression_prediction_full = expression_prediction)

# import cProfile

# stats = cProfile.run("run()", "restats")
# import pstats

# p = pstats.Stats("restats")
# p.sort_stats("cumulative").print_stats()


# %%
transcriptome_X = transcriptome.X.to(device)
coordinates = fragments.coordinates.to(device)

aggscores_windows = []
gene_aggscores_windows = []
for window_id, (window_start, window_end) in tqdm.tqdm(windows.iterrows(), total = windows.shape[0]):
    # take fragments within the window
    fragments_oi = select_window(fragments.coordinates, window_start, window_end)
    
    aggscores_window, gene_aggscores_window, _ = score_folds(coordinates, mapping, folds, models, fragments_oi, expression_prediction_full = expression_prediction)
    
    aggscores_window["window"] = window_id
    gene_aggscores_window["window"] = window_id
    
    aggscores_windows.append(aggscores_window)
    gene_aggscores_windows.append(gene_aggscores_window)
    
transcriptome_X = transcriptome.X.to("cpu")
torch.cuda.empty_cache()
    
aggscores_windows = pd.concat(aggscores_windows)
aggscores_windows = aggscores_windows.set_index("window", append = True)

gene_aggscores_windows = pd.concat(gene_aggscores_windows)
gene_aggscores_windows = gene_aggscores_windows.set_index("window", append = True)

aggscores_windows["mse_loss"] = aggscores["mse"] - aggscores_windows["mse"]
gene_aggscores_windows["mse_loss"] = gene_aggscores["mse"] - gene_aggscores_windows["mse"]

scores_windows = aggscores_windows.groupby(["phase", "window"]).mean()
gene_scores_windows = gene_aggscores_windows.groupby(["phase", "gene", "window"]).mean()

# %%
scores_dir = (prediction.path / "scoring" / "windows")
scores_dir.mkdir(parents = True, exist_ok = True)

scores_windows.to_pickle(scores_dir / "scores.pkl")
gene_scores_windows.to_pickle(scores_dir / "gene_scores.pkl")

# %% [markdown]
# ### Global view

# %%
scores_windows.loc["train"]["perc_retained"].plot()
scores_windows.loc["validation"]["perc_retained"].plot()

# %%
mse_windows = scores_windows["mse"].unstack().T
mse_dummy_windows = scores_windows["mse_dummy"].unstack().T

# %%
fig, ax_mse = plt.subplots()
patch_train = ax_mse.plot(mse_windows.index, mse_windows["train"], color = "blue", label = "train")
# ax_mse.plot(mse_dummy_windows.index, mse_dummy_windows["train"], color = "blue", alpha = 0.1)
# ax_mse.axhline(scores.loc["train", "mse"], dashes = (2, 2), color = "blue")

ax_mse2 = ax_mse.twinx()

patch_validation = ax_mse2.plot(mse_windows.index, mse_windows["validation"], color = "red", label = "validation")
# ax_mse2.plot(mse_dummy_windows.index, mse_dummy_windows["validation"], color = "red", alpha = 0.1)
# ax_mse2.axhline(scores.loc["validation", "mse"], color = "red", dashes = (2, 2))

ax_mse.set_ylabel("MSE train", rotation = 0, ha = "right", color = "blue")
ax_mse2.set_ylabel("MSE validation", rotation = 0, ha = "left", color = "red")
ax_mse.axvline(0, color = "#33333366", lw = 1)

plt.legend([patch_train[0], patch_validation[0]], ['train', 'validation'])

# %%
import sklearn.linear_model

# %%
lm = sklearn.linear_model.LinearRegression()
lm.fit(1-scores_windows.loc["validation"][["perc_retained"]], scores_windows.loc["validation"]["mse"])
mse_residual = scores_windows.loc["validation"]["mse"] - lm.predict(1-scores_windows.loc["validation"][["perc_retained"]])

# %%
mse_residual.plot()

# %%
fig, ax = plt.subplots()
ax.set_ylabel("Relative MSE", rotation = 0, ha = "right", va = "center")
# ax.plot(
#     scores_windows.loc["train"].index,
#     zscore(aggscores_windows.loc["train"]["mse"]) - zscore(1-aggscores_windows.loc["train"]["perc_retained"]),
#     color = "blue"
# )
ax.plot(
    scores_windows.loc["validation"].index,
    zscore(scores_windows.loc["validation"]["mse"]) - zscore(1-scores_windows.loc["validation"]["perc_retained"]),
    color = "red"
)

# %% [markdown]
# ### Gene-specific view

# %%
special_genes = pd.DataFrame(index = transcriptome.var.index)

# %%
gene_scores["label"] = transcriptome.symbol(gene_scores.index.get_level_values("gene")).values

# %%
gene_mse_windows = gene_scores_windows["mse"].unstack()
gene_perc_retained_windows = gene_scores_windows["perc_retained"].unstack()
gene_mse_dummy_windows = gene_scores_windows["mse_dummy"].unstack()
gene_effect_windows = gene_scores_windows["effect"].unstack()

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
sns.heatmap(gene_mse_windows_norm.loc["validation"].loc[gene_scores.loc["validation"].sort_values("mse_diff", ascending = False).index])

# %% [markdown]
# #### Plot a single gene

# %%
# if you want genes with a high "negative effect" somewhere
gene_effect_windows.loc["validation"].max(1).sort_values(ascending = False).head(8)

# if you want genes with the highest mse diff
gene_scores.loc["validation"].sort_values("mse_diff", ascending = True).head(8)

# %%
gene_id = transcriptome.gene_id("HLA-DRA")
# gene_id = transcriptome.gene_id("PTPRC")
# gene_id = transcriptome.gene_id("LTB")
# gene_id = transcriptome.gene_id("SNRPA1")
gene_id = transcriptome.gene_id("LYZ")

# gene_id = transcriptome.gene_id("Fabp7")
# gene_id = transcriptome.gene_id("Ccnd2")
# gene_id = transcriptome.gene_id("Ptprd")

# %% [markdown]
# Extract promoter info of gene

# %%
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

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

peak_methods = []

peaks_bed = pybedtools.BedTool(folder_data_preproc / "atac_peaks.bed")
peaks_cellranger = promoter_bed.intersect(peaks_bed).to_dataframe()
peaks_cellranger["method"] = "cellranger"
peaks_cellranger = center_peaks(peaks_cellranger, promoter)
peaks.append(peaks_cellranger)
peak_methods.append({"method":"cellranger"})

peaks_bed = pybedtools.BedTool(pfa.get_output() / "peaks" / dataset_name / "macs2" / "peaks.bed")
peaks_macs2 = promoter_bed.intersect(peaks_bed).to_dataframe()
peaks_macs2["method"] = "macs2"
peaks_macs2 = center_peaks(peaks_macs2, promoter)
peaks.append(peaks_macs2)
peak_methods.append({"method":"macs2"})

# peaks_bed = pybedtools.BedTool(pfa.get_output() / "peaks" / dataset_name / "genrich" / "peaks.bed")
# peaks_genrich = promoter_bed.intersect(peaks_bed).to_dataframe()
# peaks_genrich["method"] = "genrich"
# peaks_genrich = center_peaks(peaks_genrich, promoter)
# peaks.append(peaks_genrich)
# peak_methods.append({"method":"genrich"})

peaks = pd.concat(peaks)

peak_methods = pd.DataFrame(peak_methods).set_index("method")
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
ax_mse.set_ylabel("MSE train", rotation = 0, ha = "right", color = "grey")
ax_mse2.set_ylabel("MSE validation", rotation = 0, ha = "left", color = "red")
ax_mse.axvline(0, color = "#33333366", lw = 1)
ax_mse.set_xlim(gene_mse_dummy_windows.columns[0], gene_mse_dummy_windows.columns[-1])

# dummy mse
show_dummy = False
if show_dummy:
    plotdata = gene_mse_dummy_windows.loc["train"].loc[gene_id]
    ax_mse.plot(plotdata.index, plotdata, color = "grey", alpha = 0.1)
    plotdata = gene_mse_dummy_windows.loc["validation"].loc[gene_id]
    ax_mse2.plot(plotdata.index, plotdata, color = "red", alpha = 0.1)

# unperturbed mse
ax_mse.axhline(gene_scores.loc[("train", gene_id), "mse"], dashes = (2, 2), color = "grey")
ax_mse2.axhline(gene_scores.loc[("validation", gene_id), "mse"], dashes = (2, 2), color = "red")

# mse
plotdata = gene_mse_windows.loc["train"].loc[gene_id]
patch_train = ax_mse.plot(plotdata.index, plotdata, color = "grey", label = "train", alpha = 0.3)
plotdata = gene_mse_windows.loc["validation"].loc[gene_id]
patch_validation = ax_mse2.plot(plotdata.index, plotdata, color = "red", label = "train")

# effect
plotdata = gene_effect_windows.loc["train"].loc[gene_id]
patch_train = ax_effect.plot(plotdata.index, plotdata, color = "grey", label = "train")
plotdata = gene_effect_windows.loc["validation"].loc[gene_id]
patch_validation = ax_effect.plot(plotdata.index, plotdata, color = "red", label = "train")

ax_effect.axhline(0, color = "#333333", lw = 0.5, zorder = 0)
ax_effect.set_ylim(ax_effect.get_ylim()[::-1])
ax_effect.set_ylabel("Effect", rotation = 0, ha = "right", va = "center")

# perc_retained
plotdata = gene_perc_retained_windows.loc["train"].loc[gene_id]
ax_perc.plot(plotdata.index, plotdata, color = "grey", label = "train", alpha = 0.3)
plotdata = gene_perc_retained_windows.loc["validation"].loc[gene_id]
ax_perc.plot(plotdata.index, plotdata, color = "red", label = "train")

ax_perc.axvline(0, color = "#33333366", lw = 0.5)

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
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id("PAX5"))

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

# %%
special_genes["opening_decreases_expression"] = promoter_updown["up"]

# %% [markdown]
# ### Is information content of a window associated with the number of fragments?

# %%
sns.scatterplot(x = gene_scores_windows["mse_loss"], y = gene_scores_windows["perc_retained"], s = 1)

# %%
gene_mse_correlations = gene_scores_windows.groupby(["phase", "gene"]).apply(lambda x: x["perc_retained"].corr(x["mse"]))
# gene_mse_correlations = gene_aggscores_windows.query("perc_retained < 0.98").groupby(["phase", "gene"]).apply(lambda x: x["perc_retained"].corr(x["mse"]))
gene_mse_correlations = gene_mse_correlations.to_frame("cor")
gene_mse_correlations = gene_mse_correlations[~pd.isnull(gene_mse_correlations["cor"])]
gene_mse_correlations["label"] = transcriptome.symbol(gene_mse_correlations.index.get_level_values("gene")).values
gene_mse_correlations["mse_diff"] = gene_scores["mse_diff"]

# %%
genes_oi = gene_scores.loc["validation"].query("mse_diff > 1e-3").index
print(f"{len(genes_oi)=}")

# %%
fig, ax = plt.subplots()
ax.hist(gene_mse_correlations.loc["validation"].loc[:, "cor"], range = (-1, 1))
ax.hist(gene_mse_correlations.loc["validation"].loc[genes_oi, "cor"], range = (-1, 1))

# %%
# if you're interested in genes with relatively high correlation with nontheless a good prediction somewhere
gene_mse_correlations.loc["validation"].query("cor > =-.5").sort_values("mse_diff", ascending = False)

# %% [markdown]
# Interesting genes:
#  - *IL1B* and *CD74* in pbmc10k

# %%
special_genes["n_fragments_importance_not_correlated"] = gene_mse_correlations.loc["validation"]["cor"] > -0.5

# %% [markdown]
# ### Is effect of a window associated with the number of fragments?

# %%
sns.scatterplot(x = gene_scores_windows["effect"], y = gene_scores_windows["perc_retained"], s = 1)

# %%
perc_retained_cutoff = gene_scores_windows.loc["validation"]["perc_retained"].quantile(0.05)

# %%
gene_scores_windows.query("perc_retained < @perc_retained_cutoff").loc["validation"].index.get_level_values("gene").unique()

# %%
gene_effect_correlations = gene_scores_windows.query("mse_diff > @mse_loss_cutoff").groupby(["phase", "gene"]).apply(lambda x: x["perc_retained"].corr(x["effect"]))
# gene_mse_correlations = gene_aggscores_windows.query("perc_retained < 0.98").groupby(["phase", "gene"]).apply(lambda x: x["perc_retained"].corr(x["mse"]))
gene_effect_correlations = gene_effect_correlations.to_frame("cor")
gene_effect_correlations = gene_effect_correlations[~pd.isnull(gene_effect_correlations["cor"])]
gene_effect_correlations["label"] = transcriptome.symbol(gene_effect_correlations.index.get_level_values("gene")).values
gene_effect_correlations["mse_diff"] = gene_scores["mse_diff"]

# %%
genes_oi = gene_scores.loc["validation"].query("mse_diff > 1e-3").index.intersection(gene_effect_correlations.index.get_level_values("gene"))
print(f"{len(genes_oi)=}")

# %%
fig, ax = plt.subplots()
ax.hist(gene_effect_correlations.loc["validation"].loc[:, "cor"], range = (-1, 1))
ax.hist(gene_effect_correlations.loc["validation"].loc[genes_oi, "cor"], range = (-1, 1))

# %%
# if you're interested in genes with relatively low correlation with nontheless a good prediction somewhere
gene_effect_correlations.loc["validation"].query("cor <= .5").sort_values("mse_diff", ascending = False)

# if you're interested in genes with relatively high correlation
# gene_effect_correlations.loc["validation"].query("cor > .9").sort_values("mse_diff", ascending = False)

# %% [markdown]
# Interesting genes:
#  - *IL1B* and *CD74* in pbmc10k

# %%
special_genes["n_fragments_importance_not_correlated"] = gene_mse_correlations.loc["validation"]["cor"] > -0.5

# %% [markdown]
# ### Does removing a window improve test performances?

# %%
gene_scores_windows.loc["validation"].sort_values("mse_loss", ascending = False)

# %% [markdown]
# ### Is the TSS positively, negatively or not associated with gene expression?

# %%
gene_scores_windows

# %%
gene_scores_tss = gene_scores_windows.xs(key = windows.iloc[-100].name, level = "window").copy()
gene_scores_tss["label"] = transcriptome.symbol(gene_scores_tss.index.get_level_values("gene")).values

# %%
gene_scores_tss.loc["validation"].query("mse_loss < -1e-4").sort_values("effect")

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
assert (select_cutwindow(np.array([[-100, 200], [-300, -100]]), -50, 50) == np.array([True, True])).all()
assert (select_cutwindow(np.array([[-100, 200], [-300, -100]]), -310, -309) == np.array([True, True])).all()
assert (select_cutwindow(np.array([[-100, 200], [-300, -100]]), 201, 202) == np.array([True, True])).all()
assert (select_cutwindow(np.array([[-100, 200], [-300, -100]]), -200, 20) == np.array([False, False])).all()

# %%
window_size = 100
cuts = np.arange(-padding_negative, padding_positive, step = window_size)

windows = []
for window_start, window_end in zip(cuts[:-1], cuts[1:]):  
    windows.append({
        "window_start":window_start,
        "window_end":window_end,
        "window":window_start + (window_end - window_start)/2
    })
windows = pd.DataFrame(windows).set_index("window")

# %%
transcriptome_X = transcriptome.X.to(device)
coordinates = fragments.coordinates.to(device)

aggscores_cutwindows = []
gene_aggscores_cutwindows = []
for window_id, (window_start, window_end) in tqdm.tqdm(windows.iterrows(), total = windows.shape[0]):
    # take fragments within the window
    fragments_oi = select_window(fragments.coordinates, window_start, window_end)
    
    aggscores_cutwindow, gene_aggscores_cutwindow, _ = score_folds(coordinates, folds, models, fragments_oi, expression_prediction_full = expression_prediction)
    
    aggscores_cutwindow["window"] = window_id
    gene_aggscores_cutwindow["window"] = window_id
    
    aggscores_cutwindows.append(aggscores_cutwindow)
    gene_aggscores_cutwindows.append(gene_aggscores_cutwindow)
    
transcriptome_X = transcriptome.X.to("cpu")
torch.cuda.empty_cache()
    
aggscores_cutwindows = pd.concat(aggscores_cutwindows)
aggscores_cutwindows = aggscores_cutwindows.set_index("window", append = True)

gene_aggscores_cutwindows = pd.concat(gene_aggscores_cutwindows)
gene_aggscores_cutwindows = gene_aggscores_cutwindows.set_index("window", append = True)

scores_cutwindows = aggscores_cutwindows.groupby(["phase", "window"]).mean()
gene_scores_cutwindows = gene_aggscores_cutwindows.groupby(["phase", "gene", "window"]).mean()

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
# ## Performance when masking cuts around TSS

# %%
padding_positive = 200
padding_negative = 200

# %%
window_size = 10
cuts = np.arange(-padding_negative, padding_positive, step = window_size)

windows = []
for window_start, window_end in zip(cuts[:-1], cuts[1:]):  
    windows.append({
        "window_start":window_start,
        "window_end":window_end,
        "window":window_start + (window_end - window_start)/2
    })
windows = pd.DataFrame(windows).set_index("window")

# %%
transcriptome_X = transcriptome.X.to(device)
coordinates = fragments.coordinates.to(device)

aggscores_cutwindows = []
gene_aggscores_cutwindows = []
for window_id, (window_start, window_end) in tqdm.tqdm(windows.iterrows(), total = windows.shape[0]):
    # take fragments within the window
    fragments_oi = select_window(fragments.coordinates, window_start, window_end)
    
    aggscores_cutwindow, gene_aggscores_cutwindow, _ = score_folds(coordinates, folds, models, fragments_oi, expression_prediction_full = expression_prediction)
    
    aggscores_cutwindow["window"] = window_id
    gene_aggscores_cutwindow["window"] = window_id
    
    aggscores_cutwindows.append(aggscores_cutwindow)
    gene_aggscores_cutwindows.append(gene_aggscores_cutwindow)
    
transcriptome_X = transcriptome.X.to("cpu")
torch.cuda.empty_cache()
    
aggscores_cutwindows = pd.concat(aggscores_cutwindows)
aggscores_cutwindows = aggscores_cutwindows.set_index("window", append = True)

gene_aggscores_cutwindows = pd.concat(gene_aggscores_cutwindows)
gene_aggscores_cutwindows = gene_aggscores_cutwindows.set_index("window", append = True)

scores_cutwindows = aggscores_cutwindows.groupby(["phase", "window"]).mean()
gene_scores_cutwindows = gene_aggscores_cutwindows.groupby(["phase", "gene", "window"]).mean()

# %%
aggscores_cutwindows

# %% [markdown]
# ### Global view

# %%
scores_cutwindows

# %%
scores_cutwindows.loc["train"]["perc_retained"].plot()
scores_cutwindows.loc["validation"]["perc_retained"].plot()

# %%
mse_cutwindows = scores_cutwindows["mse"].unstack().T
mse_dummy_cutwindows = scores_cutwindows["mse_dummy"].unstack().T

# %%
scores_cutwindows["mse_loss"] = scores["mse"] - scores_cutwindows["mse"]

# %%
scores_cutwindows.loc["validation"]["mse_loss"].plot()

# %%
scores_cutwindows.loc["validation"]["effect"].plot()

# %%
lm = sklearn.linear_model.LinearRegression().fit(scores_cutwindows[["perc_retained"]],scores_cutwindows["mse_loss"])
cut_residual = scores_cutwindows["mse_loss"] - lm.predict(scores_cutwindows[["perc_retained"]])

(cut_residual.loc["validation"]).plot(label = "validation")
(cut_residual.loc["train"]).plot(label = "train")
plt.legend()

# %% [markdown]
# ## Performance when masking pairs of windows

# %% [markdown]
# Hypothesis: **are fragments from pairs of regions co-predictive, i.e. provide a non-linear prediction?**

# %%
padding_positive = 2000
padding_negative = 4000

# %%
import itertools

# %%
window_size = 500
cuts = np.arange(-padding_negative, padding_positive, step = window_size)

bins = list(itertools.product(zip(cuts[:-1], cuts[1:]), zip(cuts[:-1], cuts[1:])))

# %%
windowpairs = pd.DataFrame(bins, columns = ["window1", "window2"])
windowpairs["window_start1"] = windowpairs["window1"].str[0]
windowpairs["window_start2"] = windowpairs["window2"].str[0]
windowpairs["window_end1"] = windowpairs["window1"].str[1]
windowpairs["window_end2"] = windowpairs["window2"].str[1]

# %%
transcriptome_X = transcriptome.X.to(device)
coordinates = fragments.coordinates.to(device)

aggscores_windowpairs = []
gene_aggscores_windowpairs = []
for _, (window_start1, window_end1, window_start2, window_end2) in tqdm.tqdm(windowpairs[["window_start1", "window_end1", "window_start2", "window_end2"]].iterrows(), total = windowpairs.shape[0]):
    # take fragments within the window
    fragments_oi1 = select_window(fragments.coordinates, window_start1, window_end1)
    fragments_oi2 = select_window(fragments.coordinates, window_start2, window_end2)

    fragments_oi = fragments_oi1 & fragments_oi2

    aggscores_window, gene_aggscores_window, _ = score_folds(coordinates, folds, models, fragments_oi, expression_prediction_full = expression_prediction)

    window_mid1 = window_start1 + (window_end1 - window_start1)/2
    aggscores_window["window1"] = window_mid1
    gene_aggscores_window["window1"] = window_mid1
    window_mid2 = window_start2 + (window_end2 - window_start2)/2
    aggscores_window["window2"] = window_mid2
    gene_aggscores_window["window2"] = window_mid2

    aggscores_windowpairs.append(aggscores_window)
    gene_aggscores_windowpairs.append(gene_aggscores_window)
        
transcriptome_X = transcriptome.X.to("cpu")
coordinates = coordinates.to("cpu")
torch.cuda.empty_cache()
    
aggscores_windowpairs = pd.concat(aggscores_windowpairs)
aggscores_windowpairs = aggscores_windowpairs.set_index(["window1", "window2"], append = True)

gene_aggscores_windowpairs = pd.concat(gene_aggscores_windowpairs)
gene_aggscores_windowpairs = gene_aggscores_windowpairs.set_index(["window1", "window2"], append = True)

aggscores_windowpairs["mse_loss"] = aggscores["mse"] - aggscores_windowpairs["mse"]
gene_aggscores_windowpairs["mse_loss"] = gene_aggscores["mse"] - gene_aggscores_windowpairs["mse"]

scores_windowpairs = aggscores_windowpairs.groupby(["phase", "window1", "window2"]).mean()
gene_scores_windowpairs = gene_aggscores_windowpairs.groupby(["phase", "gene", "window1", "window2"]).mean()

# %%
aggscores_windowpairs["mse_loss"] = aggscores["mse"] - aggscores_windowpairs["mse"]
gene_aggscores_windowpairs["mse_loss"] = gene_aggscores["mse"] - gene_aggscores_windowpairs["mse"]

scores_windowpairs = aggscores_windowpairs.groupby(["phase", "window1", "window2"]).mean()
gene_scores_windowpairs = gene_aggscores_windowpairs.groupby(["phase", "gene", "window1", "window2"]).mean()

# %% [markdown]
# ### Global view

# %%
scores_windowpairs.loc["train", "perc_retained"].plot()

# %%
mse_windowpairs = scores_windowpairs["mse"].unstack()
mse_dummy_windowpairs = scores_windowpairs["mse_dummy"].unstack()

# %%
sns.heatmap(mse_windowpairs.loc["validation"])

# %% [markdown]
# #### Calculate interaction

# %% [markdown]
# Try to calculate whether an interactions occurs, i.e. if removing both windows make things worse or better than removing the windows individually

# %%
scores_windowpairs["mse_loss"] = scores["mse"] - scores_windowpairs["mse"]
scores_windowpairs["perc_loss"] = 1- scores_windowpairs["perc_retained"]

# %%
# determine what the reference (single-perturbation) mse values are
# in this case, we can simply use the diagonal
reference_idx = scores_windowpairs.index.get_level_values("window2") == scores_windowpairs.index.get_level_values("window1")
reference = scores_windowpairs.loc[reference_idx]
reference1 = reference.droplevel("window2")
reference2 = reference.droplevel("window1")

# %%
cols = ["mse_loss", "perc_loss", "effect"]

# %%
scores_windowpairs_test = scores_windowpairs.join(reference1[cols], rsuffix = "1").join(reference2[cols], rsuffix = "2")

# %% [markdown]
# Fragments can be removed by both perturbations at the same time, e.g. if two windows are adjacent a lot of fragments will be shared.
# We corrected for this bias using the `perc_loss_bias` column.

# %%
# calculate the bias
scores_windowpairs_test["perc_loss12"] = scores_windowpairs_test["perc_loss1"] + scores_windowpairs_test["perc_loss2"]
scores_windowpairs_test["perc_loss_bias"] = (
    scores_windowpairs_test["perc_loss"] /
    scores_windowpairs_test["perc_loss12"]
)

# %%
# calculate the (corrected) expected additive values
for col in cols:
    scores_windowpairs_test[col + "12"] = (
        (scores_windowpairs_test[col+"1"] + scores_windowpairs_test[col+"2"]) * 
        (scores_windowpairs_test["perc_loss_bias"])
    )

# calculate the interaction
for col in cols:
    scores_windowpairs_test[col + "_interaction"] = (
        scores_windowpairs_test[col] -
        scores_windowpairs_test[f"{col}12"]
    )

# %% [markdown]
# We can check that the bias correction worked correctly by checking the interaction of perc_loss, which should be 0.

# %%
plotdata = scores_windowpairs_test.loc[("validation")]["perc_loss_interaction"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.RdBu_r, center = 0., vmin = -1e-5, vmax = 1e-5)

# %% [markdown]
# #### Plot interaction

# %%
plotdata = scores_windowpairs_test.loc[("validation")]["mse_loss_interaction"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.RdBu, center = 0.)

# %%
plotdata = scores_windowpairs_test.loc[("validation")]["mse_loss_interaction"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.RdBu, center = 0.)

# %%
plotdata = scores_windowpairs_test.loc[("validation")]["effect_interaction"].unstack()
np.fill_diagonal(plotdata.values, 0)
sns.heatmap(plotdata, cmap = mpl.cm.RdBu, center = 0.)

# %%
plotdata = scores_windowpairs_test.loc[("validation")]["effect_interaction"].unstack()
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
gene_scores_windowpairs["mse_loss"] = gene_scores["mse"] - gene_scores_windowpairs["mse"]
# gene_scores_windowpairs["cor_loss"] = -gene_scores["cor"] + gene_scores_windowpairs["cor"]
gene_scores_windowpairs["perc_loss"] = 1- gene_scores_windowpairs["perc_retained"]

# %%
# determine what the reference mse values are for a single perturbation
# in this case, we can simply use the "diagonal", i.e. where window_mid1 == window_mid2, because in that case only one fragment is removed
# in other perturbation, you will probably have to use some other technique
# you could for example include a "dummy" effect
reference_idx = gene_scores_windowpairs.index.get_level_values("window2") == gene_scores_windowpairs.index.get_level_values("window1")
reference = gene_scores_windowpairs.loc[reference_idx]
reference1 = reference.droplevel("window2")
reference2 = reference.droplevel("window1")

# %%
cols = [
    "mse_loss",
    "perc_loss",
    "effect"
    # "cor_loss"
]

# %%
gene_scores_windowpairs_test = gene_scores_windowpairs.join(reference1[cols], rsuffix = "1").join(reference2[cols], rsuffix = "2")

# %%
# calculate the bias
gene_scores_windowpairs_test["perc_loss12"] = (gene_scores_windowpairs_test["perc_loss1"] + gene_scores_windowpairs_test["perc_loss2"])
eps = 1e-8
gene_scores_windowpairs_test["perc_loss_bias"] = (
    gene_scores_windowpairs_test["perc_loss"] /
    gene_scores_windowpairs_test["perc_loss12"]
).fillna(1)

# %%
# calculate the (corrected) expected additive values
for col in cols:
    gene_scores_windowpairs_test[col + "12"] = (
        (gene_scores_windowpairs_test[col+"1"] + gene_scores_windowpairs_test[col+"2"]) * 
        (gene_scores_windowpairs_test["perc_loss_bias"])
    )

# calculate the interaction
for col in cols:
    gene_scores_windowpairs_test[col + "_interaction"] = (
        gene_scores_windowpairs_test[col] -
        gene_scores_windowpairs_test[f"{col}12"]
    )

# %%
gene_scores_windowpairs_test.groupby("gene")["effect_interaction"].max().sort_values()

# %%
col = "effect"
plotdata_all = gene_scores_windowpairs_test.loc[("validation", gene_scores_windowpairs_test.index.get_level_values("gene")[0])]

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
gene_scores_windowpairs_test["label"] = transcriptome.symbol(gene_scores_windowpairs_test.index.get_level_values("gene")).values

# %%
# if you're interested in genes with a strong interaction
gene_scores_windowpairs_test.loc["validation"].sort_values("mse_loss_interaction", ascending = True).groupby("gene").first().sort_values("mse_loss_interaction", ascending = True).head(10)[["mse_loss_interaction", "label"]]

# if you're interested in genes with a strong negative interaction
# gene_scores_windowpairs_test.loc["validation"].sort_values("mse_loss_interaction", ascending = False).head(10)[["mse_loss_interaction", "label"]]

# %%
gene_id = transcriptome.gene_id("HLA-DRA")
# gene_id = transcriptome.gene_id("PLXDC2")
# gene_id = transcriptome.gene_id("TNFAIP2")
# gene_id = transcriptome.gene_id("RAB11FIP1")

# %%
col = "mse_loss"

# %%
plotdata_all = gene_scores_windowpairs_test.loc[("validation", gene_id)]

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

# %%
# visualize distribution of # fragments
plt.hist(np.bincount(fragments.mapping[:, 0][(fragments.mapping[:, 1] == fragments.var.loc[gene_id]["ix"])].cpu()))

# %% [markdown]
# ## Performance when masking peaks

# %% [markdown]
# Hypothesis: **are fragments outside of peaks also predictive?**

# %%

# %%

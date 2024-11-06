# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
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

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm

# %%
import chromatinhd as chd

chd.set_default_device("cuda:1")
chd.get_default_device()

# %%
import chromatinhd.simulation.simulate

# %%
simulation = chd.simulation.simulate.Simulation(n_genes=10, window=[-10000, 10000])
simulation.create_regions()
simulation.create_obs()

simulation.create_fragments()

simulation.create_motifscan()

# %%
fragments = simulation.fragments
fragments.create_cellxgene_indptr()

# %%
sns.histplot((fragments.coordinates[:, 1] - fragments.coordinates[:, 0]))

# %%
clustering = simulation.clustering

# %%
folds = chd.data.folds.Folds()
folds.sample_cells(fragments, 5)
fold = folds[0]

# %%
minibatcher = chd.models.diff.loader.Minibatcher(np.arange(len(fragments.obs)), np.arange(len(fragments.var)), 100, 100)

# %%
import chromatinhd.models.diff.loader.clustering_fragments

loader = chromatinhd.models.diff.loader.clustering_fragments.ClusteringFragments(
    clustering, fragments, minibatcher.cellxregion_batch_size
)
data = loader.load(next(iter(minibatcher)))

# %%
import chromatinhd.models.diff.model.playground

model = chd.models.diff.model.playground.Model(fragments, clustering)

# %% [markdown]
# ### Tryout

# %%
transform = chromatinhd.models.diff.model.spline.DifferentialQuadraticSplineStack(nbins=(128,), n_genes=1)

# %%
import truncated_normal

# %%
import math


def log_prob_normal(value, loc, scale):
    var = scale**2
    log_scale = math.log(scale)
    return -((value - loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi))


def log_prob_trun_normal(value, loc, scale, a=0, b=1):
    return truncated_normal.TruncatedNormal(loc, scale, a, b).log_prob(value)


def apply_trunc_normal(value, loc, scale, a=0, b=1):
    dist = truncated_normal.TruncatedNormal(loc, scale, a, b)
    return dist.cdf(value), dist.log_prob(value)


# %%
import torch

x = torch.linspace(0.001, 1, 100)
genes_oi = torch.tensor([0])
local_gene_ix = torch.zeros(len(x), dtype=torch.int)
delta = torch.zeros((len(x), np.sum(transform.split_deltas)))
delta[:, :30] = 1
delta[:, 30:40] = -1

log_prob = torch.zeros_like(x)

output, logabsdet = transform.transform_forward(x, genes_oi, local_gene_ix, delta)
log_prob += logabsdet

fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(8, 4))
ax_pdf.plot(x, torch.exp(log_prob).detach().numpy())
ax_cdf.plot(x.numpy(), output.detach().numpy())
assert np.isclose(np.trapz(torch.exp(log_prob).detach().numpy(), x), 1, atol=1e-2)

loc = 0.9
scale = 0.3
loc2 = output[torch.argmin((x - loc).abs())]

output, logabsdet = apply_trunc_normal(output, loc2, scale)
log_prob += logabsdet

ax_pdf.plot(x, torch.exp(log_prob).detach().numpy())
ax_cdf.plot(x.numpy(), output.detach().numpy())
assert np.isclose(np.trapz(torch.exp(log_prob).detach().numpy(), x), 1, atol=1e-2)

loc = 0.1
scale = 0.3
loc2 = output[torch.argmin((x - loc).abs())]

output, logabsdet = apply_trunc_normal(output, loc2, scale)
log_prob += logabsdet

ax_pdf.plot(x, torch.exp(log_prob).detach().numpy())
ax_cdf.plot(x.numpy(), output.detach().numpy())
assert np.isclose(np.trapz(torch.exp(log_prob).detach().numpy(), x), 1, atol=1e-2)

# %% [markdown]
# ## Train

# %%
models = {}

# %%
model = chd.models.diff.model.playground.Model(fragments, clustering, cut_embedder="dummy")
model.train_model(fragments, clustering, fold, n_epochs=100)
models["original_cutdummy"] = model

# %%
model = chd.models.diff.model.playground.Model(
    fragments,
    clustering,
)
model.train_model(fragments, clustering, fold, n_epochs=100)
models["original"] = model

# %%
model = chd.models.diff.model.playground.Model(
    fragments,
    clustering,
    cut_embedder_dropout_rate=0.0,
)
model.train_model(fragments, clustering, fold, n_epochs=100)
models["original_nodrop"] = model

# %%
model = chd.models.diff.model.playground.Model(fragments, clustering, cut_embedder="direct")
model.train_model(fragments, clustering, fold, n_epochs=100)
models["original_cutdirect"] = model

# %% [markdown]
# ## Score

# %%
scores = []
genescores = []
for model_id, model in models.items():
    model = model.eval()
    prediction_test = model.get_prediction(fragments, clustering, cell_ixs=fold["cells_test"])
    prediction_validation = model.get_prediction(fragments, clustering, cell_ixs=fold["cells_validation"])
    prediction_train = model.get_prediction(fragments, clustering, cell_ixs=fold["cells_train"])
    for phase in ["test", "validation", "train"]:
        score = {"model": model_id, "phase": phase}
        score["likelihood_position"] = (prediction_test["likelihood_position"]).sum().item()
        score["likelihood"] = (prediction_test["likelihood"]).sum().item()
        score["n_cells"] = len(fold[f"cells_{phase}"])
        scores.append(score)

        genescore = pd.DataFrame(
            {
                "model": model_id,
                "phase": phase,
                "likelihood": (prediction_test["likelihood"]).sum("cell").to_pandas(),
                "likelihood_position": (prediction_test["likelihood_position"]).sum("cell").to_pandas(),
                "n_cells": len(fold[f"cells_{phase}"]),
            }
        )
        genescores.append(genescore)
scores = pd.DataFrame(scores).set_index(["model", "phase"])
genescores = (
    pd.concat([genescores[i] for i in range(len(genescores))], axis=0)
    .reset_index()
    .set_index(["model", "phase", "gene"])
)

# %%
baseline_id = list(models.keys())[0]
scores["lr_position"] = (scores["likelihood_position"] - scores.loc[baseline_id]["likelihood_position"]).values
scores["lr"] = (scores["likelihood"] - scores.loc[baseline_id]["likelihood"]).values
genescores["lr_position"] = (
    genescores["likelihood_position"] - genescores.loc[baseline_id]["likelihood_position"]
).values
genescores["lr"] = (genescores["likelihood"] - genescores.loc[baseline_id]["likelihood"]).values

# %%
model_info = pd.DataFrame({"model": models.keys()}).set_index("model")
model_info["model_type"] = model_info.index.map(lambda x: "_".join(x.split("_")[:-1]))
model_info = model_info.sort_values(["model_type"])
model_info["ix"] = np.arange(model_info.shape[0])

# %%
fig = polyptich.grid.Figure(polyptich.grid.Wrap(padding_width=0.1))
height = len(model_info) * 0.2

panel, ax = fig.main.add(polyptich.grid.Panel((1, height)))
plotdata = scores.xs("test", level="phase").loc[model_info.index]

ax.barh(plotdata.index, plotdata["lr"])
ax.barh(plotdata.index, plotdata["lr_position"], alpha=0.5)
ax.axvline(0, color="black", linestyle="--", lw=1)
ax.set_title("Test")
ax.set_xlabel("Log-likehood ratio")

panel, ax = fig.main.add(polyptich.grid.Panel((1, height)))
plotdata = scores.xs("validation", level="phase").loc[model_info.index]
ax.set_yticks([])
ax.barh(plotdata.index, plotdata["lr"])
ax.barh(plotdata.index, plotdata["lr_position"], alpha=0.5)
ax.axvline(0, color="black", linestyle="--", lw=1)
ax.set_title("Validation")

panel, ax = fig.main.add(polyptich.grid.Panel((1, height)))
plotdata = scores.xs("train", level="phase").loc[model_info.index]
ax.set_yticks([])
ax.barh(plotdata.index, plotdata["lr"])
ax.barh(plotdata.index, plotdata["lr_position"], alpha=0.5)
ax.axvline(0, color="black", linestyle="--", lw=1)
ax.set_title("Train")
fig.plot()

# %% [markdown]
# ## Interpret

# %%
model = models["original"].eval()

# %%
gene_ix = 2
gene = fragments.var.index[gene_ix]
cluster_ixs = np.arange(clustering.n_clusters)

coordinates = torch.linspace(*fragments.regions.window, 1000)
sizes = torch.linspace(0, 500, 50)

design = chd.utils.crossing(
    coordinate=coordinates,
    size=sizes,
    gene_ix=torch.tensor([gene_ix]),
    cluster_ix=torch.tensor(cluster_ixs),
)
design["coordinate2"] = design["coordinate"] + design["size"]
design = design.loc[design["coordinate2"] <= fragments.regions.window[1]]
design = design.loc[design["coordinate2"] >= fragments.regions.window[0]]

# %%
design["prob_left"], design["prob_right"] = model.evaluate_right(
    torch.from_numpy(design["coordinate"].values),
    torch.from_numpy(design["coordinate2"].values),
    gene_ix=torch.from_numpy(design["gene_ix"].values),
    window=fragments.regions.window,
    cluster_ix=torch.from_numpy(design["cluster_ix"].values),
)

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0))
width = 10
panel, ax = fig.main.add_under(polyptich.grid.Panel((width, 0.5)))
plotdata = design.loc[design["size"] == 0].set_index(["cluster_ix", "coordinate"])[["prob_left"]]
for cluster_ix, plotdata_cluster in plotdata.groupby("cluster_ix"):
    plotdata_cluster = plotdata_cluster.droplevel("cluster_ix").sort_index()
    ax.plot(plotdata_cluster.index, np.exp(plotdata_cluster["prob_left"]), label=cluster_ix)
ax.set_xlim(*fragments.regions.window)
ax.set_xticks([])

plotdata = simulation.peaks.query("gene == @design.gene_ix.iloc[0]").sort_values("center")

ax2 = panel.add_twinx()
ax2.scatter(plotdata["center"], [0] * len(plotdata), c=plotdata["size_mean"])
ax2.set_xlim(*fragments.regions.window)

panel, ax = fig.main.add_under(polyptich.grid.Panel((width, 2)))
plotdata = np.exp(design.groupby(["size", "coordinate"]).mean()["prob_right"].unstack())
ax.matshow(plotdata, aspect="auto", extent=(*fragments.regions.window, *plotdata.index[[-1, 0]]), cmap="viridis")
ax.set_xticks([])
# ax.set_yticks(np.arange(plotdata.shape[0]))
# ax.set_yticklabels(plotdata.index)

panel = fig.main.add_under(chd.data.motifscan.plot.Motifs(simulation.motifscan, gene, width=width))

fig.plot()

# %%
main = polyptich.grid.Grid(padding_height=0.1)
fig = polyptich.grid.Figure(main)

nbins = np.array(model.mixture.transform.nbins)
bincuts = np.concatenate([[0], np.cumsum(nbins)])
binmids = bincuts[:-1] + nbins / 2

ax = main[0, 0] = polyptich.grid.Panel((10, 0.25))
ax = ax.ax
plotdata = (model.mixture.transform.unnormalized_heights.data.cpu().numpy())[[gene_ix]]
ax.imshow(plotdata, aspect="auto")
ax.set_yticks([])
for b in bincuts:
    ax.axvline(b - 0.5, color="black", lw=0.5)
ax.set_xlim(0 - 0.5, plotdata.shape[1] - 0.5)
ax.set_xticks([])
ax.set_ylabel("$h_0$", rotation=0, ha="right", va="center")

ax = main[1, 0] = polyptich.grid.Panel(dim=(10, model.n_clusters * 0.25))
ax = ax.ax
plotdata = model.decoder.delta_height_weight.data[gene_ix].cpu().numpy()
ax.imshow(plotdata, aspect="auto", cmap=mpl.cm.RdBu_r, vmax=np.log(2), vmin=np.log(1 / 2))
ax.set_yticks(range(len(clustering.cluster_info)))
ax.set_yticklabels(clustering.cluster_info.index, rotation=0, ha="right")
for b in bincuts:
    ax.axvline(b - 0.5, color="black", lw=0.5)
ax.set_xlim(-0.5, plotdata.shape[1] - 0.5)

ax.set_xticks(bincuts - 0.5, minor=True)
ax.set_xticks(binmids - 0.5)
ax.set_xticklabels(nbins)
ax.xaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=5, which="minor")
ax.set_ylabel("$\Delta h$", rotation=0, ha="right", va="center")

ax.set_xlabel("Resolution")

fig.plot()

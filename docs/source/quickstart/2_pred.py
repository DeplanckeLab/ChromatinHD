# ---
# jupyter:
#   jupytext:
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

# %% [markdown]
# # ChromatinHD-*pred*

# %% tags=["hide_code", "hide_output"]
# autoreload
import IPython

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")

# %% tags=["hide_output"]
import chromatinhd as chd
import matplotlib.pyplot as plt

# %% [markdown]
# ChromatinHD-<i>pred</i> uses accessibility fragments to predict gene expression. As such, it can detect features such as broad or narrow positioning of fragments, or fragment sizes, that are predictive for gene expression.

# %% [markdown]
# We first load in all the input data which was created in the [data preparation tutorial](../1_data).

# %%
import pathlib

dataset_folder = pathlib.Path("example")
fragments = chd.data.Fragments(dataset_folder / "fragments")
transcriptome = chd.data.Transcriptome(dataset_folder / "transcriptome")
folds = chd.data.folds.Folds(dataset_folder / "folds" / "5x1")

# %% [markdown]
# ## Train the models

# %% [markdown]
# The basic ChromatinHD-*pred* model

# %%
models = chd.models.pred.model.multiscale.Models(dataset_folder / "models", reset=True)

# %% tags=["hide_output"]
models.train_models(
    fragments=fragments, transcriptome=transcriptome, folds=folds, regions_oi=transcriptome.gene_id(["CCL4", "IRF1"])
)

# %% [markdown]
# ## Some quality checks

# %% [markdown]
# We will first check whether the model learned something, by comparing the predictive performance with a baseline

# %%
gene_cors = models.get_region_cors(fragments, transcriptome, folds)
gene_cors["symbol"] = gene_cors.index.map(transcriptome.symbol)

# %%
gene_cors

# %%
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(4, 4))

for name, group in gene_cors.iterrows():
    ax.plot([0, 1], group[["cor_n_fragments", "cor"]], color="#3338", zorder=0, marker="o", markersize=2)
ax.boxplot(
    gene_cors[["cor_n_fragments", "cor"]].values,
    positions=[0, 1],
    widths=0.1,
    showfliers=False,
    showmeans=True,
    meanline=True,
    meanprops={"color": "red", "linewidth": 2},
)
ax.set_xticks([0, 1])
ax.set_ylim(0)
ax.set_xticklabels(["# fragments", "ChromatinHD-pred"])
ax.set_ylabel("$cor$")


# %% [markdown]
# ## Predictivity per position

# %% [markdown]
# To determine which regions were important for the model to predict gene expression, we will censor fragments from windows of various sizes, and then check whether the model performance on a set of test cells decreased. This functionality is implemented in the `GeneMultiWindow` class. This will only run the censoring for a subset of genes to speed up interpretation.

# %%
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window)
import chromatinhd
censorer.__class__ = chromatinhd.models.pred.interpret.censorers.MultiWindowCensorer
regionmultiwindow = chd.models.pred.interpret.RegionMultiWindow.create(
    path = models.path / "interpret" / "regionmultiwindow",
    folds = folds,
    transcriptome = transcriptome,
    censorer = censorer,
    fragments = fragments,
)

# %%
regionmultiwindow.score(
    models = models,
    regions = transcriptome.gene_id(
        [
            "CCL4",
            "IRF1",
        ]
    ),
    folds = folds,
)

# %%
regionmultiwindow.interpolate()

# %% [markdown]
# ### Visualizing predictivity

# %% [markdown]
# We can visualize the predictivity as follows. This shows which regions of the genome are positively and negatively associated with gene expression.

# %%
symbol = "IRF1"

fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05))
width = 10

region = fragments.regions.coordinates.loc[transcriptome.gene_id(symbol)]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width, genome = "GRCh38")
fig.main.add_under(panel_genes)

panel_pileup = chd.models.pred.plot.Pileup.from_regionmultiwindow(
    regionmultiwindow, transcriptome.gene_id(symbol), width=width
)
fig.main.add_under(panel_pileup)

panel_predictivity = chd.models.pred.plot.Predictivity.from_regionmultiwindow(
    regionmultiwindow, transcriptome.gene_id(symbol), width=width
)
fig.main.add_under(panel_predictivity)

fig.plot()

# %% [markdown]
# Given that accessibility can be sparse, we often simply visualize the predictivity in regions with at least a minimum of accessibility.

# %% [markdown]
# Let's first select regions based on the number of fragments. Regions that are close together will be merged.

# %%
symbol = "IRF1"
# symbol = "CCL4"
gene_id = transcriptome.gene_id(symbol)

# %%
# decrease the lost_cutoff to see more regions
regions = regionmultiwindow.select_regions(gene_id, lost_cutoff = 0.15)
breaking = polyptich.grid.Breaking(regions)

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05))

region = fragments.regions.coordinates.loc[transcriptome.gene_id(symbol)]
panel_genes = chd.plot.genome.genes.GenesBroken.from_region(region, breaking=breaking, genome = "GRCh38")
fig.main.add_under(panel_genes)

panel_pileup = chd.models.pred.plot.PileupBroken.from_regionmultiwindow(
    regionmultiwindow, gene_id, breaking=breaking
)
fig.main.add_under(panel_pileup)

panel_predictivity = chd.models.pred.plot.PredictivityBroken.from_regionmultiwindow(
    regionmultiwindow, gene_id, breaking=breaking, ymax = -0.1
)
fig.main.add_under(panel_predictivity)

fig.plot()

# %% [markdown] vscode={"languageId": "markdown"}
# ## Co-predictivity

# %% [markdown]
# In a similar fashion we can determine the co-predictivity per position.

# %%
censorer = chd.models.pred.interpret.WindowCensorer(fragments.regions.window)
regionpairwindow = chd.models.pred.interpret.RegionPairWindow(models.path / "interpret" / "regionpairwindow", reset=True)
regionpairwindow.score(models, censorer = censorer, folds = folds, fragments = fragments)

# %% [markdown]
# ### Visualization of co-predictivity

# %%
symbol = "IRF1"
# symbol = "CCL4"
gene_id = transcriptome.gene_id(symbol)

# %%
windows = regionmultiwindow.select_regions(gene_id, lost_cutoff = 0.2)
breaking = polyptich.grid.Breaking(windows)

# %%
fig = polyptich.grid.Figure(polyptich.grid.Grid(padding_height=0.05))
width = 10

# genes
region = fragments.regions.coordinates.loc[gene_id]
panel_genes = chd.plot.genome.genes.GenesBroken.from_region(region, breaking = breaking)
fig.main.add_under(panel_genes)

# pileup
panel_pileup = chd.models.pred.plot.PileupBroken.from_regionmultiwindow(
    regionmultiwindow, gene_id, breaking = breaking,
)
fig.main.add_under(panel_pileup)

# predictivity
panel_predictivity = chd.models.pred.plot.PredictivityBroken.from_regionmultiwindow(
    regionmultiwindow, gene_id, breaking=breaking
)
fig.main.add_under(panel_predictivity)

# copredictivity
panel_copredictivity = chd.models.pred.plot.CopredictivityBroken.from_regionpairwindow(
    regionpairwindow, gene_id, breaking = breaking
)
fig.main.add_under(panel_copredictivity, padding = 0.)

fig.plot()

# %%
plotdata = regionpairwindow.get_plotdata(gene_id, windows = windows).sort_values("cor")
plotdata["deltacor_min"] = plotdata[["deltacor1", "deltacor2"]].values.min(1)
plotdata["deltacor_max"] = plotdata[["deltacor1", "deltacor2"]].values.max(1)
plotdata["deltacor_prod"] = plotdata["deltacor1"] * plotdata["deltacor2"]
plotdata["deltacor_sum"] = plotdata["deltacor1"] + plotdata["deltacor2"]

fig, ax = plt.subplots()
ax.scatter(plotdata_oi["deltacor_prod"].abs(), plotdata_oi["cor"].abs())

# %% [markdown]
# ## Extract predictive regions

# %%
regionmultiwindow.extract_predictive_regions()

# %%

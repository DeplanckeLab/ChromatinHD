# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3
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
# We first load in all the input data:

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
models = chd.models.pred.model.additive.Models(dataset_folder / "models" / "additive", reset = True)

# %% tags=["hide_output"]
models.train_models(fragments, transcriptome, folds, device = "cuda")

# %% [markdown]
# ## Some quality checks

# %% [markdown]
# We will first check whether the model learned something, by comparing the predictive performance with a baseline 

# %%
gene_cors = models.get_gene_cors(fragments, transcriptome, folds, device = "cuda")
gene_cors["symbol"] = gene_cors.index.map(transcriptome.symbol)

# %%
gene_cors.sort_values("deltacor", ascending = False).head(10)

# %%
import pandas as pd
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize = (4, 4))

for name, group in gene_cors.iterrows():
    ax.plot([0, 1], group[["cor_n_fragments", "cor_predicted"]], color = "#3338", zorder = 0, marker = "o", markersize = 2)
ax.boxplot(gene_cors[["cor_n_fragments", "cor_predicted"]].values, positions = [0, 1], widths = 0.1, showfliers = False, showmeans = True, meanline = True, meanprops = {"color": "red", "linewidth": 2})
ax.set_xticks([0, 1])
ax.set_xticklabels(["# fragments", "ChromatinHD-pred"])
ax.set_ylabel("$cor$")
;

# %% [markdown]
# Note that every gene gains from the ChromatinHD model, even if some only gain a little. The genes with a low $\Delta cor$ are often those with only a few fragments:

# %%
fig, ax = plt.subplots(figsize = (4, 4))
ax.scatter(gene_cors["n_fragments"], gene_cors["deltacor"])
ax.set_ylabel("$\\Delta$ cor")
ax.set_xlabel("# fragments")
ax.set_xscale("log")

# %% [markdown]
# ## Predictivity per position

# %% [markdown]
# To determine which regions were important for the model to predict gene expression, we will censor fragments from windows of various sizes, and then check whether the model performance on a set of test cells decreased. This functionality is implemented in the `GeneMultiWindow` class. This will only run the censoring for a subset of genes to speed up interpretation.

# %%
censorer = chd.models.pred.interpret.MultiWindowCensorer(fragments.regions.window)
genemultiwindow = chd.models.pred.interpret.GeneMultiWindow(
    models.path / "interpret" / "genemultiwindow"
)

# %%
genemultiwindow.score(
    fragments,
    transcriptome,
    models,
    folds,
    transcriptome.gene_id(
        [
            "CCL4",
            "IL1B",
            "EBF1",
            "PAX5",
            "CD79A",
            "RHEX",
        ]
    ),
    censorer=censorer,
)

# %%
genemultiwindow.interpolate()

# %%
symbol = "EBF1"

fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05))
width = 10

region = fragments.regions.coordinates.loc[transcriptome.gene_id(symbol)]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width)
fig.main.add_under(panel_genes)

panel_pileup = chd.models.pred.plot.Pileup.from_genemultiwindow(
    genemultiwindow, transcriptome.gene_id(symbol), width=width
)
fig.main.add_under(panel_pileup)

panel_predictivity = chd.models.pred.plot.Predictivity.from_genemultiwindow(
    genemultiwindow, transcriptome.gene_id(symbol), width=width
)
fig.main.add_under(panel_predictivity)

fig.plot()

# %% [markdown]
# ## Co-predictivity per position

# %% [markdown]
# In a similar fashion we can determine the co-predictivity per position.

# %%
censorer = chd.models.pred.interpret.WindowCensorer(fragments.regions.window)
genepairwindow = chd.models.pred.interpret.GenePairWindow(
    models.path / "interpret" / "genepairwindow", reset = True
)
genepairwindow.score(fragments, transcriptome, models, folds, censorer = censorer, genes = transcriptome.gene_id(["CCL4"]))

# %%
symbol = "CCL4"

fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05))
width = 10

# genes
region = fragments.regions.coordinates.loc[transcriptome.gene_id(symbol)]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width)
fig.main.add_under(panel_genes)

# pileup
panel_pileup = chd.models.pred.plot.Pileup.from_genemultiwindow(
    genemultiwindow, transcriptome.gene_id(symbol), width=width
)
fig.main.add_under(panel_pileup)

# predictivity
panel_predictivity = chd.models.pred.plot.Predictivity.from_genemultiwindow(
    genemultiwindow, transcriptome.gene_id(symbol), width=width
)
fig.main.add_under(panel_predictivity)

# copredictivity
panel_copredictivity = chd.models.pred.plot.Copredictivity.from_genepairwindow(
    genepairwindow, transcriptome.gene_id(symbol), width=width
)
fig.main.add_under(panel_copredictivity)

fig.plot()

# %%

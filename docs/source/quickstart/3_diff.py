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
# # ChromatinHD-*diff*

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
clustering = chd.data.Clustering(dataset_folder / "clustering")

# %% [markdown]
# ## Train the models

# %% [markdown]
# The basic ChromatinHD-*diff* model

# %%
models = chd.models.diff.model.cutnf.Models(dataset_folder / "models" / "cutnf", reset=True)

# %% tags=["hide_output"]
models.train_models(fragments, clustering, folds)

# %% [markdown]
# ## Interpret positionally

# %% [markdown]
# Currently, the ChromatinHD-model is purely positional, i.e. it only looks whether Tn5 insertion sites increase or decrease within a region. As such, we can only interpret it positionally:

# %%
import chromatinhd.models.diff.interpret.genepositional

# %%
clustering.cluster_info.index.name = "cluster"

# %%
genepositional = chromatinhd.models.diff.interpret.genepositional.GenePositional(
    path=models.path / "interpret" / "genepositional"
)
genepositional.score(
    fragments,
    clustering,
    models,
    force=True,
)

# %%
symbol = "EBF1"

fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))
width = 10

region = fragments.regions.coordinates.loc[transcriptome.gene_id(symbol)]
panel_genes = chd.plot.genome.genes.Genes.from_region(region, width=width)
fig.main.add_under(panel_genes)

plotdata, plotdata_mean = genepositional.get_plotdata(transcriptome.gene_id(symbol))
panel_differential = chd.models.diff.plot.Differential(
    plotdata, plotdata_mean, cluster_info=clustering.cluster_info, panel_height=0.5, width=width
)
fig.main.add_under(panel_differential)

panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
    transcriptome=transcriptome, clustering=clustering, gene=transcriptome.gene_id(symbol), panel_height=0.5
)
fig.main.add_right(panel_expression, row=panel_differential)

fig.plot()

# %%

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
simulation = chd.simulation.simulate.Simulation(n_genes=50, window=[-10000, 10000])
simulation.create_regions()
simulation.create_obs()

simulation.create_fragments()

# %%
fragments = simulation.fragments
fragments.create_cellxgene_indptr()

# %%
clustering = simulation.clustering

# %%
folds = chd.data.folds.Folds()
folds.sample_cells(fragments, 5)
fold = folds[0]

# %% [markdown]
# ## Train

# %%
models = {}

# %%
model = chd.models.diff.model.cutnf.Model(
    fragments,
    clustering,
)
model.train_model(fragments, clustering, fold, n_epochs=30)
models["original"] = model

# %%
model = chd.models.diff.model.cutnf.Model(
    fragments,
    clustering,
    mixture_delta_p_scale=0.001,
)
model.train_model(fragments, clustering, fold, n_epochs=30)
models["baseline_orig"] = model

# %%
model = chd.models.diff.model.cutnf.Model(fragments, clustering, nbins=(128,))
model.train_model(fragments, clustering, fold, n_epochs=30)
models["original_128"] = model

# %%
model = chd.models.diff.model.cutnf.Model(fragments, clustering, nbins=(256, 128, 64, 32))
model.train_model(fragments, clustering, fold, n_epochs=30)
models["original_256,128,64,32"] = model

# %% [markdown]
# ## Score

# %%
scores = []
genescores = []
for model_id, model in models.items():
    prediction_test = model.get_prediction(fragments, clustering, cell_ixs=fold["cells_test"])
    prediction_validation = model.get_prediction(fragments, clustering, cell_ixs=fold["cells_validation"])
    prediction_train = model.get_prediction(fragments, clustering, cell_ixs=fold["cells_train"])
    scores.append(
        {
            "model_id": model_id,
            "lik_test": (prediction_test["likelihood_mixture"]).sum().item(),
            "n_test": len(fold["cells_test"]),
            "lik_validation": (prediction_validation["likelihood_mixture"]).sum().item(),
            "n_validation": len(fold["cells_validation"]),
            "lik_train": (prediction_train["likelihood_mixture"]).sum().item(),
            "n_train": len(fold["cells_train"]),
        }
    )
    genescores.append(
        pd.DataFrame(
            {
                "model_id": model_id,
                "lik_test": (prediction_test["likelihood_mixture"]).sum("cell").to_pandas(),
                "n_test": len(fold["cells_test"]),
                "lik_validation": (prediction_validation["likelihood_mixture"]).sum("cell").to_pandas(),
                "n_validation": len(fold["cells_validation"]),
                "lik_train": (prediction_train["likelihood_mixture"]).sum("cell").to_pandas(),
                "n_train": len(fold["cells_train"]),
            }
        )
    )
scores = pd.DataFrame(scores).set_index("model_id")
genescores = (
    pd.concat([genescores[i] for i in range(len(genescores))], axis=0).reset_index().set_index(["model_id", "gene"])
)

# %%
baseline_id = "baseline_orig"
scores[["lr_test", "lr_validation", "lr_train"]] = (
    scores[["lik_test", "lik_validation", "lik_train"]]
    - scores[["lik_test", "lik_validation", "lik_train"]].loc[baseline_id]
)
scores[["nlr_test", "nlr_validation", "nlr_train"]] = (
    scores[["lr_test", "lr_validation", "lr_train"]].values / scores[["n_test", "n_validation", "n_train"]].values
)
genescores[["lr_test", "lr_validation", "lr_train"]] = (
    genescores[["lik_test", "lik_validation", "lik_train"]]
    - genescores[["lik_test", "lik_validation", "lik_train"]].loc[baseline_id]
)
genescores[["nlr_test", "nlr_validation", "nlr_train"]] = (
    genescores[["lr_test", "lr_validation", "lr_train"]].values
    / genescores[["n_test", "n_validation", "n_train"]].values
)

# %%
model_info = pd.DataFrame({"model": models.keys()}).set_index("model")
model_info["model_type"] = model_info.index.map(lambda x: "_".join(x.split("_")[:-1]))
model_info = model_info.sort_values(["model_type"])
model_info["ix"] = np.arange(model_info.shape[0])

# %%
fig = chd.grid.Figure(chd.grid.Wrap(padding_width=0.1))
height = len(scores) * 0.2

plotdata = scores.copy().loc[model_info.index]

panel, ax = fig.main.add(chd.grid.Ax((1, height)))
ax.barh(plotdata.index, plotdata["lr_test"])
ax.axvline(0, color="black", linestyle="--", lw=1)
ax.set_title("Test")
ax.set_xlabel("Log-likehood ratio")

panel, ax = fig.main.add(chd.grid.Ax((1, height)))
ax.set_yticks([])
ax.barh(plotdata.index, plotdata["lr_validation"])
ax.axvline(0, color="black", linestyle="--", lw=1)
ax.set_title("Validation")

panel, ax = fig.main.add(chd.grid.Ax((1, height)))
ax.set_yticks([])
ax.barh(plotdata.index, plotdata["lr_train"])
ax.axvline(0, color="black", linestyle="--", lw=1)
ax.set_title("Train")
fig.plot()

# %% [markdown]
# ## Interpret

# %%
genepositional = chd.models.diff.interpret.genepositional.GenePositional(
    path=chd.get_output() / "interpret" / "genepositional"
)

# %%
gene = "G1"
model_id = "original"
# model_id = "original_128"
# model_id = "original_256,128,64,32"

genepositional.score(
    fragments,
    clustering,
    [models[model_id]],
    force=True,
    genes=[gene],
    # genes = transcriptome.gene_id([symbol])
)

# %%
fig = chd.grid.Figure(chd.grid.Grid(padding_height=0.05, padding_width=0.05))
width = 10

plotdata, plotdata_mean = genepositional.get_plotdata(gene)
panel_differential = chd.models.diff.plot.Differential(
    plotdata, plotdata_mean, cluster_info=clustering.cluster_info, panel_height=0.5, width=width
)
fig.main.add_under(panel_differential)

# panel_expression = chd.models.diff.plot.DifferentialExpression.from_transcriptome(
#     transcriptome = transcriptome, clustering = clustering, gene = transcriptome.gene_id(symbol), panel_height = 0.5
# )
# fig.main.add_right(panel_expression, row = panel_differential)

fig.plot()

# %%

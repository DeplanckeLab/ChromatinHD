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
# # Model promoters positionally

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

#export LD_LIBRARY_PATH=/data/peak_free_atac/software/peak_free_atac/lib
import torch
import torch_sparse

import tqdm.auto as tqdm

# %%
import peakfreeatac as pfa
import peakfreeatac.fragments
import peakfreeatac.transcriptome

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_clustered"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# %%
# promoter_name, window = "4k2k", (2000, 4000)
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
window_width = window[1] - window[0]

# %%
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")
fragments = peakfreeatac.fragments.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
fold = folds[0]

# %%
from design import get_folds_inference
folds = get_folds_inference(fragments, folds)

# %% [markdown]
# What we want is
# - Know which motifs are close to a cut sites. Either precalculate or extract on-the-fly as fast as possible
# - Extract all information of a motif in relation to the fragment
#   - Which motif
#   - Whether it is relative to the left or right cut site
#   - Its distance to the cut site
#   - Its score
# - We want to be able to index extremely fast on the fragments, ideally microseconds

# %% [markdown]
# ## Load fragments

# %%
import pyximport; import sys; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))
if "peakfreeatac.loaders.extraction.fragments" in sys.modules:
    del sys.modules['peakfreeatac.loaders.extraction.fragments']
import peakfreeatac.loaders.extraction.fragments

# %% [markdown]
# ### Fragments

# %%
n_cells = 1000
n_genes = 100
cutwindow = np.array([-150, 150])
loader = peakfreeatac.loaders.fragments.Fragments(fragments, n_cells * n_genes, window)

# %%
cells_oi = np.arange(0, n_cells)
genes_oi = np.arange(0, n_genes)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

minibatch = peakfreeatac.loaders.minibatching.Minibatch(cellxgene_oi = cellxgene_oi, cells_oi = cells_oi, genes_oi = genes_oi)
data = loader.load(minibatch)

# %%
# %%timeit -n 1
data = loader.load(minibatch)

# %%
(fragments.n_cells * fragments.n_genes) / len(cellxgene_oi)

# %% [markdown]
# ## Loading using multithreading

# %%
import pyximport; import sys; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))
if "peakfreeatac.loaders.extraction.fragments" in sys.modules:
    del sys.modules['peakfreeatac.loaders.extraction.fragments']
import peakfreeatac.loaders.extraction.fragments

# %%
# n_cells = 2000
n_cells = 3
n_genes = 100

cutwindow = np.array([-150, 150])

# %%
import peakfreeatac.loaders.fragments

# %%
loaders = peakfreeatac.loaders.pool.LoaderPool(
    peakfreeatac.loaders.fragments.Fragments,
    {"fragments":fragments, "cellxgene_batch_size":n_cells * n_genes, "window":window},
    n_workers = 2
)

# %%
import gc
gc.collect()

# %%
data = []
for i in range(2):
    cells_oi = np.sort(np.random.choice(fragments.n_cells, n_cells, replace = False))
    genes_oi = np.sort(np.random.choice(fragments.n_genes, n_genes, replace = False))

    cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()
    
    data.append(pfa.loaders.minibatching.Minibatch(cells_oi = cells_oi, genes_oi = genes_oi, cellxgene_oi=cellxgene_oi))
loaders.initialize(data)

# %%
for i, data in enumerate(tqdm.tqdm(loaders)):
    print(i)
    data
    # loaders.submit_next()

# %% [markdown]
# ## Single example

# %%
transcriptome.create_X()
transcriptome.X
mean_gene_expression = transcriptome.X.dense().mean(0)

# %%
from peakfreeatac.models.positional.v14 import Model

# %%
model = Model(loaders.loaders[0], fragments.n_genes, mean_gene_expression)

# %%
data.genes_oi

# %%
data.coordinates.shape

# %%
model.forward(data)

# %% [markdown]
# ## Infer

# %% [markdown]
# ### Loaders

# %%
import peakfreeatac.loaders
import peakfreeatac.loaders.fragments
import peakfreeatac.loaders.fragmentmotif

# %%
from design import get_design, get_folds_training

# %%
design = get_design(dataset_name, transcriptome, motifscores, fragments, window)

# %%
# prediction_name = "v4_10-10"
# prediction_name = "v4_1-1"
# prediction_name = "v4_1k-1k"
# prediction_name = "v4"
prediction_name = "v4_baseline"
design_row = design[prediction_name]

# %%
# loaders
print("collecting...")
if "loaders" in globals():
    loaders.terminate()
    del loaders
    import gc
    gc.collect()
if "loaders_validation" in globals():
    loaders_validation.terminate()
    del loaders_validation
    import gc
    gc.collect()
print("collected")
loaders = pfa.loaders.LoaderPool(
    design_row["loader_cls"],
    design_row["loader_parameters"],
    n_workers = 20
)
print("haha!")
loaders_validation = pfa.loaders.LoaderPool(
    design_row["loader_cls"],
    design_row["loader_parameters"],
    n_workers = 5
)
loaders_validation.shuffle_on_iter = False

# %%
# folds & minibatching
folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
folds = get_folds_training(fragments, folds)


# %%
# loss
# cos = torch.nn.CosineSimilarity(dim = 0)
# loss = lambda x_1, x_2: -cos(x_1, x_2).mean()

def paircor(x, y, dim = 0, eps = 0.1):
    divisor = (y.std(dim) * x.std(dim)) + eps
    cor = ((x - x.mean(dim, keepdims = True)) * (y - y.mean(dim, keepdims = True))).mean(dim) / divisor
    return cor
loss = lambda x, y: -paircor(x, y).mean() * 100

# mse_loss = torch.nn.MSELoss()
# loss = lambda x, y: mse_loss(x, y) * 100

# def zscore(x, dim = 0):
#     return (x - x.mean(dim, keepdim = True)) / (x.std(dim, keepdim = True) + 1e-6)
# mse_loss = torch.nn.MSELoss()
# loss = lambda x, y: mse_loss(zscore(x), zscore(y)) * 10.

# %%
class Prediction(pfa.flow.Flow):
    pass

print(prediction_name)
prediction = Prediction(pfa.get_output() / "prediction_sequence" / dataset_name / promoter_name / prediction_name)

# %%
fold_ix = 0
fold = folds[0]

# %%
# initialize loaders
loaders.initialize(fold["minibatches_train"])
loaders_validation.initialize(fold["minibatches_validation_trace"])

# %%
# data = loaders.pull()
# data.motifcounts.sum()

# motifs["total"] = np.array((motifscores > 0).sum(0))[0]
# motifs["n"] = np.array((data.motifcounts > 0).sum(0))
# motifs["score"] = np.log(motifs["n"] / motifs["total"])
# motifs["score"] .sort_values().plot(kind = "hist")

# %%
n_epochs = 150
checkpoint_every_step = 100

# n_epochs = 10
# checkpoint_every_step = 30

# %%
# model
model = design_row["model_cls"](**design_row["model_parameters"])

# %%
## optimizer
params = model.get_parameters()

# optimization
optimize_every_step = 1
lr = 1e-2# / optimize_every_step
optim = torch.optim.Adam(params, lr=lr)

# train
import peakfreeatac.train
outcome = transcriptome.X.dense()
trainer = pfa.train.Trainer(
    model,
    loaders,
    loaders_validation,
    outcome,
    loss,
    optim,
    checkpoint_every_step = checkpoint_every_step,
    optimize_every_step = optimize_every_step,
    n_epochs = n_epochs,
    device = "cuda"
)
trainer.train()

# %%
pd.DataFrame(trainer.trace.validation_steps).groupby("checkpoint").mean()["loss"].plot(label = "validation")

# %%
pd.DataFrame(trainer.trace.train_steps).groupby("checkpoint").mean()["loss"].plot(label = "train")

# %%
# model = model.to("cpu")
pickle.dump(model, open("../../" + dataset_name + "_" + "baseline_model.pkl", "wb"))

# %%
model = model.to("cpu")
pickle.dump(model, open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "wb"))

# %% [markdown]
# -----

# %%
model = model.to("cpu")
pickle.open(open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "rb"))

# %%
folder_motifs = pfa.get_output() / "data" / "motifs" / "hs" / "hocomoco"
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs = pickle.load((folder_data_preproc / "motifs_oi.pkl").open("rb"))

# %%
motif_linear_scores = pd.Series(
    model.embedding_gene_pooler.nn1[0].weight.detach().cpu().numpy()[0],
    # model.embedding_gene_pooler.motif_bias.detach().cpu().numpy(),
    index = motifs.index
).sort_values(ascending = False)
# motif_linear_scores = pd.Series(model.embedding_gene_pooler.linear_weight[0].detach().cpu().numpy(), index = motifs_oi.index).sort_values(ascending = False)

# %%
motif_linear_scores.plot(kind = "hist")

# %%
motifs["n"] = pd.Series(data.motifcounts.sum(0).numpy(), motifs.index)
motifs["n"] = np.array((motifscores > 0).sum(0))[0]

# %%
motif_linear_scores["E2F3_HUMAN.H11MO.0.A"]

# %%
motif_linear_scores.head(10)

# %%
motif_linear_scores.tail(10)

# %%
motif_linear_scores.loc[motif_linear_scores.index.str.startswith("CEBP")]

# %%
motif_oi = motifs.query("gene_label == 'CEBPA'").index[0]
sns.heatmap(pwms[motif_oi].numpy())

# %%
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["ZNF329", "E2F5", "DBP", "BACH1", "FOXO4"]))

# %%
motifs_oi = motifs.loc[motifs["gene_label"].isin(transcriptome.var["symbol"])]

# %%
plotdata = pd.DataFrame({
    "is_variable":motifs["gene_label"].isin(transcriptome.var["symbol"]),
    "linear_score":motif_linear_scores
})
plotdata["dispersions_norm"] = pd.Series(
    transcriptome.var["dispersions_norm"][transcriptome.gene_id(motifs_oi["gene_label"]).values].values,
    index = motifs_oi.index
)

# %%
plotdata.groupby("is_variable").std()

# %%
sns.scatterplot(plotdata.dropna(), y = "linear_score", x = "dispersions_norm")

# %%
sns.stripplot(plotdata, y = "linear_score", x = "is_variable")

# %%
exp = pd.DataFrame(
    transcriptome.adata[:, transcriptome.gene_id(motifs_oi["gene_label"].values).values].X.todense(),
    index = transcriptome.adata.obs.index,
    columns = motifs_oi.index
)

# %%

# %%
sc.get.obs_df(transcriptome.adata, transcriptome.gene_id(motifs_oi["gene_label"].values).values)

# %%
transcriptome.var

# %%
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["PAX5", "BACH1"]))

# %%
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["STAT2"]))

# %%

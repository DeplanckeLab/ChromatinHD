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
# # Model promoters

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')

import pickle

import scanpy as sc
import pathlib

import torch

import tqdm.auto as tqdm

# %%
import peakfreeatac as pfa
import peakfreeatac.fragments
import peakfreeatac.transcriptome

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"
folder_data_preproc = folder_data / "lymphoma"
folder_data_preproc.mkdir(exist_ok = True, parents = True)

# %%
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")
fragments = peakfreeatac.fragments.Fragments(folder_data_preproc / "fragments")

# %%
# # !pip install torch==1.12.1 --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu113

# torch-sparse gives an error with torch 1.13
# that's why we need 1.12.1 for now
# # !pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html --force-reinstall

# %%
import torch_scatter
import torch

# %% [markdown]
# ## Single example

# %% [markdown]
# ### Subset cells and genes

# %%
gene_start = 0
gene_end = 100
gene_n = gene_end - gene_start
gene_idx = slice(gene_start, gene_end)

cell_start = 1500
cell_end = 2000
cell_n = cell_end - cell_start
cell_idx = slice(cell_start, cell_end)

# %%
fragments_selected = (
    (fragments.mapping[:, 0] >= cell_start) &
    (fragments.mapping[:, 0] < cell_end) &
    (fragments.mapping[:, 1] >= gene_start) &
    (fragments.mapping[:, 1] < gene_end)
)
print(fragments_selected.sum())

# %%
fragment_coordinates = fragments.coordinates[fragments_selected]
print(fragment_coordinates.shape)

fragments_mapping = fragments.mapping[fragments_selected]

# we should adapt this if the minibatch cells/genes would ever be non-contiguous
local_cell_idx = fragments_mapping[:, 0] - cell_start
local_gene_idx = fragments_mapping[:, 1] - gene_start

fragment_cellxgene_idx = local_cell_idx * gene_n + local_gene_idx

# %% [markdown]
# ### Create expression

# %%
transcriptome.create_X()
transcriptome.X

# %%
transcriptome.X.shape

# %%
mean_gene_expression = transcriptome.X.dense().mean(0)

# %% [markdown]
# ### Create fragment embedder

# %%
from peakfreeatac.models.promoter.v1 import FragmentEmbedder, FragmentEmbedderCounter

# %%
n_embedding_dimensions = 1000
fragment_embedder = FragmentEmbedder(n_virtual_dimensions = 100, n_embedding_dimensions = n_embedding_dimensions)
# fragment_embedder = FragmentEmbedderCounter()

# %%
fragment_embedding = fragment_embedder(fragment_coordinates)

# %% [markdown]
# ### Pool fragments

# %%
from peakfreeatac.models.promoter.v1 import EmbeddingGenePooler

# %%
embedding_gene_pooler = EmbeddingGenePooler(debug = True)

# %%
cell_gene_embedding = embedding_gene_pooler(fragment_embedding, fragment_cellxgene_idx, cell_n, gene_n)

# %% [markdown]
# ### Create expression predictor

# %%
from peakfreeatac.models.promoter.v1 import EmbeddingToExpression

# %%
embedding_to_expression = EmbeddingToExpression(fragments.n_genes, n_embedding_dimensions = n_embedding_dimensions, mean_gene_expression = mean_gene_expression)
# embedding_to_expression = EmbeddingToExpressionBias(fragments.n_genes, n_embedding_dimensions = n_embedding_dimensions, mean_gene_expression = mean_gene_expression)

# %%
expression_predicted = embedding_to_expression(cell_gene_embedding, gene_idx)

# %% [markdown]
# ### Whole model

# %%
from peakfreeatac.models.promoter.v1 import FragmentsToExpression

# %%
model = FragmentsToExpression(fragments.n_genes, mean_gene_expression)

# %%
model(fragment_coordinates, fragment_cellxgene_idx, cell_n, gene_n, gene_idx)

# %% [markdown]
# ## Train

# %% [markdown]
# ### Create training and test split

# %%
from peakfreeatac.models.promoter.v1 import Split

# %%
n_cell_step = 1000
n_gene_step = 3000

# %%
mapping_x = fragments.mapping[:, 0] * fragments.n_genes + fragments.mapping[:, 1]

# %%
import itertools

# %%
cell_start_test = int((fragments.n_cells / 3)*2) # split train/test cells

# %%
splits = []

prev_fragment_idx_end = -1

starts = list(itertools.product(np.arange(fragments.n_cells, step = n_cell_step), np.arange(fragments.n_genes, step = n_gene_step)))
for cell_start, gene_start in tqdm.tqdm(starts):
    cell_end = min(cell_start + n_cell_step, fragments.n_cells)
    gene_end = min(gene_start + n_gene_step, fragments.n_genes)
    
    phase = "train" if cell_start < cell_start_test else "test"

    fragments_selected = torch.where(
        (fragments.mapping[:, 0] >= cell_start) &
        (fragments.mapping[:, 0] < cell_end) &
        (fragments.mapping[:, 1] >= gene_start) &
        (fragments.mapping[:, 1] < gene_end)
    )[0]
    
    cell_n = cell_end - cell_start
    gene_n = gene_end - gene_start

    gene_idx = slice(gene_start, gene_end)
    cell_idx = slice(cell_start, cell_end)

    fragments_coordinates = fragments.coordinates[fragments_selected]
    fragments_mappings = fragments.mapping[fragments_selected]

    # we should adapt this if the minibatch cells/genes would ever be non-contiguous
    local_cell_idx = fragments_mappings[:, 0] - cell_start
    local_gene_idx = fragments_mappings[:, 1] - gene_start
    
    split = Split(
        cell_start = cell_start,
        cell_end = cell_end,
        gene_start = gene_start,
        gene_end = gene_end,
        cell_n = cell_n,
        gene_n = gene_n,
        local_cell_idx = local_cell_idx,
        local_gene_idx = local_gene_idx,
        fragments_coordinates = fragments_coordinates,
        fragments_mappings = fragments_mappings,
        phase = phase,
        fragments_selected = fragments_selected
    )
    splits.append(split)

# %%
splits_training = [split for split in splits if split.phase == "train"]
splits_test = [split for split in splits if split.phase == "test"]

# %% [markdown]
# ### Create model

# %%
n_embedding_dimensions = 100

# %%
model = FragmentsToExpression(fragments.n_genes, mean_gene_expression)

# %% [markdown]
# ### Train

# %%
splits_training = [split.to("cuda") for split in splits_training]
transcriptome_X = transcriptome.X.to("cuda")
model = model.to("cuda")

# %% [markdown]
# Note: we cannot use ADAM!
# Because it has momentum.
# And not all parameters are optimized (or have a grad) at every step, because we filter by gene.

# %%
params = model.parameters()
optim = torch.optim.SGD(params, lr = 10.0)
loss = torch.nn.MSELoss()

# %%
n_steps = 2000
trace_epoch_every = 10

# %%
trace = []

prev_epoch_mse = None
for epoch in range(n_steps):
    for split in splits_training:
        expression_predicted = model(
            split.fragments_coordinates,
            split.fragment_cellxgene_idx,
            split.cell_n,
            split.gene_n,
            split.gene_idx
        )
        
        transcriptome_subset = transcriptome_X.dense_subset(split.cell_idx)[:, split.gene_idx]
        
        mse = loss(expression_predicted, transcriptome_subset)
        
        mse.backward()
        optim.step()
        optim.zero_grad()
        
        trace.append({"mse":mse.detach().cpu().numpy(), "epoch":epoch})
    epoch_mse = np.mean([t["mse"] for t in trace[-len(splits):]])
    
    if (epoch % trace_epoch_every) == 0:
        print(f"{epoch} {epoch_mse:.4f} Δ{prev_epoch_mse-epoch_mse if prev_epoch_mse is not None else ''}")
    
    prev_epoch_mse = epoch_mse

# %%
trace = pd.DataFrame(list(trace))

# %%
fig, ax = plt.subplots()
plotdata = trace.groupby("epoch").mean().reset_index()
ax.scatter(trace["epoch"], trace["mse"])
ax.plot(plotdata["epoch"], plotdata["mse"], zorder = 6, color = "red")


# %%
def paircor(x, y, dim = 0):
    return ((x - x.mean(dim, keepdims = True)) * (y - y.mean(dim, keepdims = True))).mean(dim) / (y.std(dim) * x.std(dim))


# %%
def fix_class(obj):
    import importlib

    module = importlib.import_module(obj.__class__.__module__)
    cls = getattr(module, obj.__class__.__name__)
    try:
        obj.__class__ = cls
    except TypeError:
        pass


# %%
class Pickler(pickle.Pickler):
    def reducer_override(self, obj):
        if any(
            obj.__class__.__module__.startswith(module)
            for module in ["peakfreeatac."]
        ):
            fix_class(obj)
        else:
            # For any other object, fallback to usual reduction
            return NotImplemented

        return NotImplemented


# %%
def save(obj, fh, pickler=None, **kwargs):
    if pickler is None:
        pickler = Pickler
    return pickler(fh).dump(obj)


# %%
# move splits back to cpu
# otherwise if you try to load them back in they might want to immediately go to gpu
splits = [split.to("cpu") for split in splits]
save(splits, open("splits.pkl", "wb"))
save(model, open("model.pkl", "wb"))

# %%

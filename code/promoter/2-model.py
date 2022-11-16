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

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
folder_data_preproc = folder_data / dataset_name

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
import peakfreeatac.fragments

# %%
split = pfa.fragments.Split(slice(0, 100), slice(1500, 2000))

# %%
split.populate(fragments)

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
fragment_embedding = fragment_embedder(split.fragments_coordinates)

# %% [markdown]
# ### Pool fragments

# %%
from peakfreeatac.models.promoter.v1 import EmbeddingGenePooler

# %%
embedding_gene_pooler = EmbeddingGenePooler(debug = True)

# %%
cell_gene_embedding = embedding_gene_pooler(fragment_embedding, split.fragment_cellxgene_idx, split.cell_n, split.gene_n)

# %% [markdown]
# ### Create expression predictor

# %%
from peakfreeatac.models.promoter.v1 import EmbeddingToExpression

# %%
embedding_to_expression = EmbeddingToExpression(fragments.n_genes, n_embedding_dimensions = n_embedding_dimensions, mean_gene_expression = mean_gene_expression)
# embedding_to_expression = EmbeddingToExpressionBias(fragments.n_genes, n_embedding_dimensions = n_embedding_dimensions, mean_gene_expression = mean_gene_expression)

# %%
expression_predicted = embedding_to_expression(cell_gene_embedding, split.gene_idx)

# %% [markdown]
# ### Whole model

# %%
from peakfreeatac.models.promoter.v1 import FragmentsToExpression

# %%
model = FragmentsToExpression(fragments.n_genes, mean_gene_expression)

# %%
model(split.fragments_coordinates, split.fragment_cellxgene_idx, split.cell_n, split.gene_n, split.gene_idx)

# %% [markdown]
# ## Train

# %% [markdown]
# ### Create training and test split

# %%
# from peakfreeatac.models.promoter.v1 import Split
from peakfreeatac.fragments import Split

# %%
n_cell_step = 1000
n_gene_step = 5000

# %%
mapping_x = fragments.mapping[:, 0] * fragments.n_genes + fragments.mapping[:, 1]

# %%
import itertools

# %%
test_cell_cutoff = int((fragments.n_cells / 3)*2) # split train/test cells

# %%
cell_cuts = [
    *np.arange(0, test_cell_cutoff, step = n_cell_step),
    *np.arange(test_cell_cutoff, fragments.n_cells, step = n_cell_step),
    fragments.n_cells
]
cell_bins = [slice(a, b) for a, b in zip(cell_cuts[:-1], cell_cuts[1:])]
gene_cuts = list(np.arange(fragments.n_genes, step = n_gene_step)) + [fragments.n_genes]
gene_bins = [slice(a, b) for a, b in zip(gene_cuts[:-1], gene_cuts[1:])]
bins = list(itertools.product(cell_bins, gene_bins))

# %%
splits = []

for cell_idx, gene_idx in tqdm.tqdm(bins):
    split = Split(cell_idx, gene_idx, phase = "train" if cell_idx.start < test_cell_cutoff else "validation")
    split.populate(fragments)

    splits.append(split)

# %%
splits_training = [split for split in splits if split.phase == "train"]
splits_test = [split for split in splits if split.phase == "test"]

# %% [markdown]
# ### Create model

# %%
n_embedding_dimensions = 200

# %%
model = FragmentsToExpression(
    fragments.n_genes,
    mean_gene_expression,
    n_embedding_dimensions = n_embedding_dimensions
)

# %%
# if you want to reload a model
# make sure to run the "prediction = " first which is down the notebook
# model = pickle.load(open(prediction.path / "model.pkl", "rb"))

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
import itertools

# %%
params = model.parameters()
optim1 = torch.optim.SGD(itertools.chain(model.fragment_embedder.parameters(), model.embedding_gene_pooler.parameters()), lr = 10.)
optim2 = torch.optim.SGD(itertools.chain(model.embedding_to_expression.parameters()), lr = 10.)
loss = torch.nn.MSELoss()

# %%
n_steps = 2000
trace_epoch_every = 10

# %%
transcriptome_X_dense = transcriptome_X.dense()

# %%
trace = []

prev_epoch_mse = None
prev_epoch_gene_mse = None
for epoch in range(n_steps):
    for split in splits_training:
        # start = time.time()
        expression_predicted = model(
            split.fragments_coordinates,
            split.fragment_cellxgene_idx,
            split.cell_n,
            split.gene_n,
            split.gene_idx
        )
        
        # transcriptome_subset = transcriptome_X.dense_subset(split.cell_idx)[:, split.gene_idx]
        transcriptome_subset = transcriptome_X_dense[split.cell_idx, split.gene_idx]
        
        mse = loss(expression_predicted, transcriptome_subset)
        
        mse.backward()
        optim1.step()
        optim2.step()
        optim2.zero_grad()
        optim1.zero_grad()
        
        trace.append({
            "mse":mse.detach().cpu().numpy(),
            "epoch":epoch,
            "gene_mse":pd.Series(((expression_predicted - transcriptome_subset) ** 2).mean(0).detach().cpu().numpy(), split.gene_idxs)
        })
        
        # end = time.time()
        # print(end - start)
    
    if (epoch % trace_epoch_every) == 0:
        # mse
        epoch_mse = np.mean([t["mse"] for t in trace[-len(splits):]])
        text = f"{epoch} {epoch_mse:.4f}"
        
        if prev_epoch_mse is not None:
            text += f" Δ{prev_epoch_mse-epoch_mse:.1e}"
    
        prev_epoch_mse = epoch_mse
        
        # gene_mse
        if len(trace) > 0:
            epoch_gene_mse = pd.concat([t["gene_mse"] for t in trace[-len(splits):]]).groupby(level = 0).mean()
            if prev_epoch_gene_mse is not None:
                text += " {0:.1%}".format((epoch_gene_mse < prev_epoch_gene_mse).mean())
            prev_epoch_gene_mse = epoch_gene_mse
        
        print(text)

# %%
trace = pd.DataFrame(list(trace))

# %%
fig, ax = plt.subplots()
plotdata = trace.groupby("epoch").mean().reset_index()
ax.scatter(trace["epoch"], trace["mse"])
ax.plot(plotdata["epoch"], plotdata["mse"], zorder = 6, color = "red")


# %%
def fix_class(obj):
    import importlib

    module = importlib.import_module(obj.__class__.__module__)
    cls = getattr(module, obj.__class__.__name__)
    try:
        obj.__class__ = cls
    except TypeError:
        pass

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
    
def save(obj, fh, pickler=None, **kwargs):
    if pickler is None:
        pickler = Pickler
    return pickler(fh).dump(obj)


# %%
class Prediction(pfa.flow.Flow):
    pass
prediction = Prediction(pfa.get_output() / "prediction_promoter" / dataset_name / "nn")

# %%
# move splits back to cpu
# otherwise if you try to load them back in they might want to immediately go to gpu
splits = [split.to("cpu") for split in splits]

save(splits, open(prediction.path / "splits.pkl", "wb"))
save(model, open(prediction.path / "model.pkl", "wb"))

# %%

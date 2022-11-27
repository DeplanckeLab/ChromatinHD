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
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# %%
# promoter_name, (padding_negative, padding_positive) = "4k2k", (2000, 4000)
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)

promoter_size = padding_negative + padding_positive

# %%
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")
fragments = peakfreeatac.fragments.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
# # !pip install torch==1.12.1 --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu113

# torch-sparse gives an error with torch 1.13
# that's why we need 1.12.1 for now
# # !pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html --force-reinstall

# %%
import torch_scatter
import torch

# %%
folds = pickle.load(open(fragments.path / "folds.pkl", "rb"))

# %% [markdown]
# ## Single example

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
# ### Subset cells and genes

# %%
import peakfreeatac.fragments

# %%
split = pfa.fragments.Split(torch.arange(0, 100), slice(1500, 2000))

# %%
# fragments.create_cell_fragment_mapping()

# %%
split.populate(fragments)

# %% [markdown]
# ### Create fragment embedder

# %%
from peakfreeatac.models.promoter.v1 import FragmentEmbedderCounter
from peakfreeatac.models.promoter.v5 import FragmentEmbedder

# %%
# fragment_embedder = FragmentEmbedder()
fragment_embedder = FragmentEmbedder()
# fragment_embedder = FragmentEmbedderCounter()

# %%
coordinates = torch.stack([torch.arange(-padding_negative, padding_positive - 200, 200), torch.arange(-padding_negative + 200, padding_positive, 200)], -1)
global_gene_ix = torch.zeros((coordinates.shape[0], ), dtype = torch.long)
fragment_embedding = fragment_embedder(coordinates)
# fragment_embedding = fragment_embedder(coordinates, global_gene_ix)
# fragment_embedding = fragment_embedder(coordinates, global_gene_ix, {2:[]})

# %%
d = 20
k = torch.arange(20)

# %%
1/(10000)**(2 * k / d)

# %%
n_frequencies = 20
torch.tensor([[1 / 100**(2 * i/n_frequencies)] * 2 for i in range(1, n_frequencies + 1)]).flatten(-2)

# %%
fragment_embedder.frequencies

# %%
if fragment_embedding.ndim == 1:
    fragment_embedding = fragment_embedding[:, None]

# %%
fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharey = True)
sns.heatmap(coordinates.numpy(), ax = axes[0])
sns.heatmap(fragment_embedding.detach().numpy(), ax = axes[1])
axes[0].set_ylabel("Fragment")

# %%
from collections import Counter
counts = pd.Series(Counter(split.fragment_cellxgene_ix.numpy()))
pd.Series(counts).plot(kind = "hist", range = (0, 10), bins = 10)

# %%
# fragment_embedding = fragment_embedder(fragments.coordinates[split.fragments_selected])
fragment_embedding = fragment_embedder(fragments.coordinates[split.fragments_selected], fragments.mapping[split.fragments_selected, 1])

# %%
cellxgene_idxs = split.count_mapper[2]
x_stacked = fragment_embedding[cellxgene_idxs]
x = x.reshape((x_stacked.shape[0]//2, 2, *x_stacked.shape[1:]))


# %%
def self_attention(x):
    dotproduct = torch.matmul(x, x.transpose(-1, -2))
    weights = torch.nn.functional.softmax(dotproduct, -1)
    y = torch.matmul(weights, x)
    return y


# %%
dotproduct = torch.matmul(x, x.transpose(-1, -2))
weights = torch.nn.functional.softmax(dotproduct, -1)

# %%
y = torch.matmul(weights, x)

# %%
idx = 0
fig, ax = plt.subplots()
ax.set_aspect(1)
plt.scatter(x[idx][1].detach().numpy(), y[idx][1].detach().numpy())

# %%
y.reshape(x_stacked.shape).shape

# %% [markdown]
# ### Pool fragments

# %%
from peakfreeatac.models.promoter.v1 import EmbeddingGenePooler

# %%
embedding_gene_pooler = EmbeddingGenePooler(debug = True)

# %%
cell_gene_embedding = embedding_gene_pooler(fragment_embedding, split.fragment_cellxgene_ix, split.cell_n, split.gene_n)

# %%
cell_gene_embedding.detach().numpy().flatten().max()

# %%
plt.hist(cell_gene_embedding.detach().numpy().flatten(), range = (0, 10), bins = 10)

# %% [markdown]
# ### Create expression predictor

# %%
from peakfreeatac.models.promoter.v10 import EmbeddingToExpression

# %%
embedding_to_expression = EmbeddingToExpression(fragments.n_genes, n_embedding_dimensions = fragment_embedder.n_embedding_dimensions, mean_gene_expression = mean_gene_expression)
# embedding_to_expression = EmbeddingToExpressionBias(fragments.n_genes, n_embedding_dimensions = n_embedding_dimensions, mean_gene_expression = mean_gene_expression)

# %%
expression_predicted = embedding_to_expression(cell_gene_embedding, split.gene_ix)

# %%
expression_predicted

# %% [markdown]
# ### Whole model

# %%
from peakfreeatac.models.promoter.v10 import FragmentsToExpression

# %%
model = FragmentsToExpression(fragments.n_genes, mean_gene_expression)

# %%
params = [{"params":[]}]
for param in model.get_parameters():
    if torch.is_tensor(param):
        params[0]["params"].append(param)
    else:
        params.append(param)

print(len(params))

# %% [markdown]
# ## COO vs CSR

# %%
split.fragment_cellxgene_ix

# %%
fragment_i = 0
cellxgene_i = 0
idptr = []
for fragment_i, cellxgene in enumerate(split.fragment_cellxgene_ix):
    while cellxgene_i < cellxgene:
        idptr.append(fragment_i)
        cellxgene_i += 1

n_cellxgene = split.cell_n * split.gene_n
while cellxgene_i < n_cellxgene:
    idptr.append(fragment_i)
    cellxgene_i += 1

idptr = torch.tensor(idptr)
len(idptr)

folds[0][0].fragment_cellxgene_ix

# %%
coordinates = fragments.coordinates[split.fragments_selected].to("cuda:1")[:, [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
fragment_cellxgene_ix = split.fragment_cellxgene_ix.to("cuda:1")
idptr = idptr.to("cuda:1")

# %%
# %%timeit -n 100
y = torch_scatter.segment_sum_coo(coordinates, fragment_cellxgene_ix, dim_size = n_cellxgene)

# %%
y.shape

# %%
# %%timeit -n 100
y = torch_scatter.segment_sum_csr(coordinates, idptr)

# %%
y.shape

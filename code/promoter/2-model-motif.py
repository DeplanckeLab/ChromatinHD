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
# # Model promoters with motifs

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
motifscores = pickle.load(open(folder_data_preproc / "motifscores.pkl", "rb")).transpose((1, 2, 0))

# %%
# # !pip install torch==1.12.1 --force-reinstall --extra-index-url https://download.pytorch.org/whl/cu113

# torch-sparse gives an error with torch 1.13
# that's why we need 1.12.1 for now
# # !pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html --force-reinstall

# %%
import torch_scatter
import torch

# %%
folds = pickle.load(open(fragments.path / "folds2.pkl", "rb"))

# %%
split = folds[0][0]

# %%
# gather mean of motif score
# fragment_motif_scores = torch.zeros((1, motifscores.shape[0], len(split.fragments_selected)))
# for i, (gene_ix, start, end) in tqdm.tqdm(
#     enumerate(zip(fragments.mapping[split.fragments_selected, 1].numpy(), fragments.coordinates.numpy()[:, 0], fragments.coordinates.numpy()[:, 1])),
#     total = len(split.fragments_selected),
#     leave = False
# ):
#     fragment_motif_scores[0, :, i] = motifscores[:, gene_ix, max(start - 100 + padding_negative, 0):(end+100 + padding_negative)].max(1)[0][0]

# %%
import scipy.sparse

# %%
cut_padding_left = 150
cut_padding_right = 150

# %%
mapping = fragments.mapping
coordinates = fragments.coordinates

mapping = fragments.mapping[split.fragments_selected]
coordinates = fragments.coordinates[split.fragments_selected]

# %%
width = motifscores.shape[-1]

# %%
motifscores_count = (motifscores > 0).to(torch.int)
motifscores_view = motifscores_count.view((motifscores_count.shape[0], np.prod(motifscores_count.shape[1:]))).transpose(1, 0).clone()

# %%
fold = folds[0]

# %%
mapping_full = fragments.mapping[split.fragments_selected]
coordinates_full = fragments.coordinates[split.fragments_selected]

motifscores_view = motifscores_view.to("cuda")

# %%
idxs = torch.clamp(coordinates[:, 0] + torch.arange(-cut_padding_left, cut_padding_right, device = mapping.device)[:, None] + padding_negative, 0, width-1)
unwrapped_idx = idxs + mapping[:, 1] * width

# %%
max_n = 100000
fragment_cuts = np.arange(split.fragments_selected.sum(), step = max_n)
fragment_slices = [(a, b) for a, b in zip(fragment_cuts, np.hstack([fragment_cuts[1:], [np.iinfo(np.int32).max]]))]

for fragment_start, fragment_end in fragment_slices:
    mapping = mapping_full[:
    coordinates = fragments.coordinates[split.fragments_selected]

# %%
motifscores.shape


# %%
def pool_cut_windows(
    mapping,
    coordinates,
    motifscores_view,
    window,
    window_width,
    cutwindow,
    cutwindow_width
):
    assert coordinates.ndim == 2
    assert coordinates.shape[1] == 2
    assert window[1] - window[0] == window_width
    assert cutwindow[1] - cutwindow[0] == cutwindow_width
    assert (motifscores_view.shape[0] % window_width) == 0
    
    # has dimensions [fragments, cut sites(0, 1), positions(cutwindow_width)]
    idxs = coordinates[:, :, None] - window[0] + torch.arange(cutwindow[0], cutwindow[1]+1, device = coordinates.device)[None, None, :]
    idxs = torch.clamp(idxs, 0, window_width-1)
    
    # flatten index along different genes using second mapping column
    unwrapped_idx = (idxs + mapping[:, 1][:, None, None] * window_width)

    # extract the relevant scores, has dimensions [positions x fragments, channels(motifs)]
    view = motifscores_view[unwrapped_idx.flatten()]

    # pool for each fragment
    fragmentmotifscores = (torch.nn.functional.avg_pool1d(view.transpose(1, 0).to(torch.float), cutwindow_width+1)).transpose(1, 0)
    
    # pool over cut sites 0 and 1
    fragmentmotifscores = fragmentmotifscores.reshape((fragmentmotifscores.shape[0]//2, 2, fragmentmotifscores.shape[-1])).sum(1)
    
    return fragmentmotifscores


# %% tags=[]
n_genes = 2
mapping = torch.tensor([[0, 0], [0, 1], [1, 0]])
coordinates = torch.tensor([[-5, 6], [-10, 2], [-2, 9]])

cut_padding_left, cut_padding_right = 1, 1
window = (-10, 10)
window_width = window[1] - window[0]
cutwindow = (-1, 1)
cutwindow_width = cutwindow[1] - cutwindow[0]

#
motifscores_view = torch.zeros((window_width * n_genes, 2))
fragmentmotifscores = pool_cut_windows(mapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
assert (fragmentmotifscores == 0.).all()

# #
motifscores_view = torch.zeros((window_width * n_genes, 2))
motifscores_view[5, 0] = 1
fragmentmotifscores = pool_cut_windows(mapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
assert (fragmentmotifscores[0, 0] == 1/3).all(), "Score at 0,0 should be 1/3"
assert (fragmentmotifscores.flatten()[1:] == 0).all(), "Total score should be 1/3"

#
motifscores_view = torch.zeros((window_width * n_genes, 2))
motifscores_view[1 + window_width, 0] = 1
fragmentmotifscores = pool_cut_windows(mapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
assert (fragmentmotifscores[1, 0] == 1/3).all(), "Score at 1,0 should be 1/3"
assert (fragmentmotifscores.flatten().sum() == 1/3).all(), "Total score should be 1/3"

# %%
n_genes = 1
mapping = torch.tensor([[0, 0]])
coordinates = torch.tensor([[-50, 60]])

cut_padding_left, cut_padding_right = 5, 5
window = (-100, 100)
window_width = window[1] - window[0]
cutwindow = (-10, 10)
cutwindow_width = cutwindow[1] - cutwindow[0]

motifscores_view = torch.zeros((window_width * n_genes, 1))
fragmentmotifscores = pool_cut_windows(mapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
assert fragmentmotifscores.shape[0] == mapping.shape[0]
assert (fragmentmotifscores == 0).all()

motifscores_view = torch.zeros((window_width * n_genes, 1))
motifscores_view[40 + 0 * window_width, 0] = 1
fragmentmotifscores = pool_cut_windows(mapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
assert (fragmentmotifscores == 1/21).all()

motifscores_view = torch.zeros((window_width * n_genes, 1))
motifscores_view[39 + 0 * window_width, 0] = 1
fragmentmotifscores = pool_cut_windows(mapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
assert (fragmentmotifscores == 0).all()

motifscores_view = torch.zeros((window_width * n_genes, 1))
motifscores_view[50 + 0 * window_width, 0] = 1
fragmentmotifscores = pool_cut_windows(mapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
assert (fragmentmotifscores == 1/21).all()

motifscores_view = torch.zeros((window_width * n_genes, 1))
motifscores_view[60 + 0 * window_width, 0] = 1
fragmentmotifscores = pool_cut_windows(mapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
assert (fragmentmotifscores == 1/21).all()

motifscores_view = torch.zeros((window_width * n_genes, 1))
motifscores_view[61 + 0 * window_width, 0] = 1
fragmentmotifscores = pool_cut_windows(mapping, coordinates, motifscores_view, window, window_width, cutwindow, cutwindow_width)
assert (fragmentmotifscores == 0.).all()


# %%
def n_motifs(mapping, coordinates, motifscores_view):
    idxs = torch.clamp(coordinates[:, 0] + torch.arange(-cut_padding_left, cut_padding_right, device = mapping.device)[:, None] + padding_negative, 0, width-1)
    unwrapped_idx = idxs + mapping[:, 1] * width

    view = motifscores_view[unwrapped_idx.flatten()]

    fragmentmotifscores = (torch.nn.functional.avg_pool1d(view.transpose(1, 0).to(torch.float), 300)).transpose(1, 0)
    
    return fragmentmotifscores

# mapping = fragments.mapping[split.fragments_selected]
# coordinates = fragments.coordinates[split.fragments_selected]

# idxs = torch.clamp(coordinates[:, 0] + torch.arange(-cut_padding_left, cut_padding_right, device = mapping.device)[:, None] + padding_negative, 0, width-1)
# unwrapped_idx = idxs + mapping[:, 1] * width

# view = motifscores_view[unwrapped_idx.flatten()]

# fragmentmotifscores = (torch.nn.functional.avg_pool1d(view.transpose(1, 0).to(torch.float), 300)).transpose(1, 0)

# split.fragmentmotifscores = fragmentmotifscores


# %%
for split in tqdm.tqdm(fold):
    mapping = fragments.mapping[split.fragments_selected]
    coordinates = fragments.coordinates[split.fragments_selected]
    
    for 
    
    idxs = torch.clamp(coordinates[:, 0] + torch.arange(-cut_padding_left, cut_padding_right, device = mapping.device)[:, None] + padding_negative, 0, width-1)
    unwrapped_idx = idxs + mapping[:, 1] * width

    view = motifscores_view[unwrapped_idx.flatten()]

    fragmentmotifscores = (torch.nn.functional.avg_pool1d(view.transpose(1, 0).to(torch.float), 300)).transpose(1, 0)

    split.fragmentmotifscores = fragmentmotifscores

# %%
idxs = []
for i in tqdm.tqdm(range(-cut_padding_left, cut_padding_right)):
    idx = torch.clamp(coordinates[:, 0] + padding_negative + i, 0, width-1)
    unwrapped_idx = idx + mapping[:, 1] * width
    idxs.append(unwrapped_idx)

# %%
scores = torch.zeros((n_fragments, motifscores.shape[0]))
for i in tqdm.tqdm(range(-cut_padding_left, cut_padding_right)):
    idx = torch.clamp(coordinates[:, 0] + padding_negative + i, 0, width-1)
    unwrapped_idx = idx + mapping[:, 1] * width
    
    scores += torch.index_select(motifscores_view, 0, unwrapped_idx
    # torch.index_select(motifscores_view[:, 0], 0, unwrapped_idx)
    # motifscores_view[:, 0][unwrapped_idx]

# %%
slice_between = lambda x:slice(max(x + padding_negative - cut_padding_left, 0), (x + padding_negative + cut_padding_right))

# for gene_ix in tqdm.tqdm(range(fragments.n_genes)):
for gene_ix in tqdm.tqdm(range(1)):
    fragments_selected = fragments.mapping[:, 1] == gene_ix

    mapping = fragments.mapping[fragments_selected]
    coordinates = fragments.coordinates[fragments_selected]

    n_fragments = mapping.shape[0]
    
    scanner = torch.zeros((n_fragments, motifscores.shape[-1]), dtype = int)
    for i, (start, end) in enumerate(zip(coordinates[:, 0].numpy(), coordinates[:, 1].numpy())):
        scanner[i, slice_between(start)] = 1
        scanner[i, slice_between(end)] = 1

    motifscores_gene = (motifscores[:, gene_ix, :] > 0).to(torch.long)
    fragment_scores_gene = torch.matmul(motifscores_gene, scanner.T).shape

# %%
for gene_ix in tqdm.tqdm(range(fragments.n_genes)):
# for gene_ix in tqdm.tqdm(range(1)):
    fragments_selected = fragments.mapping[:, 1] == gene_ix

    mapping = fragments.mapping[fragments_selected]
    coordinates = fragments.coordinates[fragments_selected]

    n_fragments = mapping.shape[0]

    sumscore = torch.zeros((motifscores.shape[0], n_fragments))
    motifscores_gene = (motifscores[:, gene_ix, :] > 0).to(torch.long)
    idx = torch.clamp(coordinates[:, 0] + padding_negative - cut_padding_left, 0, motifscores.shape[-1]-1)
    for i in range(-150, 150):
        idx = torch.minimum(idx + 1, torch.tensor(motifscores.shape[-1]-1))
        sumscore += torch.index_select(motifscores_gene, 1, idx)
    
    # fragment_scores_gene = torch.matmul(motifscores_gene, scanner.T).shape

# %%
torch.index_select(motifscores[:, gene_ix, :], 1, torch.maximum(coordinates[:, 0] + padding_negative, torch.tensor(0))).shape

# %%

# %%
mapping = fragments.mapping
coordinates = fragments.coordinates

n_fragments = mapping.shape[0]

# %%
# gather # of binding sites higher than 0
fragment_motif_scores = torch.zeros((n_fragments, motifscores.shape[0], 1))
for i, (gene_ix, start, end) in tqdm.tqdm(
    enumerate(zip(mapping[:, 1].numpy(), coordinates[:, 0].numpy(), coordinates[:, 1].numpy())),
    total = n_fragments,
    leave = False
):
    n_left = (motifscores[:, gene_ix, max(start + padding_negative - cut_padding_left, 0):(start + padding_negative + cut_padding_right )] > 0).sum(1)
    # n_right = (motifscores[:, gene_ix, max(end + padding_negative - cut_padding_left, 0):(end + padding_negative + cut_padding_right )] > 0).sum(1)
    fragment_motif_scores[i, :, 0] = n_left# + n_right

# %%
fragment_motif_scores.shape

# %%
fragment_scores = pd.Series(fragment_motif_scores[:, 0, 0].numpy(), transcriptome.var.index[fragments.mapping[split.fragments_selected, 1].numpy()])

# %%
genes_oi = fragment_scores.groupby(level = 0).mean().sort_values(ascending = False).index[:200]

# %%
fragment_scores.groupby(level = 0).mean().sort_values().plot(kind = "hist")

# %%
fragment_scores.groupby(level = 0).mean().sort_values()

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

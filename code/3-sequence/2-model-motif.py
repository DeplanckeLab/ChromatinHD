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
motifscores = pickle.load(open(folder_data_preproc / "motifscores.pkl", "rb"))

# %%
n_fragments = fragments.coordinates.shape[0]

# %%
import timeit

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
# ## Theoretical memory analysis

# %% [markdown]
# Assume for each fragment we need to store 600 loci (=motif binding sites with a particular distance, l/r and score)

# %%
n_fragmentxlocus = (n_fragments * 600)

# %% [markdown]
# For each locus, we need to store the distance, motif_ix, left/right and score. Let's assume it's all 32 bit. The index pointers for fragments are meaningless

# %%
n_gigs = (
    (n_fragmentxlocus * 32) +
    (n_fragmentxlocus * 32) +
    (n_fragmentxlocus * 32) +
    (n_fragmentxlocus * 32)
) / 8 / 1024 / 1024 / 1024
n_gigs

# %% [markdown]
# 😬

# %% [markdown]
# This means we will have to calculate everything on-the-fly

# %% [markdown]
# Can we store the motifscores on GPU? Assume we have 400 motifs to score, we have to store both motif_ix and value

# %%
n_data_per_motif = len(motifscores.data) / motifscores.shape[1]
n_data_per_motif

# %%
n_data = n_data_per_motif * 400

# %%
n_gigs = (
    (n_data * 32) +
    (n_data * 32)
) / 8 / 1024 / 1024 / 1024
n_gigs

# %% [markdown]
# That actually looks doable, although more genes or motifs would stretch the limits. Still, this means we can calculate fragment scores on-the-fly on the GPU itself. Not that this is likely, given that it might also be fast enough on CPU and could be done in parallel while the CPU is working

# %% [markdown]
# Could we store the scores of the whole genome (on GPU)?

# %%
n_motifs_per_base = len(motifscores.data) / motifscores.shape[1] / motifscores.shape[0]
n_motifs_per_base

# %%
n_motifs_total = n_motifs_per_base * 2*(10**9) * 400

# %%
n_motifs_total

# %%
n_gigs = (
    (n_motifs_total * 32) +
    (n_motifs_total * 32)
) / 8 / 1024 / 1024 / 1024
n_gigs

# %% [markdown]
# Probably not in VRAM, but yes on normal memory 😉

# %% [markdown]
# ## Extracting motifs close to one or more fragments

# %% [markdown]
# <img src="https://matteding.github.io/images/csr.gif" width="600" />

# %%
import numpy as np
import Cython
# %load_ext cython

# %%
import pyximport; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))
import peakfreeatac.loaders.motifs
if "peakfreeatac.loaders.motifs" in sys.modules:
    del sys.modules['peakfreeatac.loaders.motifs']
import peakfreeatac.loaders.motifs

# %%
cutwindow = np.array([-150, 150])

# %%
cells_oi = np.arange(0, 1000)
genes_oi = np.arange(0, 100)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()
cellxgene_oi_indptr = np.pad(np.cumsum(fragments.cellxgene_indptr[cellxgene_oi + 1] - fragments.cellxgene_indptr[cellxgene_oi]), (1, 0), "constant")

fragments_oi = torch.isin(fragments.mapping[:, 0], torch.from_numpy(cells_oi)) & torch.isin(fragments.mapping[:, 1], torch.from_numpy(genes_oi))
n_fragments = fragments_oi.sum()

coordinates = np.ascontiguousarray(fragments.coordinates[fragments_oi].to(torch.int64).numpy())
mapping = np.ascontiguousarray(fragments.mapping[fragments_oi].to(torch.int64).numpy())
genemapping = np.ascontiguousarray(fragments.mapping[fragments_oi, 1].to(torch.int64).numpy())

# %%
n_motifs = motifscores.shape[1]

# %%
motifscores_indptr = motifscores.indptr.astype(int)
motifscores_indices = motifscores.indices.astype(int)
motifscores_data = motifscores.data.astype(np.float64)

buffer_size = coordinates.shape[0] * 1000

out_fragment_indptr = torch.from_numpy(np.zeros(buffer_size, dtype = int)).numpy()
out_motif_ix = torch.from_numpy(np.zeros(buffer_size, dtype = int)).numpy()
out_score = torch.from_numpy(np.zeros(buffer_size, dtype = np.float64)).numpy()
out_distance = torch.from_numpy(np.zeros(buffer_size, dtype = int)).numpy()
out_motifcounts = torch.from_numpy(np.ascontiguousarray(np.zeros((n_fragments, n_motifs), dtype = int))).numpy()

# %%
out_n = pfa.loaders.motifs.extract_all(
    coordinates,
    genemapping,
    motifscores_indptr,
    motifscores_indices,
    motifscores_data,
    *window,
    window_width,
    *cutwindow,
    out_fragment_indptr,
    out_motif_ix,
    out_score,
    out_distance,
    out_motifcounts
)

# %% [markdown]
# ### Check out

# %%
folder_motifs = pfa.get_output() / "data" / "motifs" / "hs" / "hocomoco"
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs_oi = pickle.load((folder_data_preproc / "motifs_oi.pkl").open("rb"))

# %%
motif_scores = pd.DataFrame({
    "motif_index":out_motif_ix[:out_n],
    "locus_score":out_score[:out_n],
    "locus_distance":out_distance[:out_n],
    "fragment_ix":np.repeat(np.arange(n_fragments), np.diff(out_fragment_indptr[:n_fragments+1]))
})
motif_scores["gene_ix"] = mapping[motif_scores["fragment_ix"].values, 1]
motif_scores["cell_ix"] = mapping[motif_scores["fragment_ix"].values, 0]
motif_scores["local_gene_ix"] = pd.Series(np.arange(len(genes_oi)), index = genes_oi)[motif_scores["gene_ix"]].values
motif_scores["local_cell_ix"] = pd.Series(np.arange(len(cells_oi)), index = cells_oi)[motif_scores["cell_ix"]].values
motif_scores["locall_cellxgene_ix"] = motif_scores["local_cell_ix"] * len(genes_oi) + motif_scores["local_gene_ix"]

# %%
# select the site that scores best on a particular motif
motifs_oi["ix"] = np.arange(motifs_oi.shape[0])
motif_ix = motifs_oi.query("gene_label == 'NFKB1'")["ix"][0]
# motif_ix = 10
best = motif_scores.query("motif_index == @motif_ix").sort_values("locus_score", ascending = False).iloc[0]
ingene_mid = int(best["locus_distance"])
gene_ix = int(best["gene_ix"])

# select the site that scores best overall
# best = motif_scores.sort_values("locus_score", ascending = False).iloc[[1]].to_dict(orient='records')[0]
# ingene_mid = int(best["locus_distance"])
# motif_ix = int(best["motif_index"])
# gene_ix = int(best["gene_ix"])

# %%
pwm = pwms[motifs_oi.iloc[motif_ix].name]

# %%
import math
site_start = ingene_mid - math.floor(pwm.shape[0] / 2) - gene_ix * window_width
site_end = ingene_mid + math.ceil(pwm.shape[0] / 2) - gene_ix * window_width


# %%
def create_onehot(seq):
    """
    Sequence contains integers 0 (A), 1 (C), 2 (G), 3 (T), and 4 (N)
    """
    return torch.tensor(np.eye(5, dtype = np.float32)[seq][:, :-1])


# %%
import gzip
if "genome" not in globals():
    genome = pickle.load(gzip.GzipFile((folder_data_preproc / "genome.pkl.gz"), "rb"))

# %%
chromosome, promoter_start, promoter_end, strand = promoters.iloc[gene_ix][["chr", "start", "end", "strand"]]
strand

# %%
# +1 here because of python starting at 0 yadda yadda
seq = genome[chromosome][(promoter_start + 1):(promoter_end + 1)]
seq = seq[::strand]

# %%
onehot = create_onehot(seq[site_start:site_end])

# %% [markdown]
# ![](https://www.frontiersin.org/files/Articles/437612/fimmu-10-01176-HTML/image_m/fimmu-10-01176-g001.jpg)

# %%
fig, (ax_score, ax_onehot, ax_pwm, ax_onehotrev, ax_scorerev) = plt.subplots(5, 1, figsize = (3, 4), sharex = True)

ntscores = pwm.flatten()[onehot.flatten().to(bool)]
ax_score.fill_between(np.arange(onehot.shape[0]), ntscores, color = "#55555533")
for x, y, s, c in zip(np.arange(onehot.shape[0]), ntscores, np.array(["A", "C", "G", "T"])[onehot.argmax(1)], np.array(sns.color_palette(n_colors = 4))[onehot.argmax(1)]):
    ax_score.text(x, y, s, ha = "center", va = "center", color = "white", path_effects = [mpl.patheffects.Stroke(linewidth = 2, foreground = c), mpl.patheffects.Normal()])

pd.DataFrame(onehot.numpy()).plot(ax = ax_onehot, legend = False)

pd.DataFrame(pwm.numpy()).plot(ax = ax_pwm, legend = False)

pd.DataFrame(onehot.numpy()[::-1, [3, 2, 1, 0]]).plot(ax = ax_onehotrev, legend = False)

onehot_rev = onehot.numpy()[::-1, [3, 2, 1, 0]]
ntscores = pwm.flatten()[onehot_rev.flatten().astype(bool)]
ax_scorerev.fill_between(np.arange(onehot.shape[0]), ntscores, color = "#55555533")
for x, y, s, c in zip(np.arange(onehot.shape[0]), ntscores, np.array(["A", "C", "G", "T"])[onehot_rev.argmax(1)], np.array(sns.color_palette(n_colors = 4))[onehot_rev.argmax(1)]):
    ax_scorerev.text(x, y, s, ha = "center", va = "center", color = "white", path_effects = [mpl.patheffects.Stroke(linewidth = 2, foreground = c), mpl.patheffects.Normal()])

# %%
forward_score = (onehot.numpy() * pwm.numpy()).sum()
reverse_score = ((onehot.numpy()[::-1, [3, 2, 1, 0]] * pwm.numpy()).sum())
forward_score, reverse_score

# %%
assert np.isclose(best["locus_score"], reverse_score) or np.isclose(best["locus_score"], forward_score)

# %% [markdown]
# ## Subsetting cellxgene

# %%
import pyximport; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))
import peakfreeatac.loaders.fragments
if "peakfreeatac.loaders.fragments" in sys.modules:
    del sys.modules['peakfreeatac.loaders.fragments']
import peakfreeatac.loaders.fragments

# %%
cells_oi = np.arange(0, 10000)
genes_oi = np.arange(0, 1000)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

# %%
buffer_size = len(cellxgene_oi) * 2

out_coordinates = torch.from_numpy(np.zeros((buffer_size, 2), dtype = np.int64)).numpy()
out_genemapping = torch.from_numpy(np.zeros(buffer_size, dtype = np.int64)).numpy()
out_local_cellxgene_ix = torch.from_numpy(np.zeros(buffer_size, dtype = np.int64)).numpy()

# %%
cellxgene_indptr = fragments.cellxgene_indptr.numpy().astype(np.int64)
coordinates = fragments.coordinates.numpy().astype(np.int64)
genemapping = fragments.mapping[:, 1].numpy().astype(np.int64)

# %%
n_fragments = extract_fragments.extract_fragments(
    cellxgene_oi,
    cellxgene_indptr,
    coordinates,
    genemapping,
    out_coordinates,
    out_genemapping,
    out_local_cellxgene_ix
)

# %% [markdown]
# ## Subset both

# %%
# # !rm extract_1.cpython-39-x86_64-linux-gnu.so
import pyximport; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))

import sys
if "extract_motifs" in sys.modules:
    del sys.modules['extract_motifs']
import extract_motifs

import sys
if "extract_fragments" in sys.modules:
    del sys.modules['extract_fragments']
import extract_fragments


# %%
class FragmentMotifLoader():
    def __init__(self, fragments, motifscores, batch_size, window, cutwindow):
        self.batch_size = batch_size
        
        # store auxilliary information
        self.window = window
        self.cutwindow = cutwindow
        self.window_width = window[1] - window[0]
        self.cutwindow_width = cutwindow[1] - cutwindow[0]
        
        # store fragment data
        self.cellxgene_indptr = fragments.cellxgene_indptr.numpy().astype(np.int64)
        self.coordinates = fragments.coordinates.numpy().astype(np.int64)
        self.genemapping = fragments.mapping[:, 1].numpy().astype(np.int64)
        
        # create buffers for coordinates
        n_fragment_per_cellxgene = 2
        fragment_buffer_size = batch_size * n_fragment_per_cellxgene

        self.out_coordinates = torch.from_numpy(np.zeros((fragment_buffer_size, 2), dtype = np.int64))#.pin_memory()
        self.out_genemapping = torch.from_numpy(np.zeros(fragment_buffer_size, dtype = np.int64))#.pin_memory()
        self.out_local_cellxgene_ix = torch.from_numpy(np.zeros(fragment_buffer_size, dtype = np.int64))#.pin_memory()
        
        # create buffers for motifs
        n_motifs = motifscores.shape[1]
        n_motifs_per_fragment = 1000 # 400 motifs
        motif_buffer_size = fragment_buffer_size * n_motifs_per_fragment
        
        self.motifscores_indptr = motifscores.indptr.astype(np.int64)
        self.motifscores_indices = motifscores.indices.astype(np.int64)
        self.motifscores_data = motifscores.data.astype(np.float64)

        self.out_fragment_indptr = torch.from_numpy(np.zeros(motif_buffer_size, dtype = int))#.pin_memory()
        self.out_motif_ix = torch.from_numpy(np.zeros(motif_buffer_size, dtype = int))#.pin_memory()
        self.out_score = torch.from_numpy(np.zeros(motif_buffer_size, dtype = np.float64))#.pin_memory()
        self.out_distance = torch.from_numpy(np.zeros(motif_buffer_size, dtype = int))#.pin_memory()
        self.out_motifcounts = torch.from_numpy(np.ascontiguousarray(np.zeros((fragment_buffer_size, n_motifs), dtype = int)))#.pin_memory()
        
    def load(self, cellxgenes_oi):
        assert len(cellxgenes_oi) <= self.batch_size
        n_fragments = extract_fragments.extract_fragments(
            cellxgene_oi,
            self.cellxgene_indptr,
            self.coordinates,
            self.genemapping,
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.out_local_cellxgene_ix.numpy()
        )
        self.out_coordinates.resize_((n_fragments, 2))
        self.out_genemapping.resize_((n_fragments))
        self.out_local_cellxgene_ix.resize_((n_fragments))
        
        self.out_motifcounts.data.zero_() # this is a big slow down (~20% of function) but unsure how to fix honestly
        
        n_motifs = extract_motifs.extract_motifcounts(
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.motifscores_indptr,
            self.motifscores_indices,
            self.motifscores_data,
            *self.window,
            self.window_width,
            *self.cutwindow,
            # self.out_fragment_indptr.numpy(),
            # self.out_motif_ix.numpy(),
            # self.out_score.numpy(),
            # self.out_distance.numpy(),
            self.out_motifcounts.numpy()
        )
        # self.out_fragment_indptr.resize_(n_fragments+1)
        # self.out_motif_ix.resize_(n_motifs)
        # self.out_score.resize_(n_motifs)
        # self.out_distance.resize_(n_motifs)
        self.out_motifcounts.resize_((n_fragments, self.out_motifcounts.shape[1]))
        
        return self.out_motifcounts, self.out_local_cellxgene_ix


# %%
n_cells = 100
n_genes = 100
cutwindow = np.array([-150, 150])
loader = FragmentMotifLoader(fragments, motifscores, n_cells * n_genes, window, cutwindow)

# %%
cells_oi = np.arange(0, n_cells)
genes_oi = np.arange(0, n_genes)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

motifcounts, local_cellxgene_ix = loader.load(cellxgene_oi)

# %%
# %%timeit -n 1
motifcounts, local_cellxgene_ix = loader.load(cellxgene_oi)

# %%
(fragments.n_cells * fragments.n_genes) / len(cellxgene_oi)

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
# ### Create fragment embedder

# %%
fragment_embedding = motifcounts.to(torch.float) # 😅

# %%
fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharey = True)
# sns.heatmap(coordinates.numpy(), ax = axes[0])
sns.heatmap(fragment_embedding.detach().numpy()[:100], ax = axes[1])
axes[0].set_ylabel("Fragment")

# %% [markdown]
# ### Pool fragments

# %%
from peakfreeatac.models.promotermotif.v1 import EmbeddingGenePooler

# %%
n_embedding_dimensions = motifscores.shape[1]

# %%
embedding_gene_pooler = EmbeddingGenePooler(n_embedding_dimensions, debug = True)

# %%
cell_gene_embedding = embedding_gene_pooler(fragment_embedding, local_cellxgene_ix, n_cells, n_genes)

# %%
fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharey = True)
# sns.heatmap(coordinates.numpy(), ax = axes[0])
sns.heatmap(cell_gene_embedding[0].detach().numpy()[:100], ax = axes[1])
axes[0].set_ylabel("Fragment")

# %% [markdown]
# ### Create expression predictor

# %%
from peakfreeatac.models.promotermotif.v1 import EmbeddingToExpression

# %%
embedding_to_expression = EmbeddingToExpression(n_embedding_dimensions = n_embedding_dimensions, mean_gene_expression = mean_gene_expression)
# embedding_to_expression = EmbeddingToExpressionBias(fragments.n_genes, n_embedding_dimensions = n_embedding_dimensions, mean_gene_expression = mean_gene_expression)

# %%
expression_predicted = embedding_to_expression(cell_gene_embedding, torch.from_numpy(genes_oi))

# %% [markdown]
# ### Whole model

# %%
from peakfreeatac.models.promotermotif.v1 import FragmentEmbeddingToExpression

# %%
model = FragmentEmbeddingToExpression(fragments.n_genes, mean_gene_expression, n_embedding_dimensions)

# %%
model(motifcounts.to(torch.float), local_cellxgene_ix, n_cells, n_genes, genes_oi)

# %% [markdown]
# ## Infer

# %%
model = FragmentEmbeddingToExpression(fragments.n_genes, mean_gene_expression, n_embedding_dimensions)

# %%
n_epochs = 1000

# %%
import itertools

# %%
device = "cpu"

# %%
transcriptome_X = transcriptome.X.to(device)
transcriptome_X_dense = transcriptome_X.dense()

# %%
params = model.get_parameters()

# lr = 1.0
lr = 1e-4
optim = torch.optim.SGD(
    params,
    lr = lr,
    momentum=0.3
)
loss = torch.nn.MSELoss(reduction = "mean")

# %%
trace = []

prev_mse_train = None
prev_mse_test = None

# %%
cells_all = np.arange(fragments.n_cells)
genes_all = np.arange(fragments.n_genes)

n_cells_step = 1000
n_genes_step = 100

cells_train = cells_all[:int(len(cells_all) * 4 / 5)]
genes_train = genes_all[:int(len(genes_all) * 4 / 5)]

cells_validation = cells_all[[cell for cell in cells_all if cell not in cells_train]]
genes_validation = genes_train
# genes_validation = genes_all[[gene for gene in genes_all if gene not in genes_train]]

# %%
def cell_gene_to_cellxgene(cells_oi, genes_oi, n_genes):
    return (cells_oi[:, None] * n_genes + genes_oi).flatten()


# %%
def create_bins(cells, genes, rg = None):
    if rg is None:
        rg = np.random.RandomState()
    cells = rg.permutation(cells)
    genes = rg.permutation(genes)

    gene_cuts = [*np.arange(0, len(genes), step = n_genes_step)] + [len(genes)]
    gene_bins = [genes[a:b] for a, b in zip(gene_cuts[:-1], gene_cuts[1:])]

    cell_cuts = [*np.arange(0, len(cells), step = n_cells_step)] + [len(cells)]
    cell_bins = [cells[a:b] for a, b in zip(cell_cuts[:-1], cell_cuts[1:])]

    cellxgene_bins = [cell_gene_to_cellxgene(cells_oi, genes_oi, fragments.n_genes) for cells_oi, genes_oi in itertools.product(cell_bins, gene_bins)]

    bins = []
    for cells_oi, genes_oi in itertools.product(cell_bins, gene_bins):
        bins.append([
            cells_oi,
            genes_oi,
            cell_gene_to_cellxgene(cells_oi, genes_oi, fragments.n_genes),
            len(cells_oi),
            len(genes_oi)
        ])
    return bins


# %%
rg = np.random.RandomState(2)
bins_train = create_bins(cells_train, genes_train, rg = rg)
bins_validation = create_bins(cells_validation, genes_validation, rg = rg)
bins_validation_trace = bins_validation[:2]

# %%
loader = FragmentMotifLoader(fragments, motifscores, n_cells_step * n_genes_step, window, cutwindow)

# %%
step_ix = 0
trace_every_step = 10
optimize_every_step = 1
trace_validation = []

for epoch in tqdm.tqdm(range(n_epochs)):
    # train
    for cells_oi, genes_oi, cellxgene_oi, n_cells, n_genes in tqdm.tqdm(bins_train, leave = False):
        if (step_ix % trace_every_step) == 0:
            with torch.no_grad():
                print("tracing")
                mse_validation = []
                mse_validation_dummy = []
                for cells_oi, genes_oi, cellxgene_oi, n_cells, n_genes in bins_validation_trace:
                    motifcounts, local_cellxgene_ix = loader.load(cellxgene_oi)
                    motifcounts = motifcounts.to(device)
                    local_cellxgene_ix = local_cellxgene_ix.to(device)

                    transcriptome_subset = transcriptome_X_dense[cells_oi, :][:, genes_oi]

                    transcriptome_predicted = model(motifcounts.to(torch.float), local_cellxgene_ix, n_cells, n_genes, genes_oi)

                    mse = loss(transcriptome_predicted, transcriptome_subset)

                    mse_validation.append(mse.item())
                    mse_validation_dummy.append(((transcriptome_subset - transcriptome_subset.mean(0, keepdim = True)) ** 2).mean())
                mse_validation = np.mean(mse_validation)
                mse_validation_dummy = np.mean(mse_validation_dummy)
                
                print(mse_validation - mse_validation_dummy)
                
                trace_validation.append({
                    "epoch":epoch,
                    "step":step_ix,
                    "mse":mse_validation,
                    "mse_dummy":mse_validation_dummy
                })
    
        torch.set_grad_enabled(True)
        motifcounts, local_cellxgene_ix = loader.load(cellxgene_oi)
        motifcounts = motifcounts.to(device)
        local_cellxgene_ix = local_cellxgene_ix.to(device)
        
        transcriptome_subset = transcriptome_X_dense[cells_oi, :][:, genes_oi]
        
        transcriptome_predicted = model(motifcounts.to(torch.float), local_cellxgene_ix, n_cells, n_genes, genes_oi)
        
        mse = loss(transcriptome_predicted, transcriptome_subset)

        mse.backward()
        
        # if (step_ix % trace_every_step) == 0:
        optim.step()
        optim.zero_grad()
        
        step_ix += 1

    # reshuffle the order of the bins
    bins_train = [bins_train[i] for i in np.random.choice(len(bins_train), len(bins_train), replace = False)]

# %%
bins_train = [bins_train[i] for i in np.random.choice(len(bins_train), len(bins_train), replace = False)]

# %%
if isinstance(trace_validation, list):
    trace_validation = pd.DataFrame(list(trace_validation))

# %%
fig, ax = plt.subplots()
plotdata = trace_validation.groupby("step").mean().reset_index()
ax.plot(plotdata["step"], plotdata["mse"], zorder = 6, color = "orange")
fig.savefig("trace.png")

# %%
folder_motifs = pfa.get_output() / "data" / "motifs" / "hs" / "hocomoco"
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs_oi = pickle.load((folder_data_preproc / "motifs_oi.pkl").open("rb"))

# %%
pd.Series(model.embedding_to_expression.weight1.detach().cpu().numpy(), index = motifs_oi.index).sort_values()

# %%
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["IRF1", "STAT2", "PCNA", "KLF4", "KLF5", "SPI1", "BCL11A"]))

# %%
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["STAT2"]))

# %%

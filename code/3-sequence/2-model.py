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
motifscan_folder = pfa.get_output() / "motifscans" / dataset_name / promoter_name
motifscores = pickle.load(open(motifscan_folder / "motifscores.pkl", "rb"))

# %%
n_fragments = fragments.coordinates.shape[0]

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
import pyximport; import sys; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))
if "peakfreeatac.loaders.extraction.motifs" in sys.modules:
    del sys.modules['peakfreeatac.loaders.extraction.motifs']
import peakfreeatac.loaders.extraction.motifs

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
out_n = pfa.loaders.extraction.motifs.extract_all(
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
motif_ix = motifs_oi.query("gene_label == 'FOXQ1'")["ix"][0]
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
nucleotides = pd.DataFrame({"nucleotide":np.arange(4), "label":["A", "C", "G", "T"]})
nucleotides["color"] = sns.color_palette(n_colors = 4)

# %%
fig, (ax_score, ax_onehot, ax_pwm, ax_onehotrev, ax_scorerev) = plt.subplots(5, 1, figsize = (3, 4), sharex = True)

ntscores = pwm.flatten()[onehot.flatten().to(bool)]
ax_score.fill_between(np.arange(onehot.shape[0]), ntscores, color = "#55555533")
for (nucleotide, x), y in pd.DataFrame(pwm).unstack().items():
    s = nucleotides.loc[nucleotide, "label"]
    c = nucleotides.loc[nucleotide, "color"]
    ax_score.text(x, y, s, ha = "center", va = "center", color = c, alpha = 0.3)
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
for (nucleotide, x), y in pd.DataFrame(pwm.numpy()[::-1, [3, 2, 1, 0]]).unstack().items():
    s = nucleotides.loc[nucleotide, "label"]
    c = nucleotides.loc[nucleotide, "color"]
    ax_scorerev.text(x, y, s, ha = "center", va = "center", color = c, alpha = 0.3)

# %%
forward_score = (onehot.numpy() * pwm.numpy()).sum()
reverse_score = ((onehot.numpy()[::-1, [3, 2, 1, 0]] * pwm.numpy()).sum())
forward_score, reverse_score

# %%
assert np.isclose(best["locus_score"], reverse_score) or np.isclose(best["locus_score"], forward_score)

# %% [markdown]
# ## Extracting fragments

# %%
import sys; import pyximport; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))
if "peakfreeatac.loaders.extraction.fragments" in sys.modules:
    del sys.modules['peakfreeatac.loaders.extraction.fragments']
import peakfreeatac.loaders.extraction.fragments

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
n_fragments = peakfreeatac.loaders.extraction.fragments.extract_fragments(
    cellxgene_oi,
    cellxgene_indptr,
    coordinates,
    genemapping,
    out_coordinates,
    out_genemapping,
    out_local_cellxgene_ix
)

# %% [markdown]
# ## Load fragments & motifs

# %%
import pyximport; import sys; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))
if "peakfreeatac.loaders.extraction.fragments" in sys.modules:
    del sys.modules['peakfreeatac.loaders.extraction.fragments']
import peakfreeatac.loaders.extraction.fragments
if "peakfreeatac.loaders.extraction.motifs" in sys.modules:
    del sys.modules['peakfreeatac.loaders.extraction.motifs']
import peakfreeatac.loaders.extraction.motifs

# %%
import peakfreeatac.loaders.fragments
import peakfreeatac.loaders.fragmentmotif

# %% [markdown]
# ### Full

# %%
n_cells = 1000
n_genes = 100
cutwindow = np.array([-150, 150])
loader = peakfreeatac.loaders.fragmentmotif.Full(fragments, motifscores, n_cells * n_genes, window, cutwindow)

# %%
cells_oi = np.arange(0, n_cells)
genes_oi = np.arange(0, n_genes)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

full_result = loader.load(cellxgene_oi, cells_oi = cells_oi, genes_oi = genes_oi)

# %%
# %%timeit -n 1
full_result = loader.load(cellxgene_oi, cells_oi = cells_oi, genes_oi = genes_oi)

# %%
(fragments.n_cells * fragments.n_genes) / len(cellxgene_oi)

# %% [markdown]
# ### Motifcounts

# %%
n_cells = 1000
n_genes = 100
cutwindow = np.array([-150, 150])
loader = peakfreeatac.loaders.fragmentmotif.Motifcounts(fragments, motifscores, n_cells * n_genes, window, cutwindow)

# %%
cells_oi = np.random.choice(fragments.n_cells, n_cells)
genes_oi = np.random.choice(fragments.n_genes, n_genes)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

minibatch = peakfreeatac.loaders.minibatching.Minibatch(cellxgene_oi = cellxgene_oi, cells_oi = cells_oi, genes_oi = genes_oi)
motifcounts_result = loader.load(minibatch)

# %%
# check subsetting the fragments
motifcounts_result = loader.load(minibatch, fragments_oi = torch.arange(10000))

# %%
# %%timeit -n 1
motifcounts_result = loader.load(minibatch)

# %% [markdown]
# ## Loading using multithreading

# %%
import pyximport; import sys; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))
if "peakfreeatac.loaders.extraction.fragments" in sys.modules:
    del sys.modules['peakfreeatac.loaders.extraction.fragments']
import peakfreeatac.loaders.extraction.fragments
if "peakfreeatac.loaders.extraction.motifs" in sys.modules:
    del sys.modules['peakfreeatac.loaders.extraction.motifs']
import peakfreeatac.loaders.extraction.motifs

# %%
# n_cells = 2000
n_cells = 3
n_genes = 100

cutwindow = np.array([-150, 150])

# %%
import peakfreeatac.loaders.fragmentmotif

# %%
loaders = peakfreeatac.loaders.pool.LoaderPool(
    peakfreeatac.loaders.fragmentmotif.Motifcounts,
    {"fragments":fragments, "motifscores":motifscores, "cellxgene_batch_size":n_cells * n_genes, "window":window, "cutwindow":cutwindow},
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
fragment_embedding = data.motifcounts

# %%
fig, ax0 = plt.subplots(1, 1, figsize = (5, 5), sharey = True)
sns.heatmap(fragment_embedding.detach().numpy()[:100], ax = ax0)
ax0.set_ylabel("Fragment")
ax0.set_xlabel("Component")

# %% [markdown]
# ### Encode fragments

# %%
from peakfreeatac.models.promotermotif.v3 import FragmentSineEncoder

# %%
fragment_encoder = FragmentSineEncoder()

# %%
fragment_encoding = fragment_encoder(data.coordinates)

# %%
fig, ax0 = plt.subplots(1, 1, figsize = (5, 5), sharey = True)
sns.heatmap(fragment_encoding.detach().numpy()[:100], ax = ax0)
ax0.set_ylabel("cellxgene")
ax0.set_xlabel("component")

# %% [markdown]
# ### Weight

# %%
from peakfreeatac.models.promotermotif.v3 import FragmentWeighter

# %%
fragment_weighter = FragmentWeighter()

# %%
fragment_weight = fragment_weighter(data.coordinates)

# %%
plt.scatter(data.coordinates[:, 0], fragment_weight.detach().numpy())

# %% [markdown]
# ### Pool fragments

# %%
from peakfreeatac.models.promotermotif.v3 import EmbeddingGenePooler

# %%
n_features = motifscores.shape[1]

# %%
embedding_gene_pooler = EmbeddingGenePooler(n_features, debug = True)

# %%
cell_gene_embedding = embedding_gene_pooler(
    fragment_embedding,
    cellxgene_ix = data.local_cellxgene_ix,
    weights = fragment_weight,
    n_cells = data.n_cells,
    n_genes = data.n_genes
)

# %%
fig, ax0 = plt.subplots(1, 1, figsize = (5, 5), sharey = True)
sns.heatmap(cell_gene_embedding[0].detach().numpy()[:100], ax = ax0)
ax0.set_ylabel("cellxgene")
ax0.set_xlabel("component")

# %% [markdown]
# ### Create expression predictor

# %%
from peakfreeatac.models.promotermotif.v3 import EmbeddingToExpression

# %%
embedding_to_expression = EmbeddingToExpression(n_components = n_features, mean_gene_expression = mean_gene_expression)
# embedding_to_expression = EmbeddingToExpressionBias(fragments.n_genes, n_components = n_components, mean_gene_expression = mean_gene_expression)

# %%
expression_predicted = embedding_to_expression(cell_gene_embedding, data.genes_oi)

# %% [markdown]
# ### Whole model

# %%
from peakfreeatac.models.promotermotif.v3 import FragmentEmbeddingToExpression

# %%
model = FragmentEmbeddingToExpression(fragments.n_genes, mean_gene_expression, n_features)

# %%
model(data)

# %% [markdown]
# ## Infer

# %%
from peakfreeatac.models.promotermotif.v2 import FragmentEmbeddingToExpression
import peakfreeatac.loaders.fragmentmotif

# %%
mean_gene_expression = transcriptome.X.dense().mean(0)

# %%
n_components = motifscores.shape[1]
cutwindow = np.array([-150, 150])

# %%
model = FragmentEmbeddingToExpression(fragments.n_genes, mean_gene_expression, n_components, weighting = True)

# %%
n_epochs = 30
device = "cuda"

# %%
import itertools

# %%
transcriptome_X = transcriptome.X.to(device)
transcriptome_X_dense = transcriptome_X.dense()

# %%
params = model.parameters()

# lr = 1.0
checkpoint_every_step = 30
optimize_every_step = 10
lr = 1e-3 / optimize_every_step
# lr = 1e-4 / optimize_every_step
optim = torch.optim.Adam(
    params,
    lr = lr
)
loss = torch.nn.MSELoss(reduction = "mean")
cos = torch.nn.CosineSimilarity(dim = 0)
loss = lambda x_1, x_2: -cos(x_1, x_2).mean()
# def correlation_loss(x, y, dim = 0):
#     cor = (x * y).sum(dim) / ((y**2).sum(dim) * (x**2).sum(dim))
#     return -cor.mean()
# def correlation_loss(x, y, dim = 0):
#     cor = pfa.utils.paircor(x, y)
#     return -cor.mean()
# loss = correlation_loss
# loss = torch.nn.PoissonNLLLoss(reduction = "mean", full = True)

# %%
# train/test split
cells_all = np.arange(fragments.n_cells)
genes_all = np.arange(fragments.n_genes)

n_cell_bins = 5
n_gene_bins = 5

chromosome_gene_counts = transcriptome.var.groupby("chr").size().sort_values(ascending = False)
chromosome_bins = np.cumsum(((np.cumsum(chromosome_gene_counts) % (chromosome_gene_counts.sum() / n_gene_bins + 1)).diff() < 0))

cells_train = cells_all[: int(len(cells_all) * (n_cell_bins-1) / n_gene_bins)]
cells_validation = cells_all[[cell not in cells_train for cell in cells_all]]

chromosomes_train = chromosome_bins.index[chromosome_bins < (n_gene_bins - 1)]
chromosomes_validation = chromosome_bins.index[chromosome_bins >= (n_gene_bins - 1)]
genes_train = fragments.var["ix"][transcriptome.var.index[transcriptome.var["chr"].isin(chromosomes_train)]].values
genes_validation = fragments.var["ix"][transcriptome.var.index[transcriptome.var["chr"].isin(chromosomes_validation)]].values

# %%
n_cells_step = 1000
n_genes_step = 300

rg = np.random.RandomState(0)
minibatches_train = pfa.loaders.minibatching.create_bins_random(
    cells_train,
    genes_train,
    n_cells_step=n_cells_step,
    n_genes_step=n_genes_step,
    n_genes_total=fragments.n_genes,
    rg=rg,
)
minibatches_validation = pfa.loaders.minibatching.create_bins_ordered(
    # cells_train,
    cells_validation,
    # genes_train,
    genes_validation,
    n_cells_step=n_cells_step,
    n_genes_step=n_genes_step,
    n_genes_total=fragments.n_genes,
    rg=rg,
)
minibatches_validation_trace = minibatches_validation[:5]

# %%
model = model.to(device)

# %%
import peakfreeatac.loaders

# %%
loaders = pfa.loaders.LoaderPool(
    peakfreeatac.loaders.fragmentmotif.Motifcounts,
    {"fragments":fragments, "motifscores":motifscores, "cellxgene_batch_size":n_cells_step * n_genes_step, "window":window, "cutwindow":cutwindow},
    n_workers = 20
)
loaders.initialize(minibatches_train)
loaders_validation = pfa.loaders.LoaderPool(
    peakfreeatac.loaders.fragmentmotif.Motifcounts,
    {"fragments":fragments, "motifscores":motifscores, "cellxgene_batch_size":n_cells_step * n_genes_step, "window":window, "cutwindow":cutwindow},
    n_workers = 5
)
loaders_validation.initialize(minibatches_validation_trace)

# %%
loaders.restart()
loaders_validation.restart()

# %%
import peakfreeatac.train

# %%
trainer = pfa.train.Trainer(
    model,
    loaders,
    loaders_validation,
    transcriptome_X_dense,
    loss,
    optim
)

# %%
trainer.train()

# %%
pd.DataFrame(trainer.trace.validation_steps).groupby("checkpoint").mean()["loss"].plot()
pd.DataFrame(trainer.trace.train_steps).groupby("checkpoint").mean()["loss"].plot()

# %%
pd.DataFrame(trace.validation_steps).groupby("checkpoint").mean()["loss"].plot()
pd.DataFrame(trace.train_steps).groupby("checkpoint").mean()["loss"].plot()

# %%
if isinstance(trace_validation, list):
    trace_validation = pd.DataFrame(list(trace_validation))

# %%
fig, ax = plt.subplots()
plotdata = trace_validation.groupby("step").mean().reset_index()
ax.plot(plotdata["step"], plotdata["mse"], zorder = 6, color = "orange")
# ax.plot(plotdata["step"], plotdata["mse_dummy"], zorder = 6, color = "red")
# ax.plot(plotdata["step"], plotdata["mean_cor"], zorder = 6, color = "red")
fig.savefig("trace.png")

# %%
model = pickle.load(open("../../model.pkl", "rb"))

# %%
folder_motifs = pfa.get_output() / "data" / "motifs" / "hs" / "hocomoco"
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs_oi = pickle.load((folder_data_preproc / "motifs_oi.pkl").open("rb"))

# %%
motif_linear_scores = pd.Series(model.embedding_to_expression.linear_weight[0].detach().cpu().numpy(), index = motifs_oi.index).sort_values(ascending = False)

# %%
motif_linear_scores.loc[motif_linear_scores.index.str.startswith("EGR")]

# %%
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["DUX4", "IRF1", "STAT2", "PCNA", "KLF4", "KLF5", "SPI1", "BCL11A", "EGR1", "SPIB"]))

# %%
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["STAT2"]))

# %%

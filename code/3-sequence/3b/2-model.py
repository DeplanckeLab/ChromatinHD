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
import peakfreeatac.data

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
# dataset_name = "pbmc10k"
# dataset_name = "pbmc10k_clustered"
dataset_name = "lymphoma+pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# %%
# promoter_name, window = "4k2k", (2000, 4000)
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
window_width = window[1] - window[0]

# %%
transcriptome = pfa.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = pfa.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
motifscan_folder = pfa.get_output() / "motifscans" / dataset_name / promoter_name / "cutoff_001"
motifscan = pfa.data.Motifscan(motifscan_folder)

# %%
n_fragments = fragments.coordinates.shape[0]

# %%
folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
fold = folds[0]

# %%
from design import get_folds_inference
folds = get_folds_inference(fragments, folds)

# %%
transcriptome.adata[folds[0]["cells_validation"]].X.sum()

# %%
transcriptome.adata[folds[0]["cells_validation"]].X.sum()

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
# Can we store the motifscan on GPU? Assume we have 400 motifs to score, we have to store both motif_ix and value

# %%
n_data_per_motif = len(motifscan.data) / motifscan.shape[1]
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
n_motifs_per_base = len(motifscan.data) / motifscan.shape[1] / motifscan.shape[0]
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
n_motifs = motifscan.shape[1]

# %%
motifscan_indptr = motifscan.indptr.astype(int)
motifscan_indices = motifscan.indices.astype(int)
motifscan_data = motifscan.data.astype(np.float64)

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
    motifscan_indptr,
    motifscan_indices,
    motifscan_data,
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
data = peakfreeatac.loaders.extraction.fragments.extract_fragments(
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
# ### Fragments

# %%
n_cells = 300
n_genes = 1000
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
# ### Full

# %%
n_cells = 100
n_genes = 10
cutwindow = np.array([-150, 150])
loader = peakfreeatac.loaders.fragmentmotif.Full(fragments, motifscan, n_cells * n_genes, window, cutwindow)

# %%
cells_oi = np.arange(0, n_cells)
genes_oi = np.arange(0, n_genes)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

minibatch = peakfreeatac.loaders.minibatching.Minibatch(cellxgene_oi = cellxgene_oi, cells_oi = cells_oi, genes_oi = genes_oi)
data = loader.load(minibatch)

# %%
# %%timeit -n 1
full_result = loader.load(cellxgene_oi, cells_oi = cells_oi, genes_oi = genes_oi)

# %%
(fragments.n_cells * fragments.n_genes) / len(cellxgene_oi)

# %% [markdown]
# ### Motifcounts

# %%
n_cells = 300
n_genes = 1000
cutwindow = np.array([-150, 150])
loader = peakfreeatac.loaders.fragmentmotif.Motifcounts(fragments, motifscan, n_cells * n_genes, window, cutwindow)
# loader = peakfreeatac.loaders.fragmentmotif.MotifcountsSplit(fragments, motifscan, n_cells * n_genes, window, cutwindow)

# %%
rg = np.random.RandomState(1)
cells_oi = rg.choice(fragments.n_cells, n_cells)
genes_oi = rg.choice(fragments.n_genes, n_genes)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

minibatch = peakfreeatac.loaders.minibatching.Minibatch(cellxgene_oi = cellxgene_oi, cells_oi = cells_oi, genes_oi = genes_oi)
data = loader.load(minibatch)

# %%
# check subsetting the fragments
data = loader.load(minibatch, fragments_oi = torch.arange(10000))

# %%
# %%timeit -n 1
data = loader.load(minibatch)

# %%
gene_oi = genes_oi[0]
cell_oi = cells_oi[0]

# %% [markdown]
# ### Motifcounts multiple

# %%
rg = np.random.RandomState(1)
cells_oi = rg.choice(fragments.n_cells, n_cells)
genes_oi = rg.choice(fragments.n_genes, n_genes)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

minibatch = peakfreeatac.loaders.minibatching.Minibatch(cellxgene_oi = cellxgene_oi, cells_oi = cells_oi, genes_oi = genes_oi)

# %%
n_cells = 300
n_genes = 1000
cutwindows = np.array([-150, 0])
loader = peakfreeatac.loaders.fragmentmotif.MotifcountsMultiple(fragments, motifscan, n_cells * n_genes, window, cutwindows)

# %%
data = loader.load(minibatch)

# %%
data.motifcounts.sum()

# %%
# %%timeit -n 1 -r 3
data = loader.load(minibatch)

# %%
n_cells = 300
n_genes = 1000
cutwindows = np.array([-150, 0])
loader = peakfreeatac.loaders.fragmentmotif.Motifcounts(fragments, motifscan, n_cells * n_genes, window, cutwindows)

# %%
data = loader.load(minibatch)

# %%
data.motifcounts.sum()

# %%
# %%timeit -n 1 -r 3
data = loader.load(minibatch)

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
    {"fragments":fragments, "motifscan":motifscan, "cellxgene_batch_size":n_cells * n_genes, "window":window, "cutwindow":cutwindow},
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
n_features = motifscan.shape[1]

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

# %% [markdown]
# ### Loaders

# %%
import peakfreeatac.loaders
import peakfreeatac.loaders.fragments
import peakfreeatac.loaders.fragmentmotif

# %%
from design import get_design, get_folds_training

# %%
design = get_design(dataset_name, transcriptome, motifscan, fragments, window)

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
    shuffle_on_iter = True,
    n_workers = 20
)
print("haha!")
loaders_validation = pfa.loaders.LoaderPool(
    design_row["loader_cls"],
    design_row["loader_parameters"],
    shuffle_on_iter = False,
    n_workers = 5
)

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

# motifs["total"] = np.array((motifscan > 0).sum(0))[0]
# motifs["n"] = np.array((data.motifcounts > 0).sum(0))
# motifs["score"] = np.log(motifs["n"] / motifs["total"])
# motifs["score"] .sort_values().plot(kind = "hist")

# %%
n_epochs = 150
checkpoint_every_epoch = 100

# n_epochs = 10
# checkpoint_every_epoch = 30

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
    checkpoint_every_epoch = checkpoint_every_epoch,
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
motifs["n"] = np.array((motifscan > 0).sum(0))[0]

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

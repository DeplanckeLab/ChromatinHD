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
import peakfreeatac.peakcounts
import peakfreeatac.data

# %%
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "alzheimer"
# dataset_name = "brain"

# dataset_name = "FLI1_7"
# dataset_name = "PAX2_7"
# dataset_name = "NHLH1_7"
# dataset_name = "CDX2_7"
# dataset_name = "CDX1_7"
# dataset_name = "MSGN1_7"
# dataset_name = "KLF4_7"
# dataset_name = "KLF5_7"
# dataset_name = "PTF1A_4"

# peaks_name = "cellranger"
# peaks_name = "genrich"
# peaks_name = "macs2"
# peaks_name = "stack"
peaks_name = "rolling_500"; window_size = 500
# peaks_name = "rolling_50"; window_size = 50

# %%
folder_data_preproc = pfa.get_output() / "data" / dataset_name
folder_root = pfa.get_output()

# %%
promoter_name, window = "10k10k", (-10000, 10000)
# promoter_name, window = "20kpromoter", (10000, 0)

promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
transcriptome = pfa.data.Transcriptome(folder_data_preproc / "transcriptome")
fragments = peakfreeatac.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
peakcounts = pfa.peakcounts.FullPeak(folder = pfa.get_output() / "peakcounts" / dataset_name / peaks_name)

# %% [markdown]
# ## Merge peaks with promoters

# %%
if peaks_name == "stack":
    peaks = promoters.reset_index()[["chr", "start", "end", "gene"]]
elif peaks_name.startswith("rolling"):
    peaks = []
    for gene, promoter in promoters.iterrows():
        starts = np.arange(promoter["start"], promoter["end"], step = window_size)
        ends = np.hstack([starts[1:], [promoter["end"]]])
        peaks.append(pd.DataFrame({"chrom":promoter["chr"], "start":starts, "ends":ends, "gene":gene}))
    peaks = pd.concat(peaks)
    
    peaks_folder = folder_root / "peaks" / dataset_name / peaks_name
    peaks_folder.mkdir(exist_ok = True, parents = True)
    peaks.to_csv(peaks_folder / "peaks.bed", index = False, header = False, sep = "\t")
else:
    peaks_folder = folder_root / "peaks" / dataset_name / peaks_name
    peaks = pd.read_table(peaks_folder / "peaks.bed", names = ["chrom", "start", "end"], usecols = [0, 1, 2])

    if peaks_name == "genrich":
        peaks["start"] += 1

# %%
import pybedtools
promoters_bed = pybedtools.BedTool.from_dataframe(promoters.reset_index()[["chr", "start", "end", "gene"]])
peaks_bed = pybedtools.BedTool.from_dataframe(peaks)

# %%
if peaks_name != "stack":
    intersect = promoters_bed.intersect(peaks_bed)
    intersect = intersect.to_dataframe()

    # peaks = intersect[["score", "strand", "thickStart", "name"]]
    peaks = intersect
peaks.columns = ["chrom", "start", "end", "gene"]
peaks = peaks.loc[peaks["start"] != -1]
peaks.index = pd.Index(peaks.chrom + ":" + peaks.start.astype(str) + "-" + peaks.end.astype(str), name = "peak")

# %%
peaks["relative_begin"] = (peaks["start"] - promoters.loc[peaks["gene"], "start"].values + window[0])
peaks["relative_stop"] = (peaks["end"] - promoters.loc[peaks["gene"], "start"].values + window[0])

# %%
peaks["relative_start"] = np.where(promoters.loc[peaks["gene"], "strand"] == 1, peaks["relative_begin"], -peaks["relative_stop"])
peaks["relative_end"] = np.where(promoters.loc[peaks["gene"], "strand"] == -1, -peaks["relative_begin"], peaks["relative_stop"])

# %%
peaks["gene_ix"] = fragments.var["ix"][peaks["gene"]].values

# %%
peaks["peak"] = peaks.index

# %%
peaks.index = peaks.peak + "_" + peaks.gene
peaks.index.name = "peak_gene"

# %%
peakcounts.peaks = peaks

# %% [markdown]
# ## Count

# %%
transcriptome.adata.obs["ix"] = np.arange(transcriptome.adata.obs.shape[0])

# %%
peakcounts.count_peaks(folder_data_preproc / "atac_fragments.tsv.gz", transcriptome.adata.obs["ix"].astype(str).values)
# peakcounts.count_peaks(folder_data_preproc / "fragments.tsv.gz", transcriptome.adata.obs["ix"].astype(str).values)

# %%
# peakcounts.count_peaks(folder_data_preproc / "atac_fragments.tsv.gz", transcriptome.obs.index)

# %%
peakcounts.counts.shape

# %%
adata_atac = sc.AnnData(peakcounts.counts.astype(np.float32), obs = transcriptome.obs)
sc.pp.normalize_total(adata_atac)
sc.pp.log1p(adata_atac)
sc.pp.highly_variable_genes(adata_atac)

# %%
import sklearn

# %%
adata_atac.var["highly_variable"].sum()

# %%
sc.pp.pca(adata_atac, use_highly_variable=False)
sc.pp.neighbors(adata_atac, use_rep = "X_pca")
sc.tl.umap(adata_atac)
sc.pl.umap(adata_atac)

# %%
adata_transcriptome = transcriptome.adata
sc.pp.neighbors(adata_transcriptome)
sc.tl.leiden(adata_transcriptome, resolution = 1.0)

# %%
adata_transcriptome

# %%
sc.pl.umap(adata_transcriptome, color = ["leiden"])

# %%
adata_atac.obs["leiden_transcriptome"] = adata_transcriptome.obs["leiden"]
adata_atac.obs["n_fragments"] = np.log1p(torch.bincount(fragments.cellmapping, minlength = fragments.n_cells).cpu().numpy())

# %%
sc.pl.umap(adata_atac, color = ["leiden_transcriptome", "n_fragments"], legend_loc = "on data")

# %%
# sc.pp.neighbors(adata_transcriptome, n_neighbors=50, key_added = "100")
# sc.pp.neighbors(adata_atac, n_neighbors=50, key_added = "100")

sc.pp.neighbors(adata_transcriptome, n_neighbors=100, key_added = "100")
sc.pp.neighbors(adata_atac, n_neighbors=100, key_added = "100")

# %%
assert (adata_transcriptome.obs.index == adata_atac.obs.index).all()

# %%
A = np.array(adata_transcriptome.obsp["100_connectivities"].todense() != 0)
B = np.array(adata_atac.obsp["100_connectivities"].todense() != 0)

# %%
intersect = A * B
union = (A+B) != 0

# %%
ab = intersect.sum() / union.sum()
ab

# %%
C = A[np.random.choice(A.shape[0], A.shape[0], replace = False)]
# C = B[np.random.choice(B.shape[0], B.shape[0], replace = False)]

# %%
intersect = C * B
union = (C+B) != 0

# %%
ac = intersect.sum() / union.sum()
ac

# %%
ab/ac

# %%
adata_atac.obs["ix"] = np.arange(adata_atac.obs.shape[0])

# %%
cells_train = np.arange(0, int(fragments.n_cells * 5 / 10))
cells_validation = np.arange(int(fragments.n_cells * 5 / 10), fragments.n_cells)
# cells_validation = adata_atac.obs.iloc[cells_validation].loc[adata_atac.obs["n_fragments"] > 8]["ix"].values

# %%
import xgboost
import sklearn
classifier = xgboost.XGBClassifier()
classifier.fit(adata_atac.obsm["X_pca"][cells_train], adata_atac.obs["leiden_transcriptome"].astype(int)[cells_train])
prediction = classifier.predict(adata_atac.obsm["X_pca"][cells_validation])
sklearn.metrics.balanced_accuracy_score(adata_atac.obs["leiden_transcriptome"].astype(int)[cells_validation], prediction)
# sklearn.metrics.accuracy_score(adata_atac.obs["leiden_transcriptome"].astype(int)[cells_validation], prediction)

# %%
adata_atac.obsm["X_pca"].shape

# %% [markdown]
# ----

# %%
from sklearn.neighbors import NearestNeighbors


# %%
def sk_indices_to_sparse(indices):
    k = indices.shape[1]-1
    A = scipy.sparse.coo_matrix((np.repeat(1, k * indices.shape[0]), (indices[:, 1:].flatten(), np.repeat(np.arange(indices.shape[0]), k))), shape = (indices.shape[0], indices.shape[0]))
    return A.tocsr()


# %%
ks = [2, 5, 10, 20, 50, 100, 200, 500]
k = max(ks)

# %%
X = adata_atac.obsm["X_pca"]

nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs = 10).fit(X)
distances, indices_A = nbrs.kneighbors(X)

# %%
X = adata_transcriptome.obsm["X_pca"]

nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', n_jobs = 10).fit(X)
distances, indices_B = nbrs.kneighbors(X)

# %%
np.arange(transcriptome.adata.obs.shape[0])

# %%
import scipy
def sparse_jaccard(m1,m2):
    intersection = m1.multiply(m2).sum(axis=1)
    a = m1.sum(axis=1)
    b = m2.sum(axis=1)
    jaccard = intersection/(a+b-intersection)

    # force jaccard to be 0 even when a+b-intersection is 0
    jaccard = np.nan_to_num(jaccard)
    return np.array(jaccard)#.flatten() 


# %%
def score_knn(indices_A, indices_B, ks):
    scores = []
    for k in ks:
        A = sk_indices_to_sparse(indices_A[:, :(k+1)])
        B = sk_indices_to_sparse(indices_B[:, :(k+1)])
        jac = sparse_jaccard(A, B).mean()

        # random_jac = np.zeros_like(jac)
        # for i in range(20):
        #     A_rand = sk_indices_to_sparse(indices_A[np.random.choice(A.shape[0], A.shape[0], replace = False), :(k+1)])
        #     random_jac += sparse_jaccard(A_rand, B).mean()
        # random_jac = random_jac / 20
        # print(random_jac)
        scores.append({
            "k":k,
            "jaccard":jac,
            # "norm_jaccard":jac/random_jac
        })
    scores = pd.DataFrame(scores)
    return scores


# %%
scores = score_knn(indices_A, indices_B, ks) # external baseline
scores

# %%
scores = score_knn(indices_A, indices_B, ks) # external baseline
scores

# %% [markdown]
# ## Loader

# %%
import peakfreeatac.loaders.peakcounts

# %%
loader = pfa.loaders.peakcounts.Peakcounts(fragments, peakcounts)

# %%
cells_oi = np.arange(fragments.n_cells)
genes_oi = np.arange(fragments.n_genes)
cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

minibatch = pfa.loaders.minibatching.Minibatch(cells_oi, genes_oi, cellxgene_oi)

# %%
data = loader.load(minibatch)

# %% [markdown]
# ## Model

# %%
import umap

# %%
model = pickle.load((pfa.get_output() / "prediction_vae/pbmc10k/10k10k/stack/pca_50/model_0.pkl").open("rb"))

# %%
embedding = model.forward(data)

# %%
adata_atac = sc.AnnData(peakcounts.counts.astype(np.float32), obs = transcriptome.obs)
adata_atac.obsm["X_pca"] = embedding

# %%
embedding.shape

# %%
sc.pp.neighbors(adata_atac, use_rep = "X_pca")
sc.tl.umap(adata_atac)
sc.pl.umap(adata_atac)

# %%
adata_transcriptome = transcriptome.adata
sc.pp.neighbors(adata_transcriptome)
sc.tl.leiden(adata_transcriptome, resolution = 1.0)

# %%
adata_atac.obs["leiden_transcriptome"] = adata_transcriptome.obs["leiden"]
adata_atac.obs["n_fragments"] = np.log1p(torch.bincount(fragments.cellmapping, minlength = fragments.n_cells).cpu().numpy())

# %%
sc.pl.umap(adata_atac, color = ["leiden_transcriptome"], legend_loc="on data")

# %% [markdown]
# ## Differential

# %%
adata_atac = sc.AnnData(peakcounts.counts.astype(np.float32), obs = transcriptome.obs, var = peakcounts.var)
sc.pp.normalize_total(adata_atac)
sc.pp.log1p(adata_atac)
sc.pp.highly_variable_genes(adata_atac)

# %%
adata_transcriptome = transcriptome.adata
sc.pp.neighbors(adata_transcriptome)
sc.tl.leiden(adata_transcriptome, resolution = 0.1)

# %%
# sc.pl.umap(adata_transcriptome, color = ["leiden", "celltype", *transcriptome.gene_id(["CD19", "SPI1", "TCF4", "CEBPB", "NRF1"])])
sc.pl.umap(adata_transcriptome, color = ["leiden", *transcriptome.gene_id(["Neurod1"])])

# %%
adata_atac.obs["leiden_transcriptome"] = adata_transcriptome.obs["leiden"]
adata_atac.obs["n_fragments"] = np.log1p(torch.bincount(fragments.cellmapping, minlength = fragments.n_cells).cpu().numpy())

# %%
sc.tl.rank_genes_groups(adata_atac, "leiden_transcriptome")

# %% [markdown]
# ----

# %%
group_id = "0"

# %%
adata_atac.obsm["X_umap"] = adata_transcriptome.obsm["X_umap"]
# sc.pl.umap(adata_atac, color = "chr9:107489462-107490358")

# %%
peakscores = sc.get.rank_genes_groups_df(adata_atac, group = group_id).rename(columns = {"names":"peak", "scores":"score"}).set_index("peak")
peakscores_joined = peakcounts.peaks.join(peakscores, on = "peak").sort_values("score", ascending = False)

# peakscores_joined = peakscores_joined.query("(relative_start > 5000) | (relative_end < -5000)")
# peakscores_joined = peakscores_joined.query("(relative_end < -1000)")

# %%
motifscan_folder = pfa.get_output() / "motifscans" / dataset_name / promoter_name / "cutoff_001"
motifscan = pfa.data.Motifscan(motifscan_folder)
motifs_oi = pickle.load((folder_data_preproc / "motifs_oi.pkl").open("rb"))

# %%
positive = peakscores_joined["logfoldchanges"] > 2.0
negative = ~positive

# %%
motif_indices = []
n = 0
for _, (relative_start, relative_end, gene_ix) in peakscores_joined[["relative_start", "relative_end", "gene_ix"]].loc[positive].iterrows():
    start_ix = gene_ix * (window[1] - window[0]) + relative_start - window[0]
    end_ix = gene_ix * (window[1] - window[0]) + relative_end - window[0]
    motif_indices.append(motifscan.indices[motifscan.indptr[start_ix]:motifscan.indptr[end_ix]])
    n += relative_end - relative_start

# %%
motif_indices = np.hstack(motif_indices)
motif_counts = np.bincount(motif_indices, minlength = motifs_oi.shape[0])


# %%
def count_motifs(relative_starts, relative_end, gene_ixs, motifscan_indices, motifscan_indptr):
    motif_indices = []
    n = 0
    for relative_start, relative_end, gene_ix in tqdm.tqdm(zip(relative_starts, relative_end, gene_ixs)):
        start_ix = gene_ix * (window[1] - window[0]) + relative_start - window[0]
        end_ix = gene_ix * (window[1] - window[0]) + relative_end - window[0]
        motif_indices.append(motifscan_indices[motifscan_indptr[start_ix]:motifscan_indptr[end_ix]])
        n += relative_end - relative_start
    motif_indices = np.hstack(motif_indices)
    motif_counts = np.bincount(motif_indices, minlength = len(motifs_oi))
    
    return motif_counts, n


# %%
motif_counts, n = count_motifs(peakscores_joined.loc[positive, "relative_start"], peakscores_joined.loc[positive, "relative_end"], peakscores_joined.loc[positive, "gene_ix"], motifscan.indices, motifscan.indptr)

# %%
motif_counts2, n2 = count_motifs(peakscores_joined.loc[negative, "relative_start"], peakscores_joined.loc[negative, "relative_end"], peakscores_joined.loc[negative, "gene_ix"], motifscan.indices, motifscan.indptr)

# %%
# motif_indices = []
# n2 = 0
# for _, (relative_start, relative_end, gene_ix) in peakscores_joined[["relative_start", "relative_end", "gene_ix"]].loc[negative].iterrows():
#     start_ix = gene_ix * (window[1] - window[0]) + relative_start - window[0]
#     end_ix = gene_ix * (window[1] - window[0]) + relative_end - window[0]
#     motif_indices.append(motifscan.indices[motifscan.indptr[start_ix]:motifscan.indptr[end_ix]])
#     n2 += relative_end - relative_start
    
# motif_indices = np.hstack(motif_indices)
# motif_counts2 = np.bincount(motif_indices, minlength = motifs_oi.shape[0])

# %%
motif_counts2, n2 = np.bincount(motifscan.indices, minlength = motifs_oi.shape[0]), fragments.n_genes * (window[1] - window[0])

# %%
motifscores = pd.DataFrame({
    "odds":(motif_counts / n) / (motif_counts2 / n2),
    "motif":motifs_oi.index
}).set_index("motif")
motifscores["logodds"] = np.log(motifscores["odds"])

# %%
motifscores.loc[motifscores.index.str.contains("SPI")]

# %%
n

# %%
n/n2

# %%
motifscores.sort_values("odds", ascending = False).head(40)

# %%
sns.histplot(motifscores["odds"])

# %%
n/n2

# %% [markdown]
# ----

# %%
group_id = "0"

# %%
peakscores = sc.get.rank_genes_groups_df(adata_atac, group = group_id).rename(columns = {"names":"peak", "scores":"score"}).set_index("peak")
peakscores_joined = peakcounts.peaks.join(peakscores, on = "peak").sort_values("score", ascending = False)

# %%
positive = peakscores_joined["logfoldchanges"] > 1.0
negative = ~positive

# %%
plt.scatter(peakscores_joined["logfoldchanges"], peakscores_joined["pvals_adj"], s = 1)
plt.xlim(-2, 2)
plt.axvline(1.0)

# %%
position_slices = peakscores_joined.loc[positive, ["relative_start", "relative_end"]].values
position_slices = position_slices - window[0]
gene_ixs_slices = peakscores_joined.loc[positive, "gene_ix"].values

# %%
sns.histplot(position_slices[:, 0], bins = np.linspace(0, (window[1] - window[0]), 20))
sns.histplot(position_slices[:, 1], bins = np.linspace(0, (window[1] - window[0]), 20))

# %% [markdown]
# GC content and background

# %%
onehot_promoters = pickle.load((folder_data_preproc / ("onehot_promoters_" + promoter_name + ".pkl")).open("rb")).flatten(0, 1)


# %%
def count_gc(relative_starts, relative_end, gene_ixs, onehot_promoters):
    gc = []
    n = 0
    for relative_start, relative_end, gene_ix in tqdm.tqdm(zip(relative_starts, relative_end, gene_ixs)):
        start_ix = gene_ix * (window[1] - window[0]) + relative_start
        end_ix = gene_ix * (window[1] - window[0]) + relative_end
        gc.append(onehot_promoters[start_ix:end_ix, [1, 2]].sum() / (end_ix - start_ix + 1e-5))
        
    gc = torch.hstack(gc).numpy()
    
    return gc


# %%
promoter_gc = count_gc(torch.ones(fragments.n_genes, dtype = torch.int) * window[0], torch.ones(fragments.n_genes, dtype = torch.int) * window[1], torch.arange(fragments.n_genes), onehot_promoters)
sns.histplot(
    promoter_gc,
    bins = np.linspace(0, 1, 50),
    stat = "density"
)
plt.axvline(promoter_gc.mean(), color = "blue")

window_oi_gc = count_gc(position_slices[:, 0], position_slices[:, 1], gene_ixs_slices, onehot_promoters)
sns.histplot(
    window_oi_gc,
    bins = np.linspace(0, 1, 50),
    stat = "density"
)
plt.axvline(window_oi_gc.mean(), color = "orange")

# %%
n_random = 100
n_select_random = 10
position_slices_repeated = position_slices.repeat(n_random, 0)
random_position_slices = np.zeros_like(position_slices_repeated)
random_position_slices[:, 0] = np.random.randint(np.ones(position_slices_repeated.shape[0]) * window[0], np.ones(position_slices_repeated.shape[0]) * window[1] - (position_slices_repeated[:, 1] - position_slices_repeated[:, 0]))
random_position_slices[:, 1] = random_position_slices[:, 0] + (position_slices_repeated[:, 1] - position_slices_repeated[:, 0])
random_gene_ixs_slices = np.random.randint(fragments.n_genes, size = random_position_slices.shape[0])

# %%
window_random_gc = count_gc(random_position_slices[:, 0], random_position_slices[:, 1], random_gene_ixs_slices, onehot_promoters)

# %%
random_difference = np.abs((window_random_gc.reshape((position_slices.shape[0], n_random)) - window_oi_gc[:, None]))

chosen_background = np.argsort(random_difference, axis = 1)[:, :n_select_random].flatten()
chosen_background_idx = np.repeat(np.arange(position_slices.shape[0]), n_select_random) * n_random + chosen_background

background_position_slices = random_position_slices[chosen_background_idx]
background_gene_ixs_slices = random_gene_ixs_slices[chosen_background_idx]

# %%
plt.scatter(window_random_gc[chosen_background_idx[::n_select_random]], window_oi_gc)

# %%
window_background_gc = count_gc(background_position_slices[:, 0], background_position_slices[:, 1], background_gene_ixs_slices, onehot_promoters)

# %%
promoter_gc = count_gc(torch.ones(fragments.n_genes, dtype = torch.int) * window[0], torch.ones(fragments.n_genes, dtype = torch.int) * window[1], torch.arange(fragments.n_genes), onehot_promoters)
sns.histplot(
    promoter_gc,
    bins = np.linspace(0, 1, 50),
    stat = "density"
)
plt.axvline(promoter_gc.mean(), color = "blue")

window_oi_gc = count_gc(position_slices[:, 0], position_slices[:, 1], gene_ixs_slices, onehot_promoters)
sns.histplot(
    window_oi_gc,
    bins = np.linspace(0, 1, 50),
    stat = "density"
)
plt.axvline(window_oi_gc.mean(), color = "orange")
window_random_gc = count_gc(random_position_slices[:, 0], random_position_slices[:, 1], random_gene_ixs_slices, onehot_promoters)
sns.histplot(
    window_random_gc,
    bins = np.linspace(0, 1, 50),
    stat = "density"
)
plt.axvline(window_random_gc.mean(), color = "green")

window_background_gc = count_gc(background_position_slices[:, 0], background_position_slices[:, 1], background_gene_ixs_slices, onehot_promoters)
sns.histplot(
    window_background_gc,
    bins = np.linspace(0, 1, 50),
    stat = "density"
)
plt.axvline(window_background_gc.mean(), color = "red")

# %%
motifscan_folder = pfa.get_output() / "motifscans" / dataset_name / promoter_name / "cutoff_0001"
motifscan = pfa.data.Motifscan(motifscan_folder)
motifs = pickle.load((motifscan_folder / "motifs.pkl").open("rb"))


# %%
def count_motifs(relative_starts, relative_end, gene_ixs, motifscan_indices, motifscan_indptr):
    motif_indices = []
    n = 0
    for relative_start, relative_end, gene_ix in tqdm.tqdm(zip(relative_starts, relative_end, gene_ixs)):
        start_ix = gene_ix * (window[1] - window[0]) + relative_start
        end_ix = gene_ix * (window[1] - window[0]) + relative_end
        motif_indices.append(motifscan_indices[motifscan_indptr[start_ix]:motifscan_indptr[end_ix]])
        n += relative_end - relative_start
    motif_indices = np.hstack(motif_indices)
    motif_counts = np.bincount(motif_indices, minlength = len(motifs))
    
    return motif_counts, n


# %%
motif_counts, n = count_motifs(position_slices[:, 0], position_slices[:, 1], gene_ixs_slices, motifscan.indices, motifscan.indptr)

# %%
n

# %%
motif_counts2, n2 = count_motifs(background_position_slices[:, 0], background_position_slices[:, 1], background_gene_ixs_slices, motifscan.indices, motifscan.indptr)
# motif_counts2, n2 = np.bincount(motifscan.indices, minlength = motifs_oi.shape[0]), fragments.n_genes * (window[1] - window[0])

# %%
n/n2

# %%
n / (fragments.n_genes * (window[1] - window[0]))

# %%
contingencies = np.stack([
    np.stack([n2 - motif_counts2, motif_counts2]),
    np.stack([n - motif_counts, motif_counts]),
]).transpose(2, 0, 1)
import scipy.stats
odds_conditional = []
for cont in contingencies:
    odds_conditional.append(scipy.stats.contingency.odds_ratio(cont + 1, kind='conditional').statistic) # pseudocount = 1

motifscores = pd.DataFrame({
    "odds":((motif_counts+1) / (n+1)) / ((motif_counts2+1) / (n2+1)),
    "odds_conditional":odds_conditional,
    "motif":motifs.index
}).set_index("motif")
motifscores["logodds"] = np.log(odds_conditional)

# %%
motifscores.loc[motifscores.index.str.contains("NEUR")]

# %%
motifscores.sort_values("odds", ascending = False)

# %%
motifscores2 = pd.read_csv(pfa.get_output() / "a.csv", index_col = 0)

# %%
import scipy

# %%
motifs_oi = np.repeat(True, motifscores.shape[0])
# motifs_oi = (motifscores["logodds"].abs() > np.log(1.2)) | (motifscores2["logodds"].abs() > np.log(1.2))
# motifs_oi = (motifscores["logodds"].abs() > np.log(1.2)) & (motifscores2["logodds"].abs() > np.log(1.2))
# motifs_oi = (motifscores["logodds"].abs() > np.log(1.5)) | (motifscores2["logodds"].abs() > np.log(1.5))
# motifs_oi = (motifscores["logodds"].abs() > np.log(1.5)) & (motifscores2["logodds"].abs() > np.log(1.5))
# motifs_oi = (motifscores["logodds"].abs() > np.log(1.5))

# %%
# linreg = scipy.stats.linregress(motifscores.loc[motifs_oi, "logodds"], motifscores2.loc[motifs_oi, "logodds"])
linreg = scipy.stats.linregress(motifscores2.loc[motifs_oi, "logodds"], motifscores.loc[motifs_oi, "logodds"])
slope = linreg.slope
intercept = linreg.intercept

# %%
1/slope

# %%
fig, ax = plt.subplots()

ax.set_aspect(1)
ax.scatter(motifscores["logodds"], motifscores2["logodds"], color = "grey", s = 1.0)
ax.scatter(motifscores.loc[motifs_oi, "logodds"], motifscores2.loc[motifs_oi, "logodds"], color = "orange", s = 2)
ax.axline((0, 0), slope=1, color = "#555", zorder = 0, lw = 1)
ax.axline((0, 0), slope=1.5, color = "#888", zorder = 0, lw = 1)
ax.axline((0, 0), slope=1/1.5, color = "#888", zorder = 0, lw = 1)
ax.axline((0, 0), slope=2, color = "#aaa", zorder = 0, lw = 1)
ax.axline((0, 0), slope=0.5, color = "#aaa", zorder = 0, lw = 1)
ax.axline((0, 0), slope=3, color = "#ccc", zorder = 0, lw = 1)
ax.axline((0, 0), slope=1/3, color = "#ccc", zorder = 0, lw = 1)
ax.axline((0, 0), slope=6, color = "#ddd", zorder = 0, lw = 1)
ax.axline((0, 0), slope=1/6, color = "#ddd", zorder = 0, lw = 1)
ax.axline((0, -intercept), slope=1/slope, color = "red")

# %%
motifscores2

# %%
sns.histplot(motifscores["odds"])
sns.histplot(motifscores2["odds"])

# %% [markdown]
# ## Predict (temporarily here 👷)

# %%
import peakfreeatac.prediction

# %%
peaks_names = [
    # "cellranger",
    # "macs2",
    "rolling_500"
]
design_peaks = pd.DataFrame({"peaks":peaks_names})
methods = [
    ["_xgboost", peakfreeatac.prediction.PeaksGene],
    # ["_linear", peakfreeatac.prediction.PeaksGeneLinear],
    # ["_polynomial", peakfreeatac.prediction.PeaksGenePolynomial],
    # ["_lasso", peakfreeatac.prediction.PeaksGeneLasso]
]
design_methods = pd.DataFrame(methods, columns = ["method_suffix", "method_class"])
dataset_names = [
    "pbmc10k",
    "lymphoma",
    "e18brain",
]
design_datasets = pd.DataFrame({"dataset":dataset_names})

# %%
design = pfa.utils.crossing(design_peaks, design_methods, design_datasets)

# %%
for _, design_row in design.iterrows():
    dataset_name = design_row["dataset"]
    peaks_name = design_row["peaks"]
    transcriptome = pfa.data.Transcriptome(pfa.get_output() / "data" / dataset_name / "transcriptome")
    peakcounts = pfa.peakcounts.FullPeak(folder = pfa.get_output() / "peakcounts" / dataset_name / peaks_name)
    
    peaks = peakcounts.peaks
    
    gene_peak_links = peaks.reset_index()
    gene_peak_links["gene"] = pd.Categorical(gene_peak_links["gene"], categories = transcriptome.adata.var.index)
    
    fragments = peakfreeatac.data.Fragments(pfa.get_output() / "data" / dataset_name / "fragments" / promoter_name)
    folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
    
    method_class = design_row["method_class"]
    method_suffix = design_row["method_suffix"]
    prediction = method_class(
        pfa.get_output() / "prediction_positional" / dataset_name / promoter_name / (peaks_name + method_suffix),
        transcriptome,
        peakcounts
    )
    
    prediction.score(gene_peak_links, folds)
    
    prediction.scores = prediction.scores
    # prediction.models = prediction.models

# %%
prediction.scores

# %%
prediction.scores["cor"].unstack().T.sort_values("validation").plot()

# %%

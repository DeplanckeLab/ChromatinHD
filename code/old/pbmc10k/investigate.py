# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PBMC10K

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams.update({'font.size': 12, 'figure.constrained_layout.use':True, 'pdf.fonttype':42, 'ps.fonttype':42})
# %config InlineBackend.figure_format = 'retina'

import seaborn as sns
sns.set_style('ticks')

import torch

import pickle
import dill

import scanpy as sc

# %%
root = laf.get_git_root()
project_root = root / "output"
laf.set_project_root(project_root)

# %%
flow_data = laf.Flow("data")

# %%
flow_data_preproc = transcriptome = flow_data.get_flow("pbmc10kgran")

# %% [markdown]
# ### Download

# %% [markdown]
# https://support.10xgenomics.com/single-cell-atac/software/pipelines/latest/algorithms/overview

# %%
# !wget https://cf.10xgenomics.com/samples/cell-arc/1.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_analysis.tar.gz -O {flow_data_preproc.path}/analysis.tar.gz

# %%
# !tar -xvf {flow_data_preproc.path}/analysis.tar.gz

# %%
# # ! wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_10k/pbmc_unsorted_10k_filtered_feature_bc_matrix.h5 -P {flow_data_preproc.path}
# ! wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_filtered_feature_bc_matrix.h5 -O {flow_data_preproc.path}/filtered_feature_bc_matrix.h5

# %%
# # !wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_10k/pbmc_unsorted_10k_atac_fragments.tsv.gz -O {flow_data_preproc.path}/atac_fragments.tsv
# !wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz -O {flow_data_preproc.path}/atac_fragments.tsv.gz

# %%
# # !wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_10k/pbmc_unsorted_10k_atac_fragments.tsv.gz.tbi -O {flow_data_preproc.path}/atac_fragments.tsv.gz.tbi
# !wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz.tbi -O {flow_data_preproc.path}/atac_fragments.tsv.gz.tbi

# %%
# # !wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_10k/pbmc_unsorted_10k_atac_peaks.bed -O {flow_data_preproc.path}/atac_peaks.bed
# !wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_atac_peaks.bed -O {flow_data_preproc.path}/atac_peaks.bed

# %%
# !cat {flow_data_preproc.path}/atac_peaks.bed | sed '/^#/d' > {flow_data_preproc.path}/peaks.tsv

# %%
# # !wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_10k/pbmc_unsorted_10k_atac_peak_annotation.tsv -O {flow_data_preproc.path}/peak_annot.tsv
# !wget https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_atac_peak_annotation.tsv -O {flow_data_preproc.path}/peak_annot.tsv

# %%
# !wget http://ftp.ensembl.org/pub/release-107/gff3/homo_sapiens/Homo_sapiens.GRCh38.107.gff3.gz -O {flow_data_preproc.path}/genes.gff.gz

# %%
# !zcat {flow_data_preproc.path}/genes.gff.gz | grep -vE "^#" | awk '$3 == "gene"' > {flow_data_preproc.path}/genes.gff

# %%
# !wget https://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/fetchChromSizes

# %%
# !wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes -O  {flow_data_preproc.path}/chromosome.sizes

# %%
gff = pd.read_table(flow_data_preproc.path/ "genes.gff", sep = "\t", names = ["chr", "type", "__", "start", "end", "dot", "strand", "dot2", "info"])
gff["chr"] = "chr" + gff["chr"]
gff["gene"] = gff["info"].str.split(";").str[1].str[5:]

# %%
import pybedtools

# %%
genewindows = gff.copy()
genewindows["start"] = np.maximum(genewindows["start"] - 20000, 0)
genewindows["end"] = genewindows["end"] + 20000

# %%
genewindows_bed = pybedtools.BedTool.from_dataframe(genewindows[["chr", "start", "end", "strand", "gene"]])

# %%
peaks = pybedtools.BedTool(flow_data_preproc.path/"atac_peaks.bed")

# %%
flow_data_preproc.gene_peak_links["peak"].isin(accessibility.adata.var.index).sum()

# %%
intersect = genewindows_bed.intersect(peaks, wo = True)
intersect = intersect.to_dataframe()

# %%
intersect["peak_id"] = intersect["strand"] + ":" + intersect["thickStart"].astype(str) + "-" + intersect["thickEnd"].astype(str)

# %%
gene_peak_links = intersect[["score", "peak_id"]].copy()
gene_peak_links.columns = ["gene", "peak"]

# %%
flow_data_preproc.gene_peak_links = gene_peak_links

# %%
accessibility.adata.var.index[accessibility.adata.var.index.str.startswith("chr1:")]

# %%
flow_data_preproc.peak_annot.query("start == 904341")

# %%
intersect

# %%
flow_data_preproc.gene_peak_links["peak"]

# %%
flow_data_preproc.gene_peak_links["peak"]

# %%
flow_data_preproc.gene_peak_links["peak"].isin(accessibility.adata.var.index.tolist()).mean()

# %%
# # !zcat {flow_data_preproc.path}/atac_fragments.tsv.gz | head -n 100

# %%
flow_data_preproc.peak_annot = laf.objects.DataFrame()

# %% [markdown]
# ### Load expression

# %%
adata = sc.read_10x_h5(flow_data_preproc.path / "filtered_feature_bc_matrix.h5")

# %%
adata.var_names_make_unique()

# %%
sc.pp.normalize_per_cell(adata)

# %%
sc.pp.log1p(adata)

# %%
sc.pp.pca(adata)

# %%
sc.pp.highly_variable_genes(adata)

# %%
sc.pp.neighbors(adata)

# %%
sc.tl.umap(adata)

# %%
sc.pl.umap(adata, color = ["PTPRC", "CD3E", "CD9", "MERTK", "ANPEP", "FCGR3A"])

# %%
flow_data_preproc.adata = adata

# %%
flow_data_preproc.variable_gene_ids = flow_data_preproc.adata.var.query("dispersions_norm > 0.5").index
print(len(flow_data_preproc.variable_gene_ids))

# %% [markdown]
# ## Fragment distribution

# %%
import gzip

# %%
sizes = []
with gzip.GzipFile(flow_data_preproc.path / "atac_fragments.tsv.gz", "r") as fragment_file:
    i = 0
    for line in fragment_file:
        line = line.decode("utf-8")
        if line.startswith("#"):
            continue
        split = line.split("\t")
        sizes.append(int(split[2]) - int(split[1]))
        i += 1
        if i > 1000000:
            break

# %%
sizes = np.array(sizes)

# %%
fig, ax = plt.subplots()
sns.histplot(sizes)
ax.set_xlim(0, 1000)

# %% [markdown]
# ## Create peaks

# %%
import sys
if str(laf.get_git_root() / "package") not in sys.path:
    sys.path.append(str(laf.get_git_root() / "package"))

# %%
import peakcheck

# %%
flow_accessibilities = laf.Flow("accessibilities")

# %%
flow_data_preproc.peak_annot.query("gene == 'RELA'")

# %%
# accessibility = peakcheck.accessibility.FullPeak(
accessibility = peakcheck.accessibility.HalfPeak(
# accessibility = peakcheck.accessibility.BroaderPeak(
# accessibility = peakcheck.accessibility.FragmentPeak(
    flow = flow_accessibilities / flow_data_preproc.name,
    reset = True
)

# %%
accessibility.create_peaks(flow_data_preproc.peak_annot, flow_data_preproc.variable_gene_ids)

# %%
accessibility.count_peaks(flow_data_preproc.path / "atac_fragments.tsv.gz", flow_data_preproc.adata.obs.index)

# %%
sns.histplot(np.log10(np.array(accessibility.counts.sum(0))[0]))

# %%
sns.histplot(np.log10(np.array(accessibility.counts.sum(1))[:, 0]))

# %%
accessibility.create_adata(flow_data_preproc.adata)

# %% [markdown]
# ## Load peaks

# %%
transcriptome.adata.var.sort_values("dispersions_norm")

# %%
# gene = "IGHE"
gene = "CD3D"
# gene = "IGHD"
# gene = "PTPRC"
# gene = "AL136456.1"
# gene = "RALGPS2"
# gene = "ANGPTL1"
# gene = "CD247"
# gene = "NVL"
# gene = "PRF1"
# gene = "IGHA1"

# %%
sc.pl.umap(flow_data_preproc.adata, color = [gene])

# %%
peaks_oi = accessibility.var.query("gene == @gene").index
peaks_oi = accessibility.var.index[accessibility.var["original_peak"].isin(flow_data_preproc.gene_peak_links.query("gene == @gene")["peak"].tolist())]

# %%
sc.pl.umap(accessibility.adata, color = peaks_oi[:10])
# sc.pl.umap(accessibility.adata, color = peak_ids[:10])

# %% [markdown]
# ## Prediction

# %%
# prediction = peakcheck.prediction.GenePrediction(
#     flow = "prediction" / accessibility.path.relative_to(laf.Flow("accessibilities").path),
#     transcriptome = flow_data_preproc,
#     accessibility = accessibility
# )
prediction = peakcheck.prediction.OriginalPeakPrediction(
    flow = "prediction" / accessibility.path.relative_to(laf.Flow("accessibilities").path),
    transcriptome = flow_data_preproc,
    accessibility = accessibility
)

# %%
prediction.score()

# %%
prediction.scores.sort_values("mse_test_ratio")

# %%
scores = prediction.scores

# %%
np.log2(scores.loc[~pd.isnull(scores["mse_test_ratio"]) & ~np.isinf(scores["mse_test_ratio"])]["mse_test_ratio"]).plot(kind = "hist")
# np.log(scores.loc[~pd.isnull(scores["mse_train_ratio"]) & ~np.isinf(scores["mse_train_ratio"])]["mse_train_ratio"]).plot(kind = "hist")

# %% [markdown]
# ### Look at one gene

# %%
accessibility = accessibility
transcriptome = flow_data_preproc

X_transcriptome = transcriptome.adata.X.tocsc().todense()
X_accessibility = accessibility.adata.X.tocsc().todense()

def extract_data(gene_oi, peaks_oi):
    x = np.array(X_accessibility[:, peaks_oi["ix"]])
    y = np.array(X_transcriptome[:, gene_oi["ix"]])[:, 0]
    return x, y


# %%
gene = scores.groupby("gene").mean().sort_values("mse_test_ratio").index[0]
gene = "IGHD"

peaks_oi = accessibility.var.query("gene == @gene")
gene_oi = transcriptome.adata.var.loc[gene]
print(gene_oi)

x, y = extract_data(gene_oi, peaks_oi)

# %%
import latenta as la

# %%
fig, axes = la.plotting.axes_wrap(peaks_oi.shape[0], w = 1, h = 1, n_col = 10)
for peak_ix, ax in enumerate(axes):
    sns.boxplot(x = x[:,peak_ix], y = y, ax = ax)
    ax.axis("off")

# %%
sc.pl.umap(transcriptome.adata, color = [gene])

# %%
sc.pl.umap(accessibility.adata, color = peaks_oi.index[:10])

# %% [markdown]
# ## Compare (gene predictions)

# %%
scores = []
peak_sources = ["full_peak", "fragment_peak"]
for peak_source in peak_sources:
    prediction = laf.Flow("prediction/" + transcriptome.name + "/" + peak_source + "/geneprediction")
    scores_flow = prediction.scores.copy()
    scores_flow["peak_source"] = peak_source
    scores.append(scores_flow)
scores = pd.concat(scores)

# %%
scores_aggregated = scores.groupby(["gene", "peak_source"]).mean().reset_index()

# %%
scores_aggregated.loc[np.isinf(scores_aggregated["mse_test_ratio"]), "mse_test_ratio"] = 1.

# %%
sns.violinplot(scores_aggregated["peak_source"], scores_aggregated["mse_test_ratio"])

# %%
scores_unstacked = scores_aggregated.set_index(["gene", "peak_source"])["mse_test"].unstack()

# %%
sns.histplot(np.clip(np.log(scores_unstacked["fragment_peak"]) - np.log(scores_unstacked["full_peak"]), -1, 1))

# %%
scores_mse_test = scores_aggregated.set_index(["gene", "peak_source"])["mse_train"].unstack()
scores_mse_test_mean = scores_aggregated.set_index(["gene", "peak_source"])["mse_train_mean"].unstack()

# scores_mse_test = scores_aggregated.set_index(["gene", "peak_source"])["mse_test"].unstack()
# scores_mse_test_mean = scores_aggregated.set_index(["gene", "peak_source"])["mse_test_mean"].unstack()

# %%
relative_scores = {}
for peak_source in peak_sources:
    relative_scores_source = pd.DataFrame({
        "mse_test":scores_mse_test[peak_source],
        "mse_test_ratio":scores_mse_test[peak_source]/scores_mse_test_mean[peak_source],
        "mse_test_ratio_full":scores_mse_test["full_peak"]/scores_mse_test_mean["full_peak"],
        "mse_test_diff_full":scores_mse_test[peak_source]-scores_mse_test["full_peak"],
        "mse_test_ratio_diff_full":scores_mse_test[peak_source]/scores_mse_test_mean[peak_source]-scores_mse_test["full_peak"]/scores_mse_test_mean["full_peak"],
    })
    relative_scores[peak_source] = relative_scores_source
relative_scores = pd.concat(relative_scores, names = ["peak_source"])

# %%
# peak_source = "half_peak"
# peak_source = "third_peak"
# peak_source = "broader_peak"
peak_source = "fragment_peak"
fig, (ax, ax1) = plt.subplots(1, 2, figsize = (6, 3))
sns.scatterplot(relative_scores.loc[peak_source]["mse_test_ratio"], relative_scores.loc[peak_source]["mse_test_diff_full"], ax = ax)
ax.axhline(0)
ax.axvline(1)
sns.scatterplot(relative_scores.loc["full_peak"]["mse_test_ratio"], relative_scores.loc[peak_source]["mse_test_ratio"], ax = ax1)
ax1.set_xlabel("MSE ratio full peak")
ax1.set_ylabel(f"MSE ratio {peak_source}")
ax1.axhline(1)
ax1.axvline(1)


# %%
(scores_mse_test["fragment_peak"] < scores_mse_test["full_peak"]).mean()

# %%
accessibility = peakcheck.accessibility.FragmentPeak(
# accessibility = peakcheck.accessibility.BroaderPeak(
    flow = flow_accessibilities / flow_data_preproc.name,
)

# %%
relative_scores.loc[accessibility.name].query("mse_test_ratio < 1.0").sort_values("mse_test_diff_full").head(20)

# %%
gene = "CD74"
gene = "FOSB"
gene = "GNLY"
gene = "AC020916.1"
gene = "IL1B"
gene = "CD79A"
gene = "ADGRE5"

# %%
peaks_oi = accessibility.var.query("gene == @gene")
peak_ids = peaks_oi.index.tolist()

# %%
x_expression = sc.get.obs_df(transcriptome.adata, gene)
x_peaks = sc.get.obs_df(accessibility.adata, peak_ids)
x_peaks_full = sc.get.obs_df(accessibility_full.adata, peak_ids_full)

# %%
fig, axes = la.plotting.axes_wrap(len(peak_ids), w = 1, h = 1, n_col = 8)
for peak_id, ax in zip(peak_ids, axes):
    sns.boxplot(x = x_peaks[peak_id], y = x_expression, ax = ax)
    ax.axis("off")
    # ax.set_title("{:.2f}".format(peaks_oi.loc[peak_id]))

# %%
plotdata = pd.DataFrame({"gene":x_expression})
plotdata[x_peaks.columns + "_new"] = x_peaks
plotdata[x_peaks_full.columns + "_full"] = x_peaks_full

# %%
fig, ax = plt.subplots(figsize = (6, 10))
sns.heatmap(plotdata.sort_values("gene"), norm = mpl.colors.Normalize(0, 1))

# %% [markdown]
# ## Compare (peak-wise predictions)

# %%
scores = []
peak_sources = [
    # "broader_peak",
    # "fragment_peak",
    "full_peak",
    "half_peak"
]
for peak_source in peak_sources:
    print(peak_source)
    prediction = laf.Flow("prediction/" + transcriptome.name + "/" + peak_source + "/originalpeakprediction")
    scores_flow = prediction.scores.copy()
    scores_flow["peak_source"] = peak_source
    scores.append(scores_flow)
scores = pd.concat(scores)

# %%
scores_aggregated = scores.groupby(["gene", "original_peak", "peak_source"]).mean().reset_index()

# %%
# scores_aggregated.loc[scores_aggregated["mse_test_ratio"] > 1.1, "mse_test_ratio"] = 1.1

# %%
scores_aggregated.loc[pd.isnull(scores_aggregated["mse_test_ratio"]) , "mse_test_ratio"] = 1.
scores_aggregated.loc[np.isinf(scores_aggregated["mse_test_ratio"]) , "mse_test_ratio"] = 1.

# %%
sns.violinplot(scores_aggregated["peak_source"], scores_aggregated["mse_test_ratio"])

# %%
scores_unstacked = scores_aggregated.set_index(["gene", "original_peak", "peak_source"])["mse_test"].unstack()
scores_unstacked = scores_aggregated.set_index(["gene", "original_peak", "peak_source"])["mse_train"].unstack()

# %%
sns.histplot(np.log(scores_unstacked["half_peak"] / scores_unstacked["full_peak"]))
# sns.histplot(np.log(scores_unstacked["broader_peak"] / scores_unstacked["full_peak"]))
# sns.histplot(np.log(scores_unstacked["fragment_peak"] / scores_unstacked["full_peak"]))

# %%
scores_mse_test = scores_aggregated.set_index(["gene", "original_peak", "peak_source"])["mse_test"].unstack()
scores_mse_test_ratio = scores_aggregated.set_index(["gene", "original_peak", "peak_source"])["mse_test_ratio"].unstack()
scores_mse_test_mean = scores_aggregated.set_index(["gene", "original_peak", "peak_source"])["mse_test_mean"].unstack()
# scores_mse_train = scores_aggregated.set_index(["gene", "peak_source"])["mse_train"].unstack()

# %% tags=[]
relative_scores = {}
for peak_source in peak_sources:
    relative_scores_source = pd.DataFrame({
        "mse_test":scores_mse_test[peak_source],
        "mse_test_ratio":scores_mse_test[peak_source]/scores_mse_test_mean[peak_source],
        "mse_test_ratio_full":scores_mse_test["full_peak"]/scores_mse_test_mean["full_peak"],
        "mse_test_diff_full":scores_mse_test[peak_source]-scores_mse_test["full_peak"],
        "mse_test_ratio_diff_full":(scores_mse_test[peak_source]/scores_mse_test_mean[peak_source])-(scores_mse_test["full_peak"]/scores_mse_test_mean["full_peak"]),
    })
    relative_scores_source["mse_test_improvement"] = (1-relative_scores_source["mse_test_ratio"]) / (1-relative_scores_source["mse_test_ratio_full"])
    relative_scores[peak_source] = relative_scores_source
relative_scores = pd.concat(relative_scores, names = ["peak_source"])

# %%
# accessibility = peakcheck.accessibility.BroaderPeak(
# accessibility = peakcheck.accessibility.FragmentPeak(
accessibility = peakcheck.accessibility.HalfPeak(
    flow = flow_accessibilities / flow_data_preproc.name,
)

# %%
peak_source = accessibility.name
fig, (ax, ax1) = plt.subplots(1, 2, figsize = (6, 3))
sns.scatterplot(relative_scores.loc[peak_source]["mse_test_ratio"], relative_scores.loc[peak_source]["mse_test_diff_full"], ax = ax)
ax.axhline(0)
ax.axvline(1)
sns.scatterplot(relative_scores.loc["full_peak"]["mse_test_ratio"], relative_scores.loc[peak_source]["mse_test_ratio"], ax = ax1)
ax1.set_xlabel("MSE ratio full peak")
ax1.set_ylabel(f"MSE ratio {peak_source}")
ax1.set_aspect(1)
ax1.axhline(1)
ax1.axvline(1)


# %%
relative_scores.loc[accessibility.name].query("mse_test_ratio < 0.99").sort_values("mse_test_diff_full", ascending = True).head(20)

# %%
sc.pl.umap(transcriptome.adata, color = "ZEB2")

# %% [markdown]
# ## Example

# %%
transcriptome.adata.var.query("highly_variable").sort_values("means", ascending = False).head(20)

# %%
# gene = "PLXDC2"
# gene = "RBM47"
# gene = "NTM"
# gene = "STX11"
# gene = "EPHB1"
# gene = "GNLY"
# gene = "NKG7"
# gene = "AFF3"
# gene = "GNLY"
gene = "BACH1"
# gene = "PRAM1"
# gene = "STX11"
# gene = "NIBAN3"
# gene = "CD79A"
# gene = "CD74"
# gene = "IGKC"
# gene = "SOX5"
# gene = "BLK"
# gene = "CD79A"
# gene = "BACH1"
# gene = "BANK1"
# gene = "IGHG2"
# gene = "ZEB2"
# gene = "CD3D"
# gene = "PLXDC2"
# gene = "LYN"
# gene = "CCSER1"
# gene = "BCL2"
# gene = "BANK1"
# gene = "LEF1"
# gene = "PAX5"
# gene = "RORA"
# gene = "GAB2"
# gene = "NEAT1"
# gene = "TEX41"
# gene = "EPHB1"
# gene = "PLA2R1"
# gene = "MET"
# gene = "CD79B"

# %%
gene_ids = transcriptome.adata.var.query("(dispersions_norm > 1.5) & (log10(means) > -2)").index

# %%
scores_oi = relative_scores.loc["full_peak"]
scores_oi = scores_oi.loc[scores_oi.index.get_level_values(0).isin(gene_ids)]

# %%
relative_scores.loc[accessibility.name].loc[gene].sort_values("mse_test_diff_full", ascending = True)

# %%
scores_oi = relative_scores.loc[accessibility.name].sort_values("mse_test_diff_full").loc[gene]

# %%
peaks_oi = accessibility.var.query("gene == @gene").copy()
gene_oi = transcriptome.adata.var.loc[gene]

# %%
best_peak_original_peak = scores_oi.sort_values("mse_test_diff_full", ascending = True).index[0]
# best_peak_original_peak = scores_oi.sort_values("mse_test_ratio", ascending = True).index[0]

# %%
accessibility_full = peakcheck.accessibility.FullPeak(
    flow = flow_accessibilities / flow_data_preproc.name,
)
accessibility_full.var = laf.Py()
accessibility_full.var = accessibility_full.adata.var
accessibility_full.var.index = accessibility_full.var["original_peak"]
accessibility_full.var = accessibility_full.var
peak_ids_full = accessibility_full.var.query("original_peak == @best_peak_original_peak").index.unique().tolist()

# %%
accessibility_full.var.loc[best_peak_original_peak]

# %%
scores_oi.loc[best_peak_original_peak]

# %%
peak_ids = accessibility.var.query("original_peak == @best_peak_original_peak").index.tolist()
# accessibility.var.loc[peak_ids]

# %%
x_expression = sc.get.obs_df(transcriptome.adata, gene)
x_peaks = sc.get.obs_df(accessibility.adata, peak_ids)
x_peaks_full = sc.get.obs_df(accessibility_full.adata, peak_ids_full)

# %%
fig, axes = la.plotting.axes_wrap(len(peak_ids), w = 1, h = 1, n_col = 10)
for peak_id, ax in zip(peak_ids, axes):
    sns.boxplot(x = x_peaks[peak_id], y = x_expression, ax = ax)
    ax.axis("off")
    # ax.set_title("{:.2f}".format(peaks_oi.loc[peak_id]))

# %%
fig, axes = la.plotting.axes_wrap(len(peak_ids), w = 1, h = 1, n_col = 10)
for peak_id, ax in zip(peak_ids, axes):
    sns.ecdfplot(hue = x_peaks[peak_id] > 0, x = x_expression, ax = ax, legend = False)
    # ax.axis("off")
    # ax.set_title("{:.2f}".format(peaks_oi.loc[peak_id]))

# %%
plotdata = pd.DataFrame({"gene":x_expression})
plotdata[x_peaks.columns + "_new"] = x_peaks
plotdata[x_peaks_full.columns + "_full"] = x_peaks_full

# %%
# ((x_peaks > 0).astype(int) * np.array([1, 2, 4, 8])).sum(1).value_counts()

# %%
plotdata.sort_values("gene").plot()

# %%
fig, ax = plt.subplots(figsize = (8, 30))
sns.heatmap(plotdata.sort_values("gene"), norm = mpl.colors.Normalize(0, 3))

# %% [markdown]
# ------

# %%
import subprocess as sp

# %%
tabix_location = peakcheck.accessibility.tabix_location

# %%
import tqdm

# %%
fragments_location = flow_data_preproc.path / "atac_fragments.tsv.gz"

# %%
import math

# %%
window_info = gff.query("gene == @gene").iloc[0].copy()
window_info = accessibility_full.var.query("original_peak == @best_peak_original_peak").iloc[0].copy()

desired_window_size = 2000

padding = desired_window_size - (window_info["end"] - window_info["start"])

window_info["start"] = window_info["start"] - math.floor(padding/2)
window_info["end"] = window_info["end"] + math.ceil(padding/2)


# %%
def name_window(window_info):
    return window_info["chrom"] + ":" + str(window_info["start"]) + "-" + str(window_info["end"])

window = name_window(window_info)

# %%
process = sp.Popen([tabix_location, fragments_location, window], stdout=sp.PIPE, stderr=sp.PIPE)
# process.wait()
# out = process.stdout.read().decode("utf-8")
fragments = pd.read_table(process.stdout, names = ["chrom", "start", "end", "barcode", "i"])

# %%
X = np.zeros((x_expression.index.size, window_info["end"] - window_info["start"]))        

# %%
barcode_map = pd.Series(np.arange(x_expression.index.size), x_expression.index)

# %%
for _, fragment in fragments.iterrows():
    if fragment["barcode"] not in barcode_map:
        continue
    X[barcode_map[fragment["barcode"]], (fragment["start"] - window_info["start"]):(fragment["end"] - window_info["start"])] = 1.

# %%
cells_oi = np.random.choice(barcode_map.index, 10000, replace = False)
cells_oi = x_expression[cells_oi].sort_values().index
cells_oi_ix = barcode_map[cells_oi].values

# %%
fig, ax = plt.subplots(figsize = (5, 10))
ax.matshow(X[cells_oi_ix, ], aspect = 5.)

# %%
str(window_info["start"] + 400-50) + str(window_info["start"] + 400+50)

# %%
window_info2 = window_info.copy()
window_info2["start"] += 400

# %%
name_window(window_info2)

# %%
plt.plot(np.arange(X.shape[1]), X.sum(0))

# %%
x_atac = fragments.groupby("barcode").size()
x_atac = x_atac.reindex(x_expression.index, fill_value = 0)

# %%
sns.ecdfplot(hue = x_atac > 1, x = x_expression)

# %%
import torch
def split(n, seed=1, train_ratio=0.8):
    generator = np.random.RandomState(seed)
    train_ix = generator.random(n) < train_ratio
    test_ix = ~train_ix

    return train_ix, test_ix


# %%
train_ix, test_ix = split(X.shape[0], seed=1)

# %%
X_train = torch.tensor(X[train_ix]).to(torch.float32)
X_test = torch.tensor(X[test_ix]).to(torch.float32)

# %%
y = x_expression
y_train = torch.tensor(y.iloc[train_ix]).to(torch.float32)
y_test = torch.tensor(y.iloc[test_ix]).to(torch.float32)


# %%
def conv1d_out_dim(L, module):
    return np.floor((L + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) / module.stride[0] + 1)


# %%
class Module(torch.nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 10, 20)
        self.conv2 = torch.nn.Conv1d(10, 1, 20)
        
        out = conv1d_out_dim(window_size, self.conv1)
        out = int(conv1d_out_dim(out, self.conv2))
        self.linear = torch.nn.Linear(out, 100)
        self.final = torch.nn.Linear(100, 1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x.unsqueeze(-2)))
        x = torch.nn.functional.relu(self.conv2(x)).squeeze(-2)
        x = torch.nn.functional.relu(self.linear(x))
        return self.final(x)


# %%
nn = Module(desired_window_size)

# %%
loss = torch.nn.MSELoss()

# %%
trainer = torch.optim.SGD(nn.parameters(), lr=0.03)

# %%
nn.forward(X_train)

# %%

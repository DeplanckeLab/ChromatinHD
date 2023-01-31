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

import tqdm.auto as tqdm

# %%
import peakfreeatac as pfa

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"
# dataset_name = "lymphoma"
# dataset_name = "e18brain"
# dataset_name = "alzheimer"
dataset_name = "brain"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

# %%
software_folder = pfa.get_git_root() / "software"

# %% [markdown]
# ## MACS2

# %%
# load latent space

# %%
# loading
latent_name = "leiden_1"
latent_name = "leiden_0.1"
# latent_name = "celltype"
# latent_name = "overexpression"
folder_data_preproc = folder_data / dataset_name
latent_folder = folder_data_preproc / "latent"
latent = pickle.load((latent_folder / (latent_name + ".pkl")).open("rb"))

n_latent_dimensions = latent.shape[-1]

# %%
peaks_folder = folder_root / "peaks" / dataset_name / ("macs2_" + latent_name)
peaks_folder.mkdir(exist_ok = True, parents = True)

# %%
tmp_fragments_folder = peaks_folder / "tmp"
tmp_fragments_folder.mkdir(exist_ok = True)

# %%
import gzip

# %%
# should take a couple of minutes for pbmc10k
cluster_fragments = [(tmp_fragments_folder / (str(cluster_ix) + ".tsv")).open("w") for cluster_ix in range(n_latent_dimensions)]
cell_to_cluster = dict(zip(latent.index, np.where(latent)[1]))

for l in gzip.GzipFile((folder_data_preproc / "atac_fragments.tsv.gz"), "r"):
    l = l.decode("utf-8")
    if l.startswith("#"):
        continue
    cell = l.split("\t")[3]
    
    if cell in cell_to_cluster:
        cluster_fragments[cell_to_cluster[cell]].write(l)

# %%
# !ls {peaks_folder}/tmp

# %%
for cluster_ix in range(n_latent_dimensions):
    # !echo 'cd {peaks_folder} && macs2 callpeak --nomodel -t {peaks_folder}/tmp/{cluster_ix}.tsv -f BEDPE && cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks_{cluster_ix}.bed'

# %%
with (peaks_folder / "peaks.bed").open("w") as outfile:
    for cluster_ix in range(n_latent_dimensions):
        bed = pd.read_table(peaks_folder / ("peaks_" + str(cluster_ix) + ".bed"), names = ["chr", "start", "end"], usecols = range(3))
        bed["strand"] = cluster_ix
        outfile.write(bed.to_csv(sep = "\t", header = False, index = False))

# %%
# !head -n 20 {(peaks_folder / "peaks.bed")}

# %% [markdown]
# ## Merged

# %%
original_peaks_folder = folder_root / "peaks" / dataset_name / ("macs2_" + latent_name)

# %%
peaks_folder = folder_root / "peaks" / dataset_name / ("macs2_" + latent_name + "_merged")
peaks_folder.mkdir(exist_ok = True, parents = True)

# %%
import pybedtools
pybedtools.BedTool(original_peaks_folder / "peaks.bed").sort().merge().saveas(peaks_folder / "peaks.bed")

# %%
x = pybedtools.BedTool(original_peaks_folder / "peaks.bed").sort().merge()
x.to_dataframe().to_csv(peaks_folder / "peaks.bed", header = False, index = False, sep = "\t")

# %%

# %%

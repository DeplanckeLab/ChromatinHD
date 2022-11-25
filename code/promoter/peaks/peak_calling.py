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

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

# %%
software_folder = pfa.get_git_root() / "software"

# %% [markdown]
# ## Cell ranger

# %%
peaks_folder = folder_root / "peaks" / dataset_name / "cellranger"
peaks_folder.mkdir(exist_ok = True, parents = True)

# %%
# !cp {folder_data_preproc}/peaks.tsv {peaks_folder}/peaks.bed

# %% [markdown]
# ## Genrich

# %%
peaks_folder = folder_root / "peaks" / dataset_name / "genrich"
peaks_folder.mkdir(exist_ok = True, parents = True)

# %% [markdown]
#
# Documentation:
#
# https://informatics.fas.harvard.edu/atac-seq-guidelines.html#peak

# %%
# !wget https://github.com/jsh58/Genrich/archive/refs/tags/v0.6.1.zip -P {software_folder}

# %%
# install
# !echo 'cd {software_folder} && unzip v0.6.1.zip'
# !echo 'cd {software_folder}/Genrich-0.6.1 && make'

# %%
# sort the reads
# !echo 'samtools sort -@ 20 -n {folder_data_preproc}/bam/atac_possorted_bam.bam -o {folder_data_preproc}/bam/atac_readsorted_bam.bam'

# create peaks folder
# !echo 'mkdir -p {peaks_folder}'

# run genrich
# !echo '{software_folder}/Genrich-0.6.1/Genrich -t {folder_data_preproc}/bam/atac_readsorted_bam.bam -j -f {peaks_folder}/log -o {peaks_folder}/peaks.bed -v'

# %%
#(run on updeplasrv7)
# create peaks folder
# !echo 'mkdir -p {peaks_folder}'

# sync from updeplasrv6 to updeplasrv7 
# !echo 'rsync wsaelens@updeplasrv6.epfl.ch:{peaks_folder}/peaks.bed {peaks_folder}/peaks.bed'

# %% [markdown]
# ## MACS2

# %%
peaks_folder = folder_root / "peaks" / dataset_name / "macs2"
peaks_folder.mkdir(exist_ok = True, parents = True)

# %%
# !echo 'ls {peaks_folder}'
# !ls {peaks_folder}

# %%
# !echo 'cd {folder_data_preproc}/bam/{peaks_folder} && macs2 callpeak -t atac_possorted_bam.bam -f BAMPE'

# %%
# !echo 'mkdir -p {peaks_folder}'

# %%
# !echo 'ls {folder_data_preproc}/bam'

# %%
# !echo 'cp {folder_data_preproc}/bam/macs/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# %%
# from updeplasrv7
# !echo 'rsync wsaelens@updeplasrv6.epfl.ch:{peaks_folder}/peaks.bed {peaks_folder}/peaks.bed -v'

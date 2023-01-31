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

# dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "lymphoma"
# dataset_name = "alzheimer"
dataset_name = "brain"

# dataset_name = "FLI1_7"
# dataset_name = "PAX2_7"
# dataset_name = "NHLH1_7"
# dataset_name = "CDX2_7"
# dataset_name = "CDX1_7"
# dataset_name = "MSGN1_7"
# dataset_name = "KLF4_7"
# dataset_name = "KLF5_7"
# dataset_name = "PTF1A_4"

# dataset_name = "morf_20"

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

# %% [markdown]
# A lot of people seem to be happy with this, so why not add it to the "benchmark"

# %% [markdown]
# If you don't have bam file, check out this: https://github.com/jsh58/Genrich/issues/95

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
peaks_folder

# %%
# !echo 'mkdir -p {peaks_folder}'

# %%
# !echo 'ls {folder_data_preproc}'

# %%
# if BAM is available
# # !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/bam/atac_possorted_bam.bam -f BAMPE'

# if BAM is not available
# !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/bam/atac_fragments.tsv.gz -f BEDPE'

# if BAM is not available
# alternative for other datasets
# !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/fragments.tsv.gz -f BEDPE && cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# %%
# !echo 'ls {peaks_folder}'

# %%
# !echo 'cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# %%
# from updeplasrv7
# !echo 'mkdir -p {peaks_folder}'
# !echo 'rsync wsaelens@updeplasrv6.epfl.ch:{peaks_folder}/peaks.bed {peaks_folder}/peaks.bed -v'

# %% [markdown]
# ## MACS2 with different parameters

# %%
peaks_folder = folder_root / "peaks" / dataset_name / "macs2_q0.20"
peaks_folder.mkdir(exist_ok = True, parents = True)

# %%
# !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/atac_fragments.tsv.gz -f BEDPE -q 0.20 && cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# %%
peaks_folder = folder_root / "peaks" / dataset_name / "macs2_q0.50"
peaks_folder.mkdir(exist_ok = True, parents = True)

# %%
# !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/atac_fragments.tsv.gz -f BEDPE -q 0.50 && cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# %% [markdown]
# ## MACS2 improved

# %%
peaks_folder = folder_root / "peaks" / dataset_name / "macs2_improved"
peaks_folder.mkdir(exist_ok = True, parents = True)

# %%
# !echo 'ls {peaks_folder}'
# !ls {peaks_folder}

# %%
peaks_folder

# %%
# if BAM is available
# # !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/bam/atac_possorted_bam.bam -f BAMPE'

# if BAM is not available
# !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/atac_fragments.tsv.gz -f BEDPE --nomodel && cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# if BAM is not available
# alternative for other datasets
# # !echo 'cd {peaks_folder} && macs2 callpeak -t {folder_data_preproc}/fragments.tsv.gz -f BEDPE && cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# %%
# !echo 'ls {peaks_folder}'

# %%
# !echo 'cp {peaks_folder}/NA_peaks.narrowPeak {peaks_folder}/peaks.bed'

# %%
# from updeplasrv7
# !echo 'mkdir -p {peaks_folder}'
# !echo 'rsync wsaelens@updeplasrv6.epfl.ch:{peaks_folder}/peaks.bed {peaks_folder}/peaks.bed -v'

# %%

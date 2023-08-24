# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Preprocess

# %%
from IPython import get_ipython

if get_ipython():
    get_ipython().run_line_magic("load_ext", "autoreload")
    get_ipython().run_line_magic("autoreload", "2")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")

import torch

import pickle

import scanpy as sc

import tqdm.auto as tqdm
import io

# %%
import chromatinhd as chd

# %%
folder_root = chd.get_output()
folder_data = folder_root / "data"

dataset_name = "pbmc10k"
genome = "GRCh38"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

# %%
dataset_folder = chd.get_output() / "datasets" / "pbmc10k"
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))

# %%
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata, path=dataset_folder / "transcriptome")

# %%
import genomepy

# genomepy.install_genome("GRCh38", genomes_dir="/data/genome/")

sizes_file = "/data/genome/GRCh38/GRCh38.fa.sizes"

# %%
regions = chd.data.regions.Regions.from_chromosomes_file(sizes_file, path = dataset_folder / "regions" / "all")

# %%
fragments_file = folder_data_preproc / "fragments.tsv.gz"

# %%
fragments = chd.data.Fragments.from_fragments_tsv(fragments_file, regions = regions, obs = transcriptome.obs, path = dataset_folder / "fragments" / "all")

# %%
fragments.create_regionxcell_indptr()

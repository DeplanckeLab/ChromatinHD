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
main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k"
genome = "GRCh38"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok=True, parents=True)

# %% [markdown]
# ## Tiny dataset

# %%
dataset_folder = chd.get_output() / "datasets" / "pbmc10ktiny"
folder_dataset_publish = chd.get_git_root() / "src" / "chromatinhd" / "data" / "examples" / "pbmc10ktiny"
folder_dataset_publish.mkdir(exist_ok=True, parents=True)

# %%
adata = pickle.load((folder_data_preproc / "adata_annotated.pkl").open("rb"))
genes_oi = adata.var.sort_values("dispersions_norm", ascending=False).index[:50]
adata = adata[:, genes_oi]
adata_tiny = sc.AnnData(X=adata[:, genes_oi].layers["magic"], obs=adata.obs, var=adata.var.loc[genes_oi])

# %%
transcriptome = chd.data.transcriptome.Transcriptome.from_adata(adata, path=dataset_folder / "transcriptome")
adata_tiny.write(folder_dataset_publish / "transcriptome.h5ad", compression="gzip")

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb"))
selected_transcripts = selected_transcripts.loc[genes_oi]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-10000, 10000], path=dataset_folder / "regions" / "10k10k"
)

# %%
import pysam

fragments_tabix = pysam.TabixFile(str(folder_data_preproc / "fragments.tsv.gz"))
coordinates = regions.coordinates

fragments_new = []
for i, (gene, promoter_info) in tqdm.tqdm(enumerate(coordinates.iterrows()), total=coordinates.shape[0]):
    start = max(0, promoter_info["start"])

    fragments_promoter = fragments_tabix.fetch(promoter_info["chrom"], start, promoter_info["end"])
    fragments_new.extend(list(fragments_promoter))

fragments_new = pd.DataFrame(
    [x.split("\t") for x in fragments_new], columns=["chrom", "start", "end", "cell", "nreads"]
)
fragments_new["start"] = fragments_new["start"].astype(int)
fragments_new = fragments_new.sort_values(["chrom", "start", "cell"])

# %%
fragments_new.to_csv(folder_dataset_publish / "fragments.tsv", sep="\t", index=False, header=False)
pysam.tabix_compress(folder_dataset_publish / "fragments.tsv", folder_dataset_publish / "fragments.tsv.gz", force=True)

# %%
# !ls -lh {folder_dataset_publish}
# !tabix -p bed {folder_dataset_publish / "fragments.tsv.gz"}
folder_dataset_publish

# %%
transcriptome.var.index

# %%
coordinates.index

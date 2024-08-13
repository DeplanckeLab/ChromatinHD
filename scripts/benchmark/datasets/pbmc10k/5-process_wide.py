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
dataset_folder = chd.get_output() / "datasets" / "pbmc10k"
dataset_folder.mkdir(exist_ok=True, parents=True)

# %%
fragments_parent = chd.data.Fragments(dataset_folder / "fragments" / "all")

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb"))
selected_transcripts = selected_transcripts.loc[selected_transcripts["chrom"].isin(chd.data.Fragments(dataset_folder / "fragments" / "all").var.index)]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-200000, 200000], dataset_folder / "regions" / "200k200k", max_n_regions = None
)

# %%
fragments = chd.data.fragments.FragmentsView.from_fragments(
    parent=chd.data.Fragments(dataset_folder / "fragments" / "all"),
    regions=regions,
    path=dataset_folder / "fragments" / "200k200k",
    overwrite=True,
)
fragments.create_regionxcell_indptr()

# %%
bc = np.bincount(fragments.regionxcell_fragmentixs[:].astype(np.int64))
plt.hist(bc, bins = 51, range = (0, 50));

# %% [markdown]
# ## Load

# %%
fragments = chd.data.fragments.FragmentsView(dataset_folder / "fragments" / "200k200k")

# %%
transcriptome.var = transcriptome.var.copy()

# %%
gene_ix = transcriptome.gene_ix("CCL4")

# %%
# minibatch = chd.loaders.minibatches.Minibatch(np.arange(0, 1000), np.array([gene_ix]))
minibatch = chd.loaders.minibatches.Minibatch(np.arange(0, 1000), np.arange(100))
loader = chd.loaders.Fragments(fragments=fragments, cellxregion_batch_size=1000*500)

# %%
import pyximport

import sys
if "chromatinhd.loaders.fragments_helpers" in sys.modules:
    del sys.modules["chromatinhd.loaders.fragments_helpers"]

pyximport.install(
    reload_support=True,
    language_level=3,
    setup_args=dict(include_dirs=[np.get_include()]),
    build_in_temp=False,
)
import chromatinhd.loaders.fragments_helpers as fragments_helpers

# %%
plt.hist(loader.load(minibatch).coordinates[:, 0], bins = 50, range = (-200000, 200000));

# %%
import cProfile

stats = cProfile.run("loader.load(minibatch)", "restats")
import pstats

p = pstats.Stats("restats")
p.sort_stats("cumulative").print_stats()

# %%
minibatcher = chd.loaders.minibatches.Minibatcher(np.arange(fragments.n_cells), np.arange(fragments.n_regions), 500, 200)
loaders = chd.loaders.LoaderPool(chd.loaders.Fragments, {"fragments":fragments, "cellxregion_batch_size":minibatcher.cellxregion_batch_size}, n_workers = 15)

# %%
chd.loaders.pool.benchmark(loaders, minibatcher, n=200)

# %%
for minibatch in minibatcher:
    loader.load(minibatch)

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

# %%
selected_transcripts = pickle.load((folder_data_preproc / "selected_transcripts.pkl").open("rb"))
selected_transcripts = selected_transcripts.loc[selected_transcripts["chrom"].isin(chd.data.Fragments(dataset_folder / "fragments" / "all").var.index)]
regions = chd.data.regions.Regions.from_transcripts(
    selected_transcripts, [-200000, 200000], dataset_folder / "regions" / "200k200k", max_n_regions = None
)
# regions.coordinates = regions.coordinates.loc[regions.coordinates["chrom"].isin(chd.data.Fragments(dataset_folder / "fragments" / "all").regions.coordinates.index)]

# %%
fragments = chd.data.fragments.FragmentsView.from_fragments(
    parent=chd.data.Fragments(dataset_folder / "fragments" / "all"),
    regions=regions,
    path=dataset_folder / "fragments" / "200k200k",
    overwrite=True,
)
fragments.create_regionxcell_indptr()

# %%
bc = np.bincount(fragments.fragmentixs[:].astype(np.int64))
plt.hist(bc, bins = 51, range = (0, 50));

# %% [markdown]
# ## Load

# %%
fragments = chd.data.fragments.FragmentsView(dataset_folder / "fragments" / "200k200k")

# %%
fragments = chd.data.fragments.Fragments(dataset_folder / "fragments" / "all")
fragments.create_regionxcell_indptr()

# %%
# !ls {dataset_folder / "fragments" / "200k200k"}
# !du -sh {dataset_folder / "fragments" / "all"}
# !du -sh {dataset_folder / "fragments" / "200k200k"}

# %%
fragments.estimate_fragment_per_cellxregion()

# %%
# minibatch = chd.models.pred.loader.Minibatch(np.arange(0, 1000), np.arange(0, fragments.n_regions))
minibatch = chd.models.pred.loader.Minibatch(np.arange(0, 1000), [5])
# minibatch = chd.models.pred.loader.Minibatch(np.arange(0, 1000), np.array([5000]))
loader = chd.models.pred.loader.Fragments(fragments=fragments, cellxregion_batch_size=1000*500)

# %%
import pyximport

import sys
if "chromatinhd.loaders.fragments_helpers" in sys.modules:
    del sys.modules["chromatinhd.loaders.fragments_helpers"]

pyximport.install(
    reload_support=True,
    language_level=3,
    setup_args=dict(include_dirs=[np.get_include()]),
    build_in_temp=False,
)
import chromatinhd.loaders.fragments_helpers

# %%
plt.hist(loader.load(minibatch).coordinates[:, 0], bins = 100, range = (-200000, 200000));

# %%
import cProfile

stats = cProfile.run("loader.load(minibatch)", "restats")
import pstats

p = pstats.Stats("restats")
p.sort_stats("cumulative").print_stats()

# %%
minibatcher = chd.models.pred.loader.Minibatcher(np.arange(fragments.n_cells), np.arange(fragments.n_regions), 500, 200)
loaders = chd.loaders.LoaderPool(chd.models.pred.loader.Fragments, {"fragments":fragments, "cellxregion_batch_size":minibatcher.cellxregion_batch_size}, n_workers = 15)

# %%
chd.loaders.pool.benchmark(loaders, minibatcher, n=200)

# %%
for minibatch in minibatcher:
    loader.load(minibatch)

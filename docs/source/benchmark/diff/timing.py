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

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
# %config InlineBackend.figure_format='retina'

import tqdm.auto as tqdm
import torch
import os
import time

# %%
import chromatinhd as chd

chd.set_default_device("cuda:1")

# %%
dataset_folder_original = chd.get_output() / "datasets" / "pbmc10k"
transcriptome_original = chd.data.Transcriptome(dataset_folder_original / "transcriptome")
fragments_original = chd.data.Fragments(dataset_folder_original / "fragments" / "10k10k")

# %%
genes_oi = transcriptome_original.var.sort_values("dispersions_norm", ascending=False).head(30).index
regions = fragments_original.regions.filter_genes(genes_oi)
fragments = fragments_original.filter_genes(regions)
fragments.create_cellxgene_indptr()
transcriptome = transcriptome_original.filter_genes(regions.coordinates.index)

# %%
folds = chd.data.folds.Folds()
folds.sample_cells(fragments, 5)

# %%
clustering = chd.data.Clustering.from_labels(transcriptome_original.obs["celltype"])

# %%
fold = folds[0]

# %%
models = {}
scores = []

# %%
import logging

logger = chd.models.diff.trainer.trainer.logger
logger.setLevel(logging.DEBUG)
logger.handlers = []
# logger.handlers = [logging.StreamHandler()]

# %%
devices = pd.DataFrame({"device": ["cuda:0", "cuda:1", "cpu"]}).set_index("device")
for device in devices.index:
    if device != "cpu":
        devices.loc[device, "label"] = torch.cuda.get_device_properties(device).name
    else:
        devices.loc[device, "label"] = os.popen("lscpu").read().split("\n")[13].split(": ")[-1].lstrip()

# %%
scores = pd.DataFrame({"device": devices.index}).set_index("device")

# %%
for device in devices.index:
    start = time.time()
    model = chd.models.diff.model.cutnf.Model(
        fragments,
        clustering,
    )
    model.train_model(fragments, clustering, fold, n_epochs=10, device=device)
    models[device] = model
    end = time.time()
    scores.loc[device, "train"] = end - start

# %%
for device in devices.index:
    genepositional = chd.models.diff.interpret.genepositional.GenePositional(
        path=chd.get_output() / "interpret" / "genepositional"
    )

    start = time.time()
    genepositional.score(fragments, clustering, [models[device]], force=True, device=device)
    end = time.time()
    scores.loc[device, "inference"] = end - start

# %%
fig = polyptich.grid.Figure(polyptich.grid.Wrap(padding_width=0.1))
height = len(scores) * 0.2

plotdata = scores.copy().loc[devices.index]

panel, ax = fig.main.add(polyptich.grid.Panel((1, height)))
ax.barh(plotdata.index, plotdata["train"])
ax.set_yticks(np.arange(len(devices)))
ax.set_yticklabels(devices.label)
ax.axvline(0, color="black", linestyle="--", lw=1)
ax.set_title("Training")
ax.set_xlabel("seconds")

panel, ax = fig.main.add(polyptich.grid.Panel((1, height)))
ax.barh(plotdata.index, plotdata["inference"])
ax.axvline(0, color="black", linestyle="--", lw=1)
ax.set_title("Inference")
ax.set_yticks([])
fig.plot()

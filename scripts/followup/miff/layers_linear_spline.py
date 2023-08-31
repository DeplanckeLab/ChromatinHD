# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3
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

# %%
import torch
import chromatinhd as chd
chd.set_default_device("cuda:0")

import tempfile
import pathlib
import pickle

# %%
import quadratic

# %%
width = 1024
positions = torch.tensor([100, 500, 1000])
nbins = [8, 8, 8]

unnormalized_heights_bins = [
    torch.tensor([[1] * 8]*3, dtype = torch.float),
    torch.tensor([[1] * 8]*3, dtype = torch.float),
    torch.tensor([[1] * 8]*3, dtype = torch.float),
    # torch.tensor([[1] * 2]*3, dtype = torch.float),
]
unnormalized_heights_bins[1][1, 3] = 10

# %%
unnormalized_heights_all = []
cur_total_n = 1
for n in nbins:
    cur_total_n *= n
    unnormalized_heights_all.append(torch.zeros(cur_total_n).reshape(-1, n))
unnormalized_heights_all[0][0, 3] = -1
unnormalized_heights_all[1][2, 2:4] = 1
unnormalized_heights_all[2][0, 2:4] = 1

# %%
import math

def transform_linear_spline(positions, n, width, unnormalized_heights):
    binsize = width//n

    normalized_heights = torch.nn.functional.log_softmax(unnormalized_heights, -1)
    if normalized_heights.ndim == positions.ndim:
        normalized_heights = normalized_heights.unsqueeze(0)

    binixs = torch.div(positions, binsize, rounding_mode = "trunc")

    logprob = torch.gather(normalized_heights, 1, binixs.unsqueeze(1)).squeeze(1)
    
    positions = positions - binixs * binsize
    width = binsize

    return logprob, positions, width

def calculate_logprob(positions, nbins, width, unnormalized_heights_bins):
    assert len(nbins) == len(unnormalized_heights_bins)

    curpositions = positions
    curwidth = width
    logprob = torch.zeros_like(positions, dtype = torch.float)
    for i, n in enumerate(nbins):
        assert (curwidth % n) == 0
        logprob_layer, curpositions, curwidth = transform_linear_spline(curpositions, n, curwidth, unnormalized_heights_bins[i])
        logprob += logprob_layer
    logprob = logprob - math.log(curwidth)
    return logprob


# %%
x = torch.arange(width)

# %%
totalnbins = np.cumprod(nbins)
totalbinwidths = torch.tensor(width//totalnbins)
totalbinixs = torch.div(x[:, None], totalbinwidths, rounding_mode="floor")
totalbinsectors = torch.div(totalbinixs, torch.tensor(nbins)[None, :], rounding_mode="trunc")
unnormalized_heights_bins = [unnormalized_heights_all[i][totalbinsector] for i, totalbinsector in enumerate(totalbinsectors.numpy().T)]

# %%
torch.tensor(width//np.array(nbins))

# %%
totalbinwidthsa

# %%
logprob = calculate_logprob(x, nbins, width, unnormalized_heights_bins)
fig, ax = plt.subplots()
ax.plot(x, torch.exp(logprob))

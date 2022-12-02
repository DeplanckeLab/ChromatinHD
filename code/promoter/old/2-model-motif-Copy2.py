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

# %% [markdown]
# # Model promoters with motifs

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc

import pathlib

#export LD_LIBRARY_PATH=/data/peak_free_atac/software/peak_free_atac/lib
import torch
import torch_sparse

import tqdm.auto as tqdm

# %%
import peakfreeatac as pfa
import peakfreeatac.fragments
import peakfreeatac.transcriptome

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

# %%
# promoter_name, window = "4k2k", (2000, 4000)
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)
window_width = window[1] - window[0]

# %%
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")
fragments = peakfreeatac.fragments.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
motifscores = pickle.load(open(folder_data_preproc / "motifscores.pkl", "rb"))

# %%
n_fragments = fragments.coordinates.shape[0]

# %%
import timeit

# %% [markdown]
# What we want is
# - Know which motifs are close to a cut sites. Either precalculate or extract on-the-fly as fast as possible
# - Extract all information of a motif in relation to the fragment
#   - Which motif
#   - Whether it is relative to the left or right cut site
#   - Its distance to the cut site
#   - Its score
# - We want to be able to index extremely fast on the fragments, ideally microseconds

# %% [markdown]
# ## Theoretical memory analysis

# %% [markdown]
# Assume for each fragment we need to store 600 loci (=motif binding sites with a particular distance, l/r and score)

# %%
n_fragmentxlocus = (n_fragments * 600)

# %% [markdown]
# For each locus, we need to store the distance, motif_ix, left/right and score. Let's assume it's all 32 bit. The index pointers for fragments are meaningless

# %%
n_gigs = (
    (n_fragmentxlocus * 32) +
    (n_fragmentxlocus * 32) +
    (n_fragmentxlocus * 32) +
    (n_fragmentxlocus * 32)
) / 8 / 1024 / 1024 / 1024
n_gigs

# %% [markdown]
# 😬

# %% [markdown]
# This means we will have to calculate everything on-the-fly

# %% [markdown]
# Can we store the motifscores on GPU? Assume we have 400 motifs to score, we have to store both motif_ix and value

# %%
n_data_per_motif = len(motifscores.data) / motifscores.shape[1]
n_data_per_motif

# %%
n_data = n_data_per_motif * 400

# %%
n_gigs = (
    (n_data * 32) +
    (n_data * 32)
) / 8 / 1024 / 1024 / 1024
n_gigs

# %% [markdown]
# That actually looks doable, although more genes or motifs would stretch the limits. Still, this means we can calculate fragment scores on-the-fly on the GPU itself. Not that this is likely, given that it might also be fast enough on CPU and could be done in parallel while the CPU is working

# %% [markdown]
# Could we store the scores of the whole genome (on GPU)?

# %%
n_motifs_per_base = len(motifscores.data) / motifscores.shape[1] / motifscores.shape[0]
n_motifs_per_base

# %%
n_motifs_total = n_motifs_per_base * 2*(10**9) * 400

# %%
n_motifs_total

# %%
n_gigs = (
    (n_motifs_total * 32) +
    (n_motifs_total * 32)
) / 8 / 1024 / 1024 / 1024
n_gigs


# %% [markdown]
# Probably not in VRAM, but yes on normal memory 😉

# %% [markdown]
# ## Single slice

# %% [markdown]
# ### Slicing CSR

# %% [markdown]
# <img src="https://matteding.github.io/images/csr.gif" width="600" />

# %% [markdown]
# ### Indexing the `motifscores` CSR

# %% [markdown]
# Our `motifscores` is already CSR, where the idptr points to an individual locus around a promoter [position x promoter] and indices point to which motif_ix. Positions are based on the `window`

# %% [markdown]
# For each fragment, we will take a slice, "clipped" according to the promoter (i.e. `window`)

# %% [markdown]
# What is the fastest way to do this?

# %%
def time_indexing(x, idx, n_idx = 1, n = 1000):
    time = timeit.Timer("x[idx]", globals = locals()).timeit(n)/n
    return f"{(time * n_fragments / n_idx)/60:.1f} minutes for all fragments"


# %% [markdown]
# Using an arange

# %%
time_indexing(motifscores, idx = np.arange(0, 1000))

# %% [markdown]
# Using a slice

# %%
time_indexing(motifscores, idx = slice(0, 150))


# %% [markdown]
# This is pretty fast, because slicing can be done very efficiently: https://github.com/scipy/scipy/blob/de80faf9d3480b9dbb9b888568b64499e0e70c19/scipy/sparse/_compressed.py#L714

# %% [markdown]
# Based on this, we can look how a manual slice would work, without creating a new scipy sparse csr object

# %%
class X():
    def __init__(self, csr):
        self.csr = csr
    
    def __getitem__(self, k):
        start = motifscores.indptr[k.start]
        end = motifscores.indptr[k.stop] + 1
        return motifscores.indices[start:end], motifscores.data[start:end]
motifscores_fast = X(motifscores)

# %%
time_indexing(motifscores_fast, idx = slice(0, 150))


# %% [markdown]
# This is the fasted, because all we really have to do is some simple slicing

# %% [markdown]
# Is the speedup because of using a slice to select from indices/data, or because we use a slice as input? Because if we want to be able to select multiple slices (for multiple fragments), we may have to go replace the `start:end` with some index array

# %%
class X():
    def __init__(self, csr):
        self.csr = csr
    
    def __getitem__(self, k):
        idx = np.arange(motifscores.indptr[k.start], motifscores.indptr[k.stop] + 1)
        return motifscores.indices[idx], motifscores.data[idx]
motifscores_fast = X(motifscores)

# %%
time_indexing(motifscores_fast, idx = slice(0, 150))

# %% [markdown]
# Ok, so there is a considerable slowdown when using indices

# %% [markdown]
# Any other approach is much slower

# %%
# slice using random access
time_indexing(motifscores, idx = np.random.choice(np.arange(0, 1000), 1000), n = 10)

# slice on csc matrix
time_indexing(motifscores.tocsc(), idx = slice(0, 150), n = 10)

# cannot slice on coo
# time_indexing(motifscores.tocoo(), idx = slice(0, 150))
'∞ minutes for all fragments'

# %% [markdown]
# Slicing the CSR is clearly and intuitively the best. However, we need to work manually with the index pointers, indices and data objects.

# %% [markdown]
# ## Multiple slices

# %% [markdown]
# Let's move towards multiple fragments

# %% [markdown]
# First create the ground truth using a simple loop

# %%
idx = [(0, 301)] * 100

# %%
indptr_gs = [0]
indices_gs = []
data_gs = []
distance_gs = []

for i in idx:
    subset = motifscores[slice(i[0], i[1])]
    indptr_gs.append(len(subset.data) + len(data_gs))
    indices_gs.extend(subset.indices)
    data_gs.extend(subset.data)
    distance_gs.extend(motifscores[slice(i[0], i[1])].tocoo().row - i[0])
indptr_gs = np.array(indptr_gs)
indices_gs = np.array(indices_gs)
data_gs = np.array(data_gs)
distance_gs = np.array(distance_gs)


# %%
def test_indexing(indptr, indices, data, distance = None):
    if torch.is_tensor(indptr):
        indptr = indptr.to("cpu")
    if torch.is_tensor(indices):
        indices = indices.to("cpu")
    if torch.is_tensor(data):
        data = data.to("cpu")
    assert np.array_equal(indptr, indptr_gs)
    assert np.array_equal(indices, indices_gs), (len(indices), len(indices_gs), indices.dtype, indices_gs.dtype)
    assert np.array_equal(data, data_gs)
    
    if distance is not None:
        if torch.is_tensor(distance):
            distance = distance.to("cpu")
        assert np.array_equal(distance, distance_gs)
    
    return True


# %% [markdown]
# First using multiple slices

# %%
class X():
    csr = None
    def __init__(self, csr):
        self.csr = csr
    
    def __getitem__(self, idx):
        return (
            np.cumsum([0] + [self.csr.indptr[end] - self.csr.indptr[start] for start, end in idx]),
            np.hstack([self.csr.indices[self.csr.indptr[start]:(self.csr.indptr[end])] for start, end in idx]),
            np.hstack([self.csr.data[self.csr.indptr[start]:(self.csr.indptr[end])] for start, end in idx]),
        )
motifscores_fast = X(motifscores)

# %%
idx = [(0, 301)] * 100
time_indexing(motifscores_fast, idx = idx, n_idx=len(idx))

# %%
test_indexing(*motifscores_fast[idx])


# %% [markdown]
# Let's try to convert the slices to indices

# %%
class X():
    def __init__(self, csr):
        self.csr = csr
    
    def __getitem__(self, k):
        idx = np.hstack([np.arange(self.csr.indptr[start], (self.csr.indptr[end])) for start, end in k])
        return (
            np.cumsum([0] + [self.csr.indptr[end] - self.csr.indptr[start] for start, end in k]),
            self.csr.indices[idx],
            self.csr.data[idx]
        )
motifscores_fast = X(motifscores)

# %%
idx = [(0, 301)] * 100
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx))

# %%
test_indexing(*motifscores_fast[idx])

# %% [markdown]
# Both are equally fast

# %% [markdown]
# Try avoiding the hstack  
# We could try something from here: https://stackoverflow.com/questions/367565/how-do-i-build-a-numpy-array-from-a-generator  
# But I guess this is the fastest: https://stackoverflow.com/questions/55916932/multiple-ranges-np-arange

# %%
from numba import njit

@njit()
def n_ranges_nb(t1, t2):
    a = np.arange(np.max(t2))
    n = (t2 - t1).sum()
    out = np.empty(n, dtype = np.int32)
    l, l_old = 0, 0
    for i,j in zip(t1, t2):
        l += j-i
        out[l_old:l] = a[i:j]
        l_old = l
    return out

t1 = np.array([0,13,22])
t2 = np.array([4,14,25])

n_ranges_nb(t1, t2+1)

# %%
# %%timeit -n 100
t1 = np.arange(10, 100000, step = 400)
t2 = np.arange(50, 100000, step = 400)
n_ranges_nb(t1, t2)


# %%
class X():
    def __init__(self, csr):
        self.csr = csr
    
    def __getitem__(self, k):
        new_indptr = np.cumsum([0] + [self.csr.indptr[end] - self.csr.indptr[start] for start, end in k])
        idx = n_ranges_nb(motifscores.indptr[k[:, 0]], motifscores.indptr[k[:, 1]])
        return (
            new_indptr,
            self.csr.indices[idx],
            self.csr.data[idx]
        )
motifscores_fast = X(motifscores)

# %% tags=[]
idx = np.array([(0, 301)] * 100)
time_indexing(motifscores_fast, idx = idx, n_idx = idx.shape[0]) # first run for jit compile
time_indexing(motifscores_fast, idx = idx, n_idx = idx.shape[0])

# %%
test_indexing(*motifscores_fast[idx])


# %% [markdown]
# 😎

# %% [markdown]
# Let's try to optimize the `new_indptr`, because now it's a major time sink (60% of total time!)

# %%
class X():
    def __init__(self, csr):
        self.csr = csr
    
    def __getitem__(self, k):
        new_indptr = np.cumsum(np.insert(motifscores.indptr[k[:, 1]] - motifscores.indptr[k[:, 0]], 0, 0))
        idx = n_ranges_nb(motifscores.indptr[k[:, 0]], motifscores.indptr[k[:, 1]])
        return (
            new_indptr,
            self.csr.indices[idx],
            self.csr.data[idx]
        )
motifscores_fast = X(motifscores)

# %%
idx = np.array([(0, 301)] * 100)
time_indexing(motifscores_fast, idx = idx, n_idx = idx.shape[0])

# %%
test_indexing(*motifscores_fast[idx])

# %% [markdown]
# 😎😎

# %% [markdown]
# Test whether this is still fast when taking more fragments

# %%
idx = np.array([(0, 301)] * 100000)
time_indexing(motifscores_fast, idx = idx, n_idx = idx.shape[0], n = 10)

# %% [markdown]
# Yup!

# %% [markdown]
# Test transporting to torch/cuda on the fly

# %%
indptr, indices, data = motifscores_fast[idx]


# %%
class X():
    def __init__(self, csr):
        self.csr = csr
    
    def __getitem__(self, k):
        new_indptr = np.cumsum(np.insert(motifscores.indptr[k[:, 1]] - motifscores.indptr[k[:, 0]], 0, 0))
        idx = n_ranges_nb(motifscores.indptr[k[:, 0]], motifscores.indptr[k[:, 1]])
        return (
            torch.tensor(new_indptr, device = "cuda"),
            torch.tensor(self.csr.indices[idx], device = "cuda"),
            torch.tensor(self.csr.data[idx], device = "cuda")
        )
motifscores_fast = X(motifscores)

# %%
idx = np.array([(0, 301)] * 100000)
torch.tensor(1., device = "cuda") # init cuda
time_indexing(motifscores_fast, idx = idx, n_idx = idx.shape[0], n = 10)


# %% [markdown]
# ### Torch version

# %% [markdown]
# Let's try to create a torch version

# %%
class X():
    def __init__(self, csr, device = "cpu"):
        self.indices = torch.tensor(csr.indices).to(device)
        self.indptr = torch.tensor(csr.indptr).to(device)
        self.data = torch.tensor(csr.data).to(device)
    
    def __getitem__(self, k):
        new_indptr = torch.nn.functional.pad(torch.cumsum(self.indptr[k[:, 1]] - self.indptr[k[:, 0]], 0), (1, 0))
        idx = n_ranges_nb(self.indptr[k[:, 0]].cpu().numpy(), self.indptr[k[:, 1]].cpu().numpy())
        return (
            new_indptr,
            self.indices[idx],
            self.data[idx]
        )


# %%
motifscores_fast = X(motifscores)
idx = np.array([(0, 301)] * 100)
time_indexing(motifscores_fast, idx = idx, n_idx = idx.shape[0], n = 10)

# %%
test_indexing(*motifscores_fast[idx])

# %%
motifscores_fast = X(motifscores, "cuda:1")
# idx = torch.tensor([(0, 301)] * 100, device = "cuda:1")
idx = np.array([(0, 301)] * 100)
time_indexing(motifscores_fast, idx = idx, n_idx = idx.shape[0], n = 10)

# %%
test_indexing(*motifscores_fast[idx])


# %% [markdown]
# For some reason, both n_ranges_torch is pretty slow... especially on the gpu

# %% [markdown]
# So let's do it on cpu/numpy instead

# %%
class X():
    indptr:torch.tensor
    
    def __init__(self, csr, device = "cpu"):
        self.indices = torch.tensor(csr.indices, dtype = torch.long).to(device)
        # self.indptr = csr.indptr
        self.indptr = torch.tensor(csr.indptr, dtype = torch.long).to(device)
        self.data = torch.tensor(csr.data).to(device)
    
    def __getitem__(self, k):
        new_indptr = torch.nn.functional.pad(torch.cumsum(self.indptr[k[:, 1]] - self.indptr[k[:, 0]], 0), (1, 0))
        idx = n_ranges_nb(self.indptr[k[:, 0]].cpu().numpy(), self.indptr[k[:, 1]].cpu().numpy())
        
        # new_indptr = np.pad(np.cumsum(self.indptr[k[:, 1]] - self.indptr[k[:, 0]], 0), (1, 0))
        # idx = n_ranges_nb(self.indptr[k[:, 0]], self.indptr[k[:, 1]])
        
        return (
            new_indptr,
            self.indices[idx],
            self.data[idx]
        )


# %%
motifscores_fast = X(motifscores)
idx = np.array([(0, 301)] * 100)
time_indexing(motifscores_fast, idx = idx, n_idx = idx.shape[0], n = 10)

# %%
test_indexing(*motifscores_fast[idx])

# %% [markdown]
# It also works with larger indices

# %%
motifscores_fast = X(motifscores)
idx = np.array([(0, 301)] * 10000)
time_indexing(motifscores_fast, idx = idx, n_idx = idx.shape[0], n = 10)

# %% [markdown]
# It's the indexing, e.g. `self.data[idx]`, that is slowing things down by a lot. I think particularly if memory access is overloaded (which it is if Victor is running things :-))
#
# We don't have this problem on GPU!

# %%
motifscores_fast = X(motifscores, "cuda:1")
idx = np.array([(0, 301)] * 1000)
time_indexing(motifscores_fast, idx = idx, n_idx = idx.shape[0], n = 10)

# %%
indptr, indices, data = motifscores_fast[idx]

# %%
import torch_scatter

# %%
torch_scatter.segment_mean_csr(indices, indptr)[:10]


# %% [markdown]
# Alright :-)

# %% [markdown]
# ### Extracting the relative position

# %% [markdown]
# Now extract for each slice it's position within the slice as well.

# %%
class X():
    def __init__(self, csr, device = "cpu", cutwindow_width = 300):
        self.indices = torch.tensor(csr.indices, dtype = torch.long).to(device)
        self.indptr = torch.tensor(csr.indptr, dtype = torch.long).to(device)
        self.data = torch.tensor(csr.data).to(device)
        
        self.cutindices = np.arange(cutwindow_width+1)
    
    def __getitem__(self, k):
        new_indptr = torch.nn.functional.pad(torch.cumsum(self.indptr[k[:, 1]] - self.indptr[k[:, 0]], 0), (1, 0))
        idx = n_ranges_nb(self.indptr[k[:, 0]].cpu().numpy(), self.indptr[k[:, 1]].cpu().numpy())
        diff = torch.tensor(np.hstack([np.repeat(self.cutindices, self.indptr[(start+1):(end+1)] - self.indptr[start:end]) for start, end in k]))
        return (
            new_indptr,
            self.indices[idx],
            self.data[idx],
            diff
        )


# %%
motifscores_fast = X(motifscores)
idx = np.array([(0, 301)] * 100)
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx), n = 10)

# %%
test_indexing(*motifscores_fast[idx])


# %% [markdown]
# Woopsie

# %% [markdown]
# We can fix this by simply storing a combined coo and csr tensor. We use the row indices (as in coo) to directly return the row index. However, we still don't know ther relative distance of each motif to the input key

# %%
def expandptr(indptr, n = None):
    """
    store relative row indices, used to quickly return the row index of each hit (along with the column index)
    this expand something like [0, 0, 0, 1, 0, 0, 2, 0] to [3, 6, 6]
    original code from https://github.com/scipy/scipy/blob/de80faf9d3480b9dbb9b888568b64499e0e70c19/scipy/sparse/_compressed.py#L1034
    """
    
    from scipy.sparse import _sparsetools
    if n is None:
        n = indptr[-1]
    major_indices = np.empty(n, dtype=int)
    _sparsetools.expandptr(len(indptr)-1, indptr, major_indices)
    return major_indices
indptr = np.cumsum(np.array([0, 0, 0, 1, 0, 0, 2, 0]))
assert np.array_equal(expandptr(indptr), np.array([2, 5, 5]))

# %%
indptr = torch.cumsum(torch.tensor([0, 0, 0, 1, 0, 0, 2, 0]), 0)
assert np.array_equal(torch.ops.torch_sparse.ptr2ind(indptr, indptr[-1]), torch.tensor([2, 5, 5]))

# %%
# %%timeit -n 10000 -r 1
expandptr(indptr)

# %%
# %%timeit -n 10000 -r 1
torch.ops.torch_sparse.ptr2ind(indptr, indptr[-1]), torch.tensor([2, 5, 5])


# %%
class X():
    indptr:torch.tensor
    
    def __init__(self, csr, device = "cpu"):
        self.indices = torch.tensor(csr.indices, dtype = torch.long).to(device)
        self.indptr = torch.tensor(csr.indptr, dtype = torch.long).to(device)
        self.data = torch.tensor(csr.data).to(device)

        # stores indptr but in coo format
        self.row_indices = expandptr(self.indptr)
    
    def __getitem__(self, k):
        new_indptr = torch.nn.functional.pad(torch.cumsum(self.indptr[k[:, 1]] - self.indptr[k[:, 0]], 0), (1, 0))
        idx = n_ranges_nb(self.indptr[k[:, 0]].cpu().numpy(), self.indptr[k[:, 1]].cpu().numpy())
        
        return (
            new_indptr,
            self.indices[idx],
            self.data[idx],
            self.row_indices[idx]
        )


# %%
motifscores_fast = X(motifscores)
idx = np.array([(0, 301)] * 100)
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx), n = 10)


# %% [markdown]
# For that we need to extract for each output motif the window left position of the window, and substract from that the actual position. For that we need to convert the output indptr to coo as well

# %%
class X():
    def __init__(self, csr, device = "cpu"):
        self.indices = torch.tensor(csr.indices, dtype = torch.long).to(device)
        self.indptr = torch.tensor(csr.indptr, dtype = torch.long).to(device)
        self.data = torch.tensor(csr.data).to(device)
        
        # stores indptr but in coo format
        self.row_indices = torch.tensor(expandptr(self.indptr.cpu().numpy()), device = device)
    
    def __getitem__(self, k):
        new_indptr = torch.nn.functional.pad(torch.cumsum(self.indptr[k[:, 1]] - self.indptr[k[:, 0]], 0), (1, 0))
        new_row_indices = torch.ops.torch_sparse.ptr2ind(new_indptr, new_indptr[-1])
        # new_row_indices = expandptr(new_indptr.cpu().numpy())
        
        idx = n_ranges_nb(self.indptr[k[:, 0]].cpu().numpy(), self.indptr[k[:, 1]].cpu().numpy())
        
        return (
            new_indptr,
            self.indices[idx],
            self.data[idx],
            self.row_indices[idx] - k[new_row_indices, 0]
        )


# %%
motifscores_fast = X(motifscores)
idx = np.array([(0, 301)] * 100)
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx), n = 10)

# %%
test_indexing(*motifscores_fast[idx])

# %% [markdown]
# Does it work fast on cuda?

# %%
motifscores_fast = X(motifscores, "cuda:0")
# idx = np.array([(0, 301)] * 1000000)
idx = torch.tensor(np.array([(0, 301)] * 1000000), device = motifscores_fast.indices.device)
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx), n = 1)


# %% [markdown]
# How fast is the numpy/CPU version, with a subsequent transfer to torch?

# %% tags=[]
class X():
    def __init__(self, csr, device = "cpu"):
        self.indices = csr.indices
        self.indptr = csr.indptr
        self.data = csr.data
        
        # stores the row indices in coo format
        self.row_indices = expandptr(self.indptr)
    
    def __getitem__(self, k):
        new_indptr = np.pad(np.cumsum(self.indptr[k[:, 1]] - self.indptr[k[:, 0]], 0), (1, 0))
        new_row_indices = expandptr(new_indptr, new_indptr[-1])
        
        idx = n_ranges_nb(self.indptr[k[:, 0]], self.indptr[k[:, 1]])
        
        return (
            torch.from_numpy(new_indptr).cuda(),
            torch.from_numpy(self.indices[idx]).cuda(),
            torch.from_numpy(self.data[idx]).cuda(),
            torch.from_numpy(self.row_indices[idx] - k[new_row_indices, 0]).cuda()
        )


# %%
motifscores_fast = X(motifscores)
idx = np.array([(0, 301)] * 100000)
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx), n = 2)

# %% [markdown]
# ### Using pinned memory

# %% [markdown]
# We may want to (1) avoid creating new tensors, which requires paging/memory allocaton and (2) use an (existing) pinned tensor which would speed up transfer to GPU.
#
# ~Both ideas require a tensor of fixed size (I think), which may sound problematic because we don't know the size of the tensors initially.
# However, we can just cut away the end?~
#
# It's not so well documented it seems, but your initial pinned tensor needs to be large enough. In that case, resizing the tensor will work. Even if you enlarge it

# %%
x = torch.ones((100, )).pin_memory()

# works
x.resize_((10, ))

# does not work
try:
    x.resize_((1000, ))
except RuntimeError:
    print("it indeed errors")
    
# still works
x.resize_((50, ))


# %%
class X():
    def __init__(self, csr):
        self.indices = csr.indices
        self.indptr = csr.indptr
        self.data = csr.data
        
        # stores the row indices in coo format
        self.row_indices = expandptr(self.indptr)
        
        self.max_len = 37000000
        self.data_pinned = torch.empty((self.max_len, ), dtype = torch.float32).pin_memory()
        self.indices_pinned = torch.empty((self.max_len, ), dtype = torch.int32).pin_memory()
        self.distances_pinned = torch.empty((self.max_len, ), dtype = torch.float32).pin_memory()
        
    def __getitem__(self, k):
        new_indptr = np.pad(np.cumsum(self.indptr[k[:, 1]] - self.indptr[k[:, 0]], 0), (1, 0))
        new_row_indices = expandptr(new_indptr, new_indptr[-1])
        
        idx = n_ranges_nb(self.indptr[k[:, 0]], self.indptr[k[:, 1]])
        
        self.indices_pinned.copy_(torch.from_numpy(self.indices[idx]))
        self.data_pinned.copy_(torch.from_numpy(self.data[idx]))
        self.distances_pinned.copy_(torch.from_numpy(self.row_indices[idx] - k[new_row_indices, 0]))
        
        return (
            torch.tensor(new_indptr).cuda(),
            self.indices_pinned.cuda(),
            self.data_pinned.cuda(),
            self.distances_pinned.cuda()
        )


# %%
motifscores_fast = X(motifscores)

# %%
idx = np.array([(0, 301)] * 100000)
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx), n = 2)


# %% [markdown]
# We can speed this up a bit by not creating an intermediate array/tensor (`self.indices[idx]`) but just using index_select and filling the existing tensor

# %%
class X():
    def __init__(self, csr):
        self.indices = torch.from_numpy(csr.indices).to(torch.int64)
        self.indptr = csr.indptr
        self.data = torch.from_numpy(csr.data).to(torch.float64)
        
        # stores the row indices in coo format
        self.row_indices = torch.tensor(expandptr(self.indptr), dtype = torch.int64)
        
        self.max_len = 37000000
        self.indices_pinned = torch.empty((self.max_len, ), dtype = self.indices.dtype).pin_memory()
        self.data_pinned = torch.empty((self.max_len, ), dtype = self.data.dtype).pin_memory()
        self.distances_pinned = torch.empty((self.max_len, ), dtype = torch.int64).pin_memory()
        
    def __getitem__(self, k):
        new_indptr = np.pad(np.cumsum(self.indptr[k[:, 1]] - self.indptr[k[:, 0]], 0), (1, 0))
        new_row_indices = expandptr(new_indptr, new_indptr[-1])
        
        idx = torch.from_numpy(n_ranges_nb(self.indptr[k[:, 0]], self.indptr[k[:, 1]])).to(torch.int64)
        
        self.indices_pinned.resize_(len(idx))
        self.data_pinned.resize_(len(idx))
        self.distances_pinned.resize_(len(idx))
        
        torch.index_select(self.indices, 0, idx, out = self.indices_pinned)
        torch.index_select(self.data, 0, idx, out = self.data_pinned)
        torch.index_select(self.row_indices, 0, idx, out = self.distances_pinned)
        self.distances_pinned.sub_(torch.from_numpy(k[new_row_indices, 0]))
        
        return (
            torch.tensor(new_indptr).cuda(),
            self.indices_pinned.cuda(),
            self.data_pinned.cuda(),
            self.distances_pinned.cuda()
        )


# %%
motifscores_fast = X(motifscores) # setup can take a bit longer here
idx = np.array([(0, 301)] * 10000)
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx), n = 2)


# %% [markdown]
# We could also resize the pinned memory. Not sure whether this has a cost?

# %%
class X():
    def __init__(self, csr):
        self.indices = torch.from_numpy(csr.indices).to(torch.int64)
        self.indptr = csr.indptr
        self.data = torch.from_numpy(csr.data).to(torch.float64)
        
        # stores the row indices in coo format
        self.row_indices = torch.tensor(expandptr(self.indptr), dtype = torch.int64)
        
        self.max_len = 100000000
        self.indices_pinned = torch.empty((self.max_len, ), dtype = self.indices.dtype).pin_memory()
        self.data_pinned = torch.empty((self.max_len, ), dtype = self.data.dtype).pin_memory()
        self.distances_pinned = torch.empty((self.max_len, ), dtype = torch.int64).pin_memory()
        
    def __getitem__(self, k):
        new_indptr = np.pad(np.cumsum(self.indptr[k[:, 1]] - self.indptr[k[:, 0]], 0), (1, 0))
        new_row_indices = expandptr(new_indptr, new_indptr[-1])
        
        idx = torch.from_numpy(n_ranges_nb(self.indptr[k[:, 0]], self.indptr[k[:, 1]])).to(torch.int64)
        
        self.indices_pinned.resize_(len(idx))
        self.data_pinned.resize_(len(idx))
        self.distances_pinned.resize_(len(idx))
        
        torch.index_select(self.indices, 0, idx, out= self.indices_pinned)
        torch.index_select(self.data, 0, idx, out = self.data_pinned)
        torch.index_select(self.row_indices, 0, idx, out = self.distances_pinned)
        self.distances_pinned.sub_(torch.from_numpy(k[new_row_indices, 0]))
        
        return (
            torch.tensor(new_indptr).cuda(),
            self.indices_pinned.cuda(),
            self.data_pinned.cuda(),
            self.distances_pinned.cuda()
        )


# %%
motifscores_fast = X(motifscores) # setup can take a bit longer here, because tensors are preallocated and pinned

# %%
idx = np.array([(0, 301)] * 100000)
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx), n = 2)

# %%
# still works with smaller/other windows
idx = np.array([(0, 301)] * 100)
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx), n = 2)

# %%
test_indexing(*motifscores_fast[idx])

# %% [markdown]
# Anyway, this seems to be the most consistently fast method to load all the data

# %% [markdown]
# ### Only counting

# %%
import torch_scatter


# %%
class X():
    indptr:torch.tensor
    
    def __init__(self, csr):
        self.indices = torch.from_numpy(csr.indices).to(torch.int64)
        self.indptr = torch.from_numpy(csr.indptr).to(torch.int64)
        self.data = torch.from_numpy(csr.data).to(torch.float64)
        
        # stores the row indices in coo format
        self.row_indices = torch.tensor(expandptr(self.indptr), dtype = torch.int64)
        
        self.n_cols = csr.shape[1]
        
        self.counts = torch.zeros((1000 * self.n_cols))
    
    def __getitem__(self, k):
        new_indptr = torch.nn.functional.pad(torch.cumsum(self.indptr[k[:, 1]] - self.indptr[k[:, 0]], 0), (1, 0))
        new_row_indices = torch.ops.torch_sparse.ptr2ind(new_indptr, new_indptr[-1])
        
        idx = n_ranges_nb(self.indptr[k[:, 0]].cpu().numpy(), self.indptr[k[:, 1]].cpu().numpy())
        
        # calculate the rowxcol indices
        # inplace operation seems quite a bit faster...
        new_row_indices.mul_(self.n_cols)
        new_row_indices.add_(self.indices[idx])
        rowxcol_indices = new_row_indices
        
        # comapred to the the outplace operaton
        # rowxcol_indices = (new_row_indices * self.n_cols + self.indices[idx])
        
        # calculate the sums
        # torch_scatter sum seems to be a bit faster than torch.scatter which is a bit faster than bincount
        # not a big difference though
        ## 1
        counts = torch_scatter.scatter_sum(
            torch.ones(len(rowxcol_indices)),
            rowxcol_indices,
            out = torch.zeros((k.shape[0] * self.n_cols))#.pin_memory()
        ).reshape(
            (k.shape[0], self.n_cols)
        )
        
        ## 2
        # counts = torch.zeros((k.shape[0] * self.n_cols)).pin_memory()
        # counts.scatter_(0, rowxcol_indices, torch.ones(len(rowxcol_indices)))
        
        ## 3
        # counts = torch.bincount(rowxcol_indices, minlength = k.shape[0] * self.n_cols).reshape((k.shape[0], self.n_cols))
        
        return counts


# %%
motifscores_fast = X(motifscores)
idx = torch.tensor(np.array([(0, 301)] * 1000))

# %%
# %%timeit -n 10
idx = torch.tensor(np.array([(0, 301)] * 1000))
fragmentxmotif_indices = motifscores_fast[idx]

# %%
motifscores_fast = X(motifscores)
idx = np.array([(0, 301)] * 100000)
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx), n = 10)

# %%
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# %%
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference3"):
        motifscores_fast[idx]

# %%
x = prof.key_averages(group_by_input_shape=True)

# %%
import IPython.display
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
# IPython.display.HTML("<div style='white-space: nowrap;height:500px'>" + prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10).replace("\n", "<br>") + "<\div>")

# %%
import cProfile

stats = cProfile.run("motifscores_fast[idx]", "restats")
import pstats

p = pstats.Stats("restats")
p.sort_stats("cumulative").print_stats()

# %% [markdown]
# ### Check out

# %%
gene_ix = 32


# %%
class X():
    def __init__(self, csr, max_len = 10**9):
        self.indices = torch.from_numpy(csr.indices).to(torch.int64)
        self.indptr = csr.indptr
        self.data = torch.from_numpy(csr.data).to(torch.float64)
        
        self.max_len = max_len
        
        # stores the row indices in coo format
        self.row_indices = torch.tensor(expandptr(self.indptr), dtype = torch.int64)
        
        self.indices_pinned = torch.empty((self.max_len, ), dtype = self.indices.dtype).pin_memory()
        self.data_pinned = torch.empty((self.max_len, ), dtype = self.data.dtype).pin_memory()
        self.distances_pinned = torch.empty((self.max_len, ), dtype = torch.int64).pin_memory()
        
    def __getitem__(self, k):
        new_indptr = np.pad(np.cumsum(self.indptr[k[:, 1]] - self.indptr[k[:, 0]], 0), (1, 0))
        new_row_indices = expandptr(new_indptr, new_indptr[-1])
        
        idx = torch.from_numpy(n_ranges_nb(self.indptr[k[:, 0]], self.indptr[k[:, 1]])).to(torch.int64)
        
        self.indices_pinned.resize_(len(idx))
        self.data_pinned.resize_(len(idx))
        self.distances_pinned.resize_(len(idx))
        
        torch.index_select(self.indices, 0, idx, out= self.indices_pinned)
        torch.index_select(self.data, 0, idx, out = self.data_pinned)
        torch.index_select(self.row_indices, 0, idx, out = self.distances_pinned)
        self.distances_pinned.sub_(torch.from_numpy(k[new_row_indices, 0]))
        
        return (
            torch.tensor(new_indptr),
            self.indices_pinned,
            self.data_pinned,
            self.distances_pinned
        )


# %%
motifscores_fast = X(motifscores) # setup can take a bit longer here, because tensors are preallocated and pinned

# %%
idx = np.array([gene_ix * window_width, (gene_ix+1) * window_width])[None, :]
fragment_indptr, motif_indices, locus_scores, locus_distance = motifscores_fast[idx]

# %%
motif_scores = pd.DataFrame({"motif_index":motif_indices.cpu().numpy(), "locus_score":locus_scores.cpu().numpy(), "locus_distance":locus_distance})

# %%
# select the site that scores best on a particular motif
# motif_ix = 10
# best = motif_scores.query("motif_index == @motif_ix").sort_values("locus_score", ascending = False).iloc[0]
# ingene_mid = int(best["locus_distance"])

# select the site that scores best overall
best = motif_scores.sort_values("locus_score", ascending = False).iloc[[1]].to_dict(orient='records')[0]
ingene_mid = int(best["locus_distance"])
motif_ix = int(best["motif_index"])

# %%
pwm = pwms[motifs_oi.iloc[motif_ix].name]

# %%
import math
site_start = ingene_mid - math.floor(pwm.shape[0] / 2)
site_end = ingene_mid + math.ceil(pwm.shape[0] / 2)


# %%
def create_onehot(seq):
    """
    Sequence contains integers 0 (A), 1 (C), 2 (G), 3 (T), and 4 (N)
    """
    return torch.tensor(np.eye(5, dtype = np.float32)[seq][:, :-1])


# %%
import gzip
if "genome" not in globals():
    genome = pickle.load(gzip.GzipFile((folder_data_preproc / "genome.pkl.gz"), "rb"))

# %%
chromosome, promoter_start, promoter_end, strand = promoters.iloc[gene_ix][["chr", "start", "end", "strand"]]
strand

# %%
# +1 here because of python starting at 0 yadda yadda
seq = genome[chromosome][(promoter_start + 1):(promoter_end + 1)]
seq = seq[::strand]

# %%
onehot = create_onehot(seq[site_start:site_end])

# %%
fig, (ax_score, ax_onehot, ax_pwm, ax_onehotrev, ax_scorerev) = plt.subplots(5, 1, figsize = (3, 4), sharex = True)

ntscores = pwm.flatten()[onehot.flatten().to(bool)]
ax_score.fill_between(np.arange(onehot.shape[0]), ntscores, color = "#55555533")
ax_score.scatter(np.arange(onehot.shape[0]), ntscores, c = np.array(sns.color_palette(n_colors = 4))[onehot.argmax(1)])

pd.DataFrame(onehot.numpy()).plot(ax = ax_onehot, legend = False)

pd.DataFrame(pwm.numpy()).plot(ax = ax_pwm, legend = False)

pd.DataFrame(onehot.numpy()[::-1, [3, 2, 1, 0]]).plot(ax = ax_onehotrev, legend = False)

onehot_rev = onehot.numpy()[::-1, [3, 2, 1, 0]]
ntscores = pwm.flatten()[onehot_rev.flatten().astype(bool)]
ax_scorerev.fill_between(np.arange(onehot.shape[0]), ntscores, color = "#55555533")
ax_scorerev.scatter(np.arange(onehot.shape[0]), ntscores, c = np.array(sns.color_palette(n_colors = 4))[onehot_rev.argmax(1)])

# %%
forward_score = (onehot.numpy() * pwm.numpy()).sum()
reverse_score = ((onehot.numpy()[::-1, [3, 2, 1, 0]] * pwm.numpy()).sum())
forward_score, reverse_score

# %%
assert np.isclose(best["locus_score"], reverse_score) or np.isclose(best["locus_score"], forward_score)

# %% [markdown]
# ## Creating the windows

# %%
cutwindow = np.array([-150, 150])

# %%
folds = pickle.load(open(fragments.path / "folds2.pkl", "rb"))

# %%
split = folds[0][0]

# %%
coordinates = fragments.coordinates[split.fragments_selected]
genemapping = fragments.mapping[split.fragments_selected, 1]

# %%
idx = coordinates[:, :, None] - window[0] + torch.tensor(cutwindow)
idx = idx.clamp_(0, window_width-1)

# %%
idx = coordinates[:, :, None] - window[0] + torch.tensor(cutwindow)
idx = idx.clamp_(0, window_width-1)
print(idx.data_ptr())
idx.add_(genemapping[:, None, None] * window_width)
print(idx.data_ptr())
unwrappedidx = torch.flatten(idx, -3, -2)
print(unwrappedidx.data_ptr()) # indeed returns a view
unwrappedidx.shape


# %%
def unwrap_idx(coordinates, genemapping, window, cutwindow):
    idx = coordinates[:, :, None] - window[0] + torch.tensor(cutwindow)
    idx = idx.clamp_(0, window_width-1)
    idx.add_(genemapping[:, None, None] * window_width)
    unwrappedidx = torch.flatten(idx, -3, -2).numpy()
    return unwrappedidx


# %%
motifscores_fast = X(motifscores) # setup can take a bit longer here for pinning I think

# %%
motifscores_fast[idx]

# %% [markdown]
# ## Using Cython

# %%
import numpy as np
import Cython
# %load_ext cython

# %%
# # !rm extract_1.cpython-39-x86_64-linux-gnu.so
import pyximport; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))

# %%
import sys
if "extract_motifs" in sys.modules:
    del sys.modules['extract_motifs']
import extract_motifs

# %%
cutwindow = np.array([-150, 150])

# %%
cells_oi = np.arange(0, 1000)
genes_oi = np.arange(0, 100)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()
cellxgene_oi_indptr = np.pad(np.cumsum(fragments.cellxgene_indptr[cellxgene_oi + 1] - fragments.cellxgene_indptr[cellxgene_oi]), (1, 0), "constant")

fragments_oi = torch.isin(fragments.mapping[:, 0], torch.from_numpy(cells_oi)) & torch.isin(fragments.mapping[:, 1], torch.from_numpy(genes_oi))
n_fragments = fragments_oi.sum()

coordinates = np.ascontiguousarray(fragments.coordinates[fragments_oi].to(torch.int64).numpy())
mapping = np.ascontiguousarray(fragments.mapping[fragments_oi].to(torch.int64).numpy())
genemapping = np.ascontiguousarray(fragments.mapping[fragments_oi, 1].to(torch.int64).numpy())

# %%
n_motifs = motifscores.shape[1]

# %%
motifscores_indptr = motifscores.indptr.astype(int)
motifscores_indices = motifscores.indices.astype(int)
motifscores_data = motifscores.data.astype(np.float64)

buffer_size = coordinates.shape[0] * 1000

out_fragment_indptr = torch.from_numpy(np.zeros(buffer_size, dtype = int)).numpy()
out_motif_ix = torch.from_numpy(np.zeros(buffer_size, dtype = int)).numpy()
out_score = torch.from_numpy(np.zeros(buffer_size, dtype = np.float64)).numpy()
out_distance = torch.from_numpy(np.zeros(buffer_size, dtype = int)).numpy()
out_motifcounts = torch.from_numpy(np.ascontiguousarray(np.zeros((n_fragments, n_motifs), dtype = int))).numpy()

# %%
out_n = extract_motifs.extract_all(
    coordinates,
    genemapping,
    motifscores_indptr,
    motifscores_indices,
    motifscores_data,
    *window,
    window_width,
    *cutwindow,
    out_fragment_indptr,
    out_motif_ix,
    out_score,
    out_distance,
    out_motifcounts
)

# %% [markdown]
# ### Check out

# %%
folder_motifs = pfa.get_output() / "data" / "motifs" / "hs" / "hocomoco"
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs_oi = pickle.load((folder_data_preproc / "motifs_oi.pkl").open("rb"))

# %%
motif_scores = pd.DataFrame({
    "motif_index":out_motif_ix[:out_n],
    "locus_score":out_score[:out_n],
    "locus_distance":out_distance[:out_n],
    "fragment_ix":np.repeat(np.arange(n_fragments), np.diff(out_fragment_indptr[:n_fragments+1]))
})
motif_scores["gene_ix"] = mapping[motif_scores["fragment_ix"].values, 1]
motif_scores["cell_ix"] = mapping[motif_scores["fragment_ix"].values, 0]
motif_scores["local_gene_ix"] = pd.Series(np.arange(len(genes_oi)), index = genes_oi)[motif_scores["gene_ix"]].values
motif_scores["local_cell_ix"] = pd.Series(np.arange(len(cells_oi)), index = cells_oi)[motif_scores["cell_ix"]].values
motif_scores["locall_cellxgene_ix"] = motif_scores["local_cell_ix"] * len(genes_oi) + motif_scores["local_gene_ix"]

# %%
# select the site that scores best on a particular motif
# motif_ix = 10
# best = motif_scores.query("motif_index == @motif_ix").sort_values("locus_score", ascending = False).iloc[0]
# ingene_mid = int(best["locus_distance"])

# select the site that scores best overall
best = motif_scores.sort_values("locus_score", ascending = False).iloc[[1]].to_dict(orient='records')[0]
ingene_mid = int(best["locus_distance"])
motif_ix = int(best["motif_index"])
gene_ix = best["gene_ix"]

# %%
pwm = pwms[motifs_oi.iloc[motif_ix].name]

# %%
import math
site_start = ingene_mid - math.floor(pwm.shape[0] / 2) - gene_ix * window_width
site_end = ingene_mid + math.ceil(pwm.shape[0] / 2) - gene_ix * window_width


# %%
def create_onehot(seq):
    """
    Sequence contains integers 0 (A), 1 (C), 2 (G), 3 (T), and 4 (N)
    """
    return torch.tensor(np.eye(5, dtype = np.float32)[seq][:, :-1])


# %%
import gzip
if "genome" not in globals():
    genome = pickle.load(gzip.GzipFile((folder_data_preproc / "genome.pkl.gz"), "rb"))

# %%
chromosome, promoter_start, promoter_end, strand = promoters.iloc[gene_ix][["chr", "start", "end", "strand"]]
strand

# %%
# +1 here because of python starting at 0 yadda yadda
seq = genome[chromosome][(promoter_start + 1):(promoter_end + 1)]
seq = seq[::strand]

# %%
onehot = create_onehot(seq[site_start:site_end])

# %%
fig, (ax_score, ax_onehot, ax_pwm, ax_onehotrev, ax_scorerev) = plt.subplots(5, 1, figsize = (3, 4), sharex = True)

ntscores = pwm.flatten()[onehot.flatten().to(bool)]
ax_score.fill_between(np.arange(onehot.shape[0]), ntscores, color = "#55555533")
ax_score.scatter(np.arange(onehot.shape[0]), ntscores, c = np.array(sns.color_palette(n_colors = 4))[onehot.argmax(1)])

pd.DataFrame(onehot.numpy()).plot(ax = ax_onehot, legend = False)

pd.DataFrame(pwm.numpy()).plot(ax = ax_pwm, legend = False)

pd.DataFrame(onehot.numpy()[::-1, [3, 2, 1, 0]]).plot(ax = ax_onehotrev, legend = False)

onehot_rev = onehot.numpy()[::-1, [3, 2, 1, 0]]
ntscores = pwm.flatten()[onehot_rev.flatten().astype(bool)]
ax_scorerev.fill_between(np.arange(onehot.shape[0]), ntscores, color = "#55555533")
ax_scorerev.scatter(np.arange(onehot.shape[0]), ntscores, c = np.array(sns.color_palette(n_colors = 4))[onehot_rev.argmax(1)])

# %%
forward_score = (onehot.numpy() * pwm.numpy()).sum()
reverse_score = ((onehot.numpy()[::-1, [3, 2, 1, 0]] * pwm.numpy()).sum())
forward_score, reverse_score

# %%
assert np.isclose(best["locus_score"], reverse_score) or np.isclose(best["locus_score"], forward_score)

# %% [markdown]
# ## Subsetting cellxgene

# %%
# # !rm extract_1.cpython-39-x86_64-linux-gnu.so
import pyximport; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))

# %%
import sys
if "extract_fragments" in sys.modules:
    del sys.modules['extract_fragments']
import extract_fragments

# %%
cells_oi = np.arange(0, 10000)
genes_oi = np.arange(0, 1000)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

# %%
buffer_size = len(cellxgene_oi) * 2

out_coordinates = torch.from_numpy(np.zeros((buffer_size, 2), dtype = np.int64)).numpy()
out_genemapping = torch.from_numpy(np.zeros(buffer_size, dtype = np.int64)).numpy()
out_local_cellxgene_ix = torch.from_numpy(np.zeros(buffer_size, dtype = np.int64)).numpy()

# %%
cellxgene_indptr = fragments.cellxgene_indptr.numpy().astype(np.int64)
coordinates = fragments.coordinates.numpy().astype(np.int64)
genemapping = fragments.mapping[:, 1].numpy().astype(np.int64)

# %%
n_fragments = extract_fragments.extract_fragments(
    cellxgene_oi,
    cellxgene_indptr,
    coordinates,
    genemapping,
    out_coordinates,
    out_genemapping,
    out_local_cellxgene_ix
)

# %% [markdown]
# ## Subset both

# %%
# # !rm extract_1.cpython-39-x86_64-linux-gnu.so
import pyximport; pyximport.install(reload_support=True, language_level=3, setup_args=dict(include_dirs=[np.get_include()]))

import sys
if "extract_motifs" in sys.modules:
    del sys.modules['extract_motifs']
import extract_motifs

import sys
if "extract_fragments" in sys.modules:
    del sys.modules['extract_fragments']
import extract_fragments


# %%
class FragmentMotifLoader():
    def __init__(self, fragments, motifscores, batch_size, window, cutwindow):
        self.batch_size = batch_size
        
        # store auxilliary information
        self.window = window
        self.cutwindow = cutwindow
        self.window_width = window[1] - window[0]
        self.cutwindow_width = cutwindow[1] - cutwindow[0]
        
        # store fragment data
        self.cellxgene_indptr = fragments.cellxgene_indptr.numpy().astype(np.int64)
        self.coordinates = fragments.coordinates.numpy().astype(np.int64)
        self.genemapping = fragments.mapping[:, 1].numpy().astype(np.int64)
        
        # create buffers for coordinates
        n_fragment_per_cellxgene = 2
        fragment_buffer_size = batch_size * n_fragment_per_cellxgene

        self.out_coordinates = torch.from_numpy(np.zeros((fragment_buffer_size, 2), dtype = np.int64))#.pin_memory()
        self.out_genemapping = torch.from_numpy(np.zeros(fragment_buffer_size, dtype = np.int64))#.pin_memory()
        self.out_local_cellxgene_ix = torch.from_numpy(np.zeros(fragment_buffer_size, dtype = np.int64))#.pin_memory()
        
        # create buffers for motifs
        n_motifs = motifscores.shape[1]
        n_motifs_per_fragment = 1000 # 400 motifs
        motif_buffer_size = fragment_buffer_size * n_motifs_per_fragment
        
        self.motifscores_indptr = motifscores.indptr.astype(np.int64)
        self.motifscores_indices = motifscores.indices.astype(np.int64)
        self.motifscores_data = motifscores.data.astype(np.float64)

        self.out_fragment_indptr = torch.from_numpy(np.zeros(motif_buffer_size, dtype = int))#.pin_memory()
        self.out_motif_ix = torch.from_numpy(np.zeros(motif_buffer_size, dtype = int))#.pin_memory()
        self.out_score = torch.from_numpy(np.zeros(motif_buffer_size, dtype = np.float64))#.pin_memory()
        self.out_distance = torch.from_numpy(np.zeros(motif_buffer_size, dtype = int))#.pin_memory()
        self.out_motifcounts = torch.from_numpy(np.ascontiguousarray(np.zeros((fragment_buffer_size, n_motifs), dtype = int)))#.pin_memory()
        
    def load(self, cellxgenes_oi):
        assert len(cellxgenes_oi) <= self.batch_size
        n_fragments = extract_fragments.extract_fragments(
            cellxgene_oi,
            self.cellxgene_indptr,
            self.coordinates,
            self.genemapping,
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.out_local_cellxgene_ix.numpy()
        )
        self.out_coordinates.resize_((n_fragments, 2))
        self.out_genemapping.resize_((n_fragments))
        self.out_local_cellxgene_ix.resize_((n_fragments))
        
        self.out_motifcounts.data.zero_() # this is a big slow down (~20% of function) but unsure how to fix honestly
        
        n_motifs = extract_motifs.extract_motifcounts(
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.motifscores_indptr,
            self.motifscores_indices,
            self.motifscores_data,
            *self.window,
            self.window_width,
            *self.cutwindow,
            # self.out_fragment_indptr.numpy(),
            # self.out_motif_ix.numpy(),
            # self.out_score.numpy(),
            # self.out_distance.numpy(),
            self.out_motifcounts.numpy()
        )
        # self.out_fragment_indptr.resize_(n_fragments+1)
        # self.out_motif_ix.resize_(n_motifs)
        # self.out_score.resize_(n_motifs)
        # self.out_distance.resize_(n_motifs)
        self.out_motifcounts.resize_((n_fragments, self.out_motifcounts.shape[1]))
        
        return self.out_motifcounts, self.out_local_cellxgene_ix


# %%
n_cells = 100
n_genes = 100
cutwindow = np.array([-150, 150])
loader = FragmentMotifLoader(fragments, motifscores, n_cells * n_genes, window, cutwindow)

# %%
cells_oi = np.arange(0, n_cells)
genes_oi = np.arange(0, n_genes)

cellxgene_oi = (cells_oi[:, None] * fragments.n_genes + genes_oi).flatten()

motifcounts, local_cellxgene_ix = loader.load(cellxgene_oi)

# %%
# %%timeit -n 1
motifcounts, local_cellxgene_ix = loader.load(cellxgene_oi)

# %%
(fragments.n_cells * fragments.n_genes) / len(cellxgene_oi)

# %% [markdown]
# ## Single example

# %% [markdown]
# ### Create expression

# %%
transcriptome.create_X()
transcriptome.X

# %%
transcriptome.X.shape

# %%
mean_gene_expression = transcriptome.X.dense().mean(0)

# %% [markdown]
# ### Create fragment embedder

# %%
fragment_embedding = motifcounts.to(torch.float) # 😅

# %%
fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharey = True)
# sns.heatmap(coordinates.numpy(), ax = axes[0])
sns.heatmap(fragment_embedding.detach().numpy()[:100], ax = axes[1])
axes[0].set_ylabel("Fragment")

# %% [markdown]
# ### Pool fragments

# %%
from peakfreeatac.models.promotermotif.v1 import EmbeddingGenePooler

# %%
n_embedding_dimensions = motifscores.shape[1]

# %%
embedding_gene_pooler = EmbeddingGenePooler(n_embedding_dimensions, debug = True)

# %%
cell_gene_embedding = embedding_gene_pooler(fragment_embedding, local_cellxgene_ix, n_cells, n_genes)

# %%
fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharey = True)
# sns.heatmap(coordinates.numpy(), ax = axes[0])
sns.heatmap(cell_gene_embedding[0].detach().numpy()[:100], ax = axes[1])
axes[0].set_ylabel("Fragment")

# %% [markdown]
# ### Create expression predictor

# %%
from peakfreeatac.models.promotermotif.v1 import EmbeddingToExpression

# %%
embedding_to_expression = EmbeddingToExpression(n_embedding_dimensions = n_embedding_dimensions, mean_gene_expression = mean_gene_expression)
# embedding_to_expression = EmbeddingToExpressionBias(fragments.n_genes, n_embedding_dimensions = n_embedding_dimensions, mean_gene_expression = mean_gene_expression)

# %%
expression_predicted = embedding_to_expression(cell_gene_embedding, torch.from_numpy(genes_oi))

# %% [markdown]
# ### Whole model

# %%
from peakfreeatac.models.promotermotif.v1 import FragmentEmbeddingToExpression

# %%
model = FragmentEmbeddingToExpression(fragments.n_genes, mean_gene_expression, n_embedding_dimensions)

# %%
model(motifcounts.to(torch.float), local_cellxgene_ix, n_cells, n_genes, genes_oi)

# %% [markdown]
# ## Infer

# %%
n_epochs = 1000
trace_epoch_every = 10

# lr = 1.0
lr = 1e-4

# %%
import itertools

# %%
device = "cpu"

# %%
transcriptome_X = transcriptome.X.to(device)
transcriptome_X_dense = transcriptome_X.dense()

# %%
params = model.get_parameters()

optim = torch.optim.SGD(
    params,
    lr = lr,
    momentum=0.3
)
loss = torch.nn.MSELoss(reduction = "mean")

# %%
trace = []

prev_mse_train = None
prev_mse_test = None

# %%
cells_all = np.arange(fragments.n_cells)
genes_all = np.arange(fragments.n_genes)

n_cells_step = 1000
n_genes_step = 100

cells_train = cells_all[:int(len(cells_all) * 4 / 5)]
genes_train = genes_all[:int(len(genes_all) * 4 / 5)]

cells_validation = cells_all[[cell for cell in cells_all if cell not in cells_train]]
genes_validation = genes_train
# genes_validation = genes_all[[gene for gene in genes_all if gene not in genes_train]]

# %%
def cell_gene_to_cellxgene(cells_oi, genes_oi, n_genes):
    return (cells_oi[:, None] * n_genes + genes_oi).flatten()


# %%
def create_bins(cells, genes, rg = None):
    if rg is None:
        rg = np.random.RandomState()
    cells = rg.permutation(cells)
    genes = rg.permutation(genes)

    gene_cuts = [*np.arange(0, len(genes), step = n_genes_step)] + [len(genes)]
    gene_bins = [genes[a:b] for a, b in zip(gene_cuts[:-1], gene_cuts[1:])]

    cell_cuts = [*np.arange(0, len(cells), step = n_cells_step)] + [len(cells)]
    cell_bins = [cells[a:b] for a, b in zip(cell_cuts[:-1], cell_cuts[1:])]

    cellxgene_bins = [cell_gene_to_cellxgene(cells_oi, genes_oi, fragments.n_genes) for cells_oi, genes_oi in itertools.product(cell_bins, gene_bins)]

    bins = []
    for cells_oi, genes_oi in itertools.product(cell_bins, gene_bins):
        bins.append([
            cells_oi,
            genes_oi,
            cell_gene_to_cellxgene(cells_oi, genes_oi, fragments.n_genes),
            len(cells_oi),
            len(genes_oi)
        ])
    return bins


# %%
rg = np.random.RandomState(1)
bins_train = create_bins(cells_train, genes_train, rg = rg)
bins_validation = create_bins(cells_validation, genes_validation, rg = rg)
bins_validation_trace = bins_validation[:2]

# %%
loader = FragmentMotifLoader(fragments, motifscores, n_cells_step * n_genes_step, window, cutwindow)

# %%
step_ix = 0
trace_every_step = 10
trace_validation = []

for epoch in tqdm.tqdm(range(n_epochs)):
    # train
    for cells_oi, genes_oi, cellxgene_oi, n_cells, n_genes in tqdm.tqdm(bins_train, leave = False):
        if (step_ix % trace_every_step) == 0:
            with torch.no_grad():
                print("tracing")
                mse_validation = []
                mse_validation_dummy = []
                for cells_oi, genes_oi, cellxgene_oi, n_cells, n_genes in bins_validation_trace:
                    motifcounts, local_cellxgene_ix = loader.load(cellxgene_oi)
                    motifcounts = motifcounts.to(device)
                    local_cellxgene_ix = local_cellxgene_ix.to(device)

                    transcriptome_subset = transcriptome_X_dense[cells_oi, :][:, genes_oi]

                    transcriptome_predicted = model(motifcounts.to(torch.float), local_cellxgene_ix, n_cells, n_genes, genes_oi)

                    mse = loss(transcriptome_predicted, transcriptome_subset)

                    mse_validation.append(mse.item())
                    mse_validation_dummy.append(((transcriptome_subset - transcriptome_subset.mean(0, keepdim = True)) ** 2).mean())
                mse_validation = np.mean(mse_validation)
                mse_validation_dummy = np.mean(mse_validation_dummy)
                
                print(mse_validation - mse_validation_dummy)
                
                trace_validation.append({
                    "epoch":epoch,
                    "step":step_ix,
                    "mse":mse_validation,
                    "mse_dummy":mse_validation_dummy
                })
    
        torch.set_grad_enabled(True)
        motifcounts, local_cellxgene_ix = loader.load(cellxgene_oi)
        motifcounts = motifcounts.to(device)
        local_cellxgene_ix = local_cellxgene_ix.to(device)
        
        transcriptome_subset = transcriptome_X_dense[cells_oi, :][:, genes_oi]
        
        transcriptome_predicted = model(motifcounts.to(torch.float), local_cellxgene_ix, n_cells, n_genes, genes_oi)
        
        mse = loss(transcriptome_predicted, transcriptome_subset)

        mse.backward()

        optim.step()
        optim.zero_grad()
        
        step_ix += 1

    # reshuffle the order of the bins
    bins_train = [bins_train[i] for i in np.random.choice(len(bins_train), len(bins_train), replace = False)]

# %%
bins_train = [bins_train[i] for i in np.random.choice(len(bins_train), len(bins_train), replace = False)]

# %%
if isinstance(trace_validation, list):
    trace_validation = pd.DataFrame(list(trace_validation))

# %%
fig, ax = plt.subplots()
plotdata = trace_validation.groupby("step").mean().reset_index()
ax.plot(plotdata["step"], plotdata["mse"], zorder = 6, color = "orange")
fig.savefig("trace.png")

# %%
folder_motifs = pfa.get_output() / "data" / "motifs" / "hs" / "hocomoco"
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs_oi = pickle.load((folder_data_preproc / "motifs_oi.pkl").open("rb"))

# %%
pd.Series(model.embedding_to_expression.weight1.detach().cpu().numpy(), index = motifs_oi.index).sort_values()

# %%
pd.Series(model.embedding_to_expression.weight1.detach().cpu().numpy(), index = motifs_oi.index).sort_values()

# %%
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["IRF1", "STAT2"]))

# %%
sc.pl.umap(transcriptome.adata, color = transcriptome.gene_id(["STAT2"]))

# %%

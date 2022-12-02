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
# promoter_name, (padding_negative, padding_positive) = "4k2k", (2000, 4000)
promoter_name, window = "10k10k", (-10000, 10000)
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
# #### Theoretical memory analysis

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
# That actually looks doable, although more genes or motifs would stretch the limits. Still, this means we can calculate fragment scores on-the-fly on the GPU itself

# %% [markdown]
# Could we store the scores of the whole genome on GPU?

# %%
n_motifs_per_base = len(motifscores.data) / motifscores.shape[1] / window_width / motifscores.shape[0]
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
# #### Slicing CSR

# %% [markdown]
# <img src="https://matteding.github.io/images/csr.gif" width="600" />

# %% [markdown]
# #### Indexing the `motifscores` CSR

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
# ### Multiple slices

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
    assert np.array_equal(indices, indices_gs), (len(indices), len(indices_gs))
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

@njit
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
# #### Torch version

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
        idx = n_ranges_torch(self.indptr[k[:, 0]], self.indptr[k[:, 1]])
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
len(indptr)

# %%
torch_scatter.segment_mean_csr(indices, indptr)[:10]


# %% [markdown]
# Alright :-)

# %% [markdown]
# #### Extracting the relative position

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
# #### Using pinned memory

# %% [markdown]
# We may want to (1) avoid creating new tensors, which requires paging/memory allocaton and (2) use an (existing) pinned tensor which would speed up transfer to GPU.
#
# Both ideas require a tensor of fixed size (I think), which may sound problematic because we don't know the size of the tensors initially.
# However, we can just cut away the end?

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
            self.data_pinned.cuda(),
            self.indices_pinned.cuda(),
            self.distances_pinned.cuda()
        )


# %%
motifscores_fast = X(motifscores)
idx = np.array([(0, 301)] * 100000)
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx), n = 2)


# %% [markdown]
# We can speed this up a bit by not creating an intermediate array/tensor (`self.indices[idx]`) but just using take/gather functions

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
        
        torch.index_select(self.indices, 0, idx, out= self.indices_pinned)
        torch.index_select(self.data, 0, idx, out = self.data_pinned)
        torch.index_select(self.row_indices, 0, idx, out = self.distances_pinned)
        self.distances_pinned.sub_(torch.from_numpy(k[new_row_indices, 0]))
        
        return (
            torch.tensor(new_indptr).cuda(),
            self.data_pinned.cuda(),
            self.indices_pinned.cuda(),
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
        
        torch.index_select(self.indices, 0, idx, out= self.indices_pinned)
        torch.index_select(self.data, 0, idx, out = self.data_pinned)
        torch.index_select(self.row_indices, 0, idx, out = self.distances_pinned)
        self.distances_pinned.sub_(torch.from_numpy(k[new_row_indices, 0]))
        
        return (
            torch.tensor(new_indptr).cuda(),
            self.data_pinned.cuda(),
            self.indices_pinned.cuda(),
            self.distances_pinned.cuda()
        )


# %%
motifscores_fast = X(motifscores) # setup can take a bit longer here
idx = np.array([(0, 301)] * 100000)
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx), n = 2)

# %% [markdown]
# Anyway, this seems to be the most consistently fast method to load all the data

# %% [markdown]
# #### Only counting

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
        # counts = torch_scatter.scatter_sum(
        #     torch.ones(len(rowxcol_indices)),
        #     rowxcol_indices,
        #     dim_size = k.shape[0] * self.n_cols
        # ).reshape(
        #     (k.shape[0], self.n_cols)
        # )
        
        ## 2
        counts = torch.zeros((k.shape[0] * self.n_cols))
        counts.scatter_(0, rowxcol_indices, torch.ones(len(rowxcol_indices)))
        
        ## 3
        # counts = torch.bincount(rowxcol_indices, minlength = k.shape[0] * self.n_cols).reshape((k.shape[0], self.n_cols))
        
        return counts


# %%
self = motifscores_fast

# %%
k = np.array([(0, 301)] * 100000)
idx = n_ranges_nb(self.indptr[k[:, 0]].cpu().numpy(), self.indptr[k[:, 1]].cpu().numpy())

# %%
self.indptr[k[:, 0]].cpu().numpy(), self.indptr[k[:, 1]].cpu().numpy()

# %%
self.indptr[k[:, 1]].cpu().numpy().shape

# %%
# import torch_sparse

# %%
motifscores_fast = X(motifscores)
idx = torch.tensor(np.array([(0, 301)] * 1000))

# %%
# fragmentxmotif_indices = motifscores_fast[idx]

# %%
# %%timeit -n 10
idx = torch.tensor(np.array([(0, 301)] * 1000))
fragmentxmotif_indices = motifscores_fast[idx]

# %%
motifscores_fast = X(motifscores)
idx = np.array([(0, 301)] * 100000)
time_indexing(motifscores_fast, idx = idx, n_idx = len(idx), n = 2)

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
# ## Score fragments based on motifs

# %%
folds = pickle.load(open(fragments.path / "folds2.pkl", "rb"))

# %%
cutwindow = (-150, 150)

# %%
# motifscores_count = (motifscores > np.log(100)).to(torch.float)
# motifscores_view = motifscores_count.view((motifscores_count.shape[0] * motifscores_count.shape[1], motifscores_count.shape[2]))

# %%
# motifscores_view = motifscores_view.to("cpu")
np.prod(motifscores.shape) * 32 / 8 / 1024 / 1024 / 1024

# %%
import peakfreeatac.genome.motifs

# %%
window = (-padding_negative, padding_positive)
cutwindow = (-150, 150)
# cutwindow = (-20, 20)

# %%
motifscores2 = torch.tensor(np.array((motifscores > 0).todense()), dtype = torch.float)


# %%
def get_motifscores(fragments, motifscores, window, cutwindow, fragment_batch_size = 100000):
    # extract data of split
    mapping_full = fragments.mapping
    genemapping_full = mapping_full[:, 1]
    coordinates_full = fragments.coordinates

    # total number of fragments
    n_fragments = coordinates_full.shape[0]

    fragment_cuts = np.hstack([np.arange(n_fragments, step = fragment_batch_size), [n_fragments]])
    fragment_slices = [(a, b) for a, b in zip(fragment_cuts[:-1],fragment_cuts[1:])]

    fragmentscores = torch.zeros((mapping_full.shape[0], motifscores.shape[-1]))

    for fragment_start, fragment_end in tqdm.tqdm(fragment_slices, leave = False):
        genemapping = genemapping_full[fragment_start:fragment_end]
        coordinates = coordinates_full[fragment_start:fragment_end]

        fragmentscores[fragment_start:fragment_end] = peakfreeatac.genome.motifs.score_fragments(
            genemapping,
            coordinates,
            motifscores,
            window,
            cutwindow
        )
    return fragmentscores


# %%
# sorted
fragmentscores = get_motifscores(fragments, motifscores2, window, cutwindow, fragment_batch_size = 100000)

# %%
idx = np.random.choice(motifscores.shape[0], 1000000)
idx_sorted = np.sort(idx)

# %%
# %%timeit -n 10 -r 1
densified = motifscores[idx]

# %%
# %%timeit -n 10 -r 1
densified = motifscores2[idx]

# %%
# %%timeit -n 10 -r 1
densified = motifscores2[idx_sorted]

# %%
# %%timeit -n 10 -r 1
densified = motifscores[idx_sorted]

# %%
m = (motifscores > 0).astype(float)

# %%
# %%timeit -n 10 -r 1
densified = m[idx].todense()

# %%
# %%timeit -n 10 -r 1
densified = m[idx_sorted].todense()

# %%
m2 = (motifscores > 0)

# %%
# %%timeit -n 10 -r 1
densified = m2[idx].todense()

# %%
# %%timeit -n 10 -r 1
densified = m2[idx_sorted].todense()

# %%
torch.nn.functional.avg_pool1d(

# %%
# not sorted
params = get_motifscores(fragments, motifscores > 0, window, cutwindow, fragment_batch_size = 100000)

# %%
motifscores > 0

# %%
params = get_motifscores(fragments, motifscores > 0, window, cutwindow, fragment_batch_size = 100000)

# %%
import cProfile

stats = cProfile.run("peakfreeatac.genome.motifs.score_fragments(*params)", "restats")
import pstats

p = pstats.Stats("restats")
p.sort_stats("cumulative").print_stats()

# %%
fragmentscores = get_motifscores(fragments, motifscores > 0, window, cutwindow, fragment_batch_size = 100000)

# %%
pickle.dump(fragmentscores, open("./fragmentscores.pkl", "wb"))

# %%
fragmentscores = pickle.load(open("./fragmentscores.pkl", "rb"))

# %% [markdown]
# Check out

# %%
mapping_full = fragments.mapping
genemapping_full = mapping_full[:, 1]
coordinates_full = fragments.coordinates

# %%
motif_ix = 0

# %%
maxy = fragmentscores[:, motif_ix].max(0)
print(maxy[0])
fragment_ix = maxy[1]

window_width = window[1] - window[0]

# 1st coordinate
local_start_left = coordinates_full[fragment_ix, 0].numpy() - window[0] + cutwindow[0]
local_cut_left = coordinates_full[fragment_ix, 0].numpy() - window[0]
local_end_left = coordinates_full[fragment_ix, 0].numpy() - window[0] + cutwindow[1] + 1

# 2nd coordinate
local_start_right = coordinates_full[fragment_ix, 1].numpy() - window[0] + cutwindow[0]
local_cut_right = coordinates_full[fragment_ix, 1].numpy() - window[0]
local_end_right = coordinates_full[fragment_ix, 1].numpy() - window[0] + cutwindow[1] + 1

# %%
# check whether scores were indeed calculated correctly
(
    (motifscores_count[genemapping_full[fragment_ix]][local_start_left:local_end_left][:, motif_ix].numpy()).mean() +
    (motifscores_count[genemapping_full[fragment_ix]][local_start_right:local_end_right][:, motif_ix].numpy()).mean()
)

# %%
x_left = np.arange(local_start_left,local_end_left)
x_right = np.arange(local_start_right,local_end_right)
x_tot = np.arange(local_start_left,local_end_right)
x_overlap = np.arange(local_start_right, local_end_left)

# %%
# plot region around cut site
fig, (ax) = plt.subplots(1, 1)
ax.plot(x_tot, motifscores[genemapping_full[fragment_ix]][x_tot][:, motif_ix].numpy(), color = "grey")
ax.plot(x_right, motifscores[genemapping_full[fragment_ix]][x_right][:, motif_ix].numpy(), color = "blue")
ax.plot(x_left, motifscores[genemapping_full[fragment_ix]][x_left][:, motif_ix].numpy(), color = "red")
ax.plot(x_overlap, motifscores[genemapping_full[fragment_ix]][x_overlap][:, motif_ix].numpy(), color = "purple")
ax.axvline(local_cut_left, color = "red", dashes = (2, 2))
ax.axvline(local_cut_right, color = "blue", dashes = (2, 2))

# %% [markdown]
# Look at the sequence

# %%
folder_motifs = pfa.get_output() / "data" / "motifs" / "hs" / "hocomoco"
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs_oi = pickle.load((folder_data_preproc / "motifs_oi.pkl").open("rb"))

# %%
pwm = pwms[motifs_oi.iloc[motif_ix].name]

# %%
gene_ix = genemapping_full[fragment_ix].item()
gene = fragments.var.iloc[gene_ix].name

# %%
transcriptome.symbol(gene)

# %%
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
chr, promoter_start, promoter_end, strand = promoters.loc[gene][["chr", "start", "end", "strand"]]
strand

# %%
import gzip
genome = pickle.load(gzip.GzipFile((folder_data_preproc / "genome.pkl.gz"), "rb"))

# %%
promoter = promoters.loc[gene]

# %%
# +1 here because of python starting at 0 yadda yadda
seq = genome[promoter["chr"]][(promoter_start + 1):(promoter_end + 1)]
seq = seq[::strand]

# %%
left_motifscores = motifscores[genemapping_full[fragment_ix]][x_left][:, motif_ix]

site_ix = left_motifscores.max(0)[1]
site_ix = np.argsort(left_motifscores.numpy())[::-1][5]

site_mid = x_left[site_ix]

# %%
import math
site_start = site_mid - math.floor(pwm.shape[0] / 2)
site_end = site_mid + math.ceil(pwm.shape[0] / 2)


# %%
def create_onehot(seq):
    """
    Sequence contains integers 0 (A), 1 (C), 2 (G), 3 (T), and 4 (N)
    """
    return torch.tensor(np.eye(5, dtype = np.float32)[seq][:, :-1])


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

# %% [markdown]
# ## Single example

# %%
fragmentscores = pickle.load(open("./fragmentscores.pkl", "rb"))

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
# ### Subset cells and genes

# %%
import peakfreeatac.fragments

# %%
split = pfa.fragments.SplitDouble(torch.arange(0, 100), torch.arange(0, 100))

# %%
# fragments.create_cell_fragment_mapping()

# %%
split.populate(fragments)

# %%
split_motifscores = fragmentscores[split.fragments_selected]

# %%
n_embedding_dimensions = split_motifscores.shape[1]

# %% [markdown]
# ### Create fragment embedder

# %%
fragment_embedding = split_motifscores # 😅

# %%
fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharey = True)
# sns.heatmap(coordinates.numpy(), ax = axes[0])
sns.heatmap(fragment_embedding.detach().numpy(), ax = axes[1])
axes[0].set_ylabel("Fragment")

# %% [markdown]
# ### Pool fragments

# %%
from peakfreeatac.models.promotermotif.v1 import EmbeddingGenePooler

# %%
embedding_gene_pooler = EmbeddingGenePooler(debug = True)

# %%
cell_gene_embedding = embedding_gene_pooler(fragment_embedding, split.fragment_cellxgene_ix, split.cell_n, split.gene_n)

# %%
cell_gene_embedding.detach().numpy().flatten().max()

# %% [markdown]
# ### Create expression predictor

# %%
from peakfreeatac.models.promotermotif.v1 import EmbeddingToExpression

# %%
embedding_to_expression = EmbeddingToExpression(n_embedding_dimensions = n_embedding_dimensions, mean_gene_expression = mean_gene_expression)
# embedding_to_expression = EmbeddingToExpressionBias(fragments.n_genes, n_embedding_dimensions = n_embedding_dimensions, mean_gene_expression = mean_gene_expression)

# %%
expression_predicted = embedding_to_expression(cell_gene_embedding, split.gene_ix)

# %% [markdown]
# ### Whole model

# %%
from peakfreeatac.models.promotermotif.v1 import FragmentEmbeddingToExpression

# %%
model = FragmentEmbeddingToExpression(fragments.n_genes, mean_gene_expression, n_embedding_dimensions)

# %% [markdown]
# ## Infer

# %%
fragmentscores = pickle.load(open("./fragmentscores.pkl", "rb"))

# %%
n_steps = 1000
trace_epoch_every = 10

# lr = 1.0
lr = 1e-3

# %%
import itertools

# %%
device = "cuda"

# %%
transcriptome_X = transcriptome.X.to(device)
transcriptome_X_dense = transcriptome_X.dense()

coordinates = fragments.coordinates.to(device)
mapping = fragments.mapping.to(device)

# %%
fold = folds[0]

# %%
for split in fold:
    split.motifscores = fragmentscores[split.fragments_selected].to(device)

# %%
splits_training = [split.to(device) for split in fold if split.phase == "train"]
splits_test = [split.to(device) for split in fold if split.phase == "validation"]
model = model.to(device).train(True)

# %%
params = model.get_parameters()

optim = torch.optim.SGD(
    params,
    lr = lr,
    momentum=0.9
)
loss = torch.nn.MSELoss(reduction = "mean")



trace = []

prev_mse_train = None
prev_mse_test = None
for epoch in tqdm.tqdm(range(n_steps)):
    # trace
    if (epoch % trace_epoch_every) == 0:
        # mse
        mse_test = []
        mse_train = []
        for split in itertools.chain(splits_training, splits_test):
            with torch.no_grad():
                expression_predicted = model.forward2(
                    split,
                    coordinates,
                    mapping,
                )

                transcriptome_subset = transcriptome_X_dense[split.cell_ix, :][:, split.gene_ix]
                mse = loss(expression_predicted, transcriptome_subset)

                if split.phase == "train":
                    mse_train.append(mse.detach().cpu().item())
                else:
                    mse_test.append(mse.detach().cpu().item())
        mse_train = np.mean(mse_train)
        mse_test = np.mean(mse_test)

        # train mse
        text = f"{epoch} {mse_train:.6f}"

        if prev_mse_train is not None:
            text += f" Δ{prev_mse_train-mse_train:.1e}"

        prev_mse_train = mse_train

        # mse test
        text += f" {mse_test:.6f}"

        if prev_mse_test is not None:
            text += f" Δ{prev_mse_test-mse_test:.1e}"

            # if prev_mse_test-mse_test < 0:
            #     break

        prev_mse_test = mse_test

        print(text)

        trace.append({
            "mse_train":mse_train,
            "mse_test":mse_test,
            "epoch":epoch
        })

    # train
    for split in splits_training:
        expression_predicted = model.forward2(
            split,
            coordinates,
            mapping,
        )

        # transcriptome_subset = transcriptome_X.dense_subset(split.cell_ix)[:, split.gene_ix]
        transcriptome_subset = transcriptome_X_dense[split.cell_ix, :][:, split.gene_ix]

        mse = loss(expression_predicted, transcriptome_subset) * 1000

        mse.backward()

        optim.step()
        optim.zero_grad()

    # reshuffle the order of the splits
    splits_training = [splits_training[i] for i in np.random.choice(len(splits_training), len(splits_training), replace = False)]

# %%
transcriptome_X_dense.shape

# %%
if isinstance(trace, list):
    trace = pd.DataFrame(list(trace))

# %%
fig, ax = plt.subplots()
plotdata = trace.groupby("epoch").mean().reset_index()
ax.plot(plotdata["epoch"], plotdata["mse_train"], zorder = 6, color = "red")
ax.plot(plotdata["epoch"], plotdata["mse_test"], zorder = 6, color = "orange")
fig.savefig("trace.png")

# %%

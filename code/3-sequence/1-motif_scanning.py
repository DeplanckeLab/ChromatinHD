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
# %config InlineBackend.figure_format='retina'

import pickle

import scanpy as sc
import pathlib

import torch_scatter
import torch

import tqdm.auto as tqdm

device = "cuda:0"

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
import gzip

# %%
folder_motifs = pfa.get_output() / "data" / "motifs" / "hs" / "hocomoco"
folder_motifs.mkdir(parents = True, exist_ok = True)

# %% [markdown]
# ## Process motif PWMs

# %%
# !wget https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_standard_thresholds_HUMAN_mono.txt -O {folder_motifs}/pwm_cutoffs.txt

# %%
# !wget https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_pwms_HUMAN_mono.txt -O {folder_motifs}/pwms.txt

# %%
pwms = {}
motif = None
for line in (folder_motifs / "pwms.txt").open():
    if line.startswith(">"):
        if motif is not None:
            pwms[motif_id] = motif
        motif_id = line[1:].strip("\n")
        motif = []
    else:
        motif.append([float(x) for x in line.split("\t")])

# %%
pwms = {motif_id:torch.tensor(pwm) for motif_id, pwm in pwms.items()}

# %%
pickle.dump(pwms, (folder_motifs / "pwms.pkl").open("wb"))

# %%
motifs = pd.DataFrame({"motif":pwms.keys()}).set_index("motif")

# %%
motifs["gene_label"] = motifs.index.str.split("_").str[0]

# %%
motifs["k"] = [pwms[motif].shape[0] for motif in motifs.index]

# %%
motif_cutoffs = pd.read_table(folder_motifs/"pwm_cutoffs.txt", names = ["motif", "cutoff_001", "cutoff_0005", "cutoff_0001"], skiprows=1).set_index("motif")

# %%
motifs = motifs.join(motif_cutoffs)

# %%
motifs.to_pickle(folder_motifs / "motifs.pkl")

# %%
# # !ln -s /home/wsaelens/projects/peak_free_atac/output/data/pbmc10k/genome.pkl.gz /home/wsaelens/projects/peak_free_atac/output/data/lymphoma/genome.pkl.gz

# %% [markdown]
# ## Process promoter sequences

# %%
promoter_name, window = "10k10k", np.array([-10000, 10000])
# promoter_name, window = "1k1k", np.array([-1000, 1000])
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
genome = pickle.load(gzip.GzipFile((folder_data_preproc / "genome.pkl.gz"), "rb"))

# %%
import math


# %%
def create_onehot(seq):
    """
    Sequence contains integers 0 (A), 1 (C), 2 (G), 3 (T), and 4 (N)
    """
    return torch.tensor(np.eye(5, dtype = np.float32)[seq][:, :-1])


# %%
onehot_promoters = torch.empty((promoters.shape[0], window[1] - window[0], 4))
for promoter_ix, (gene, promoter) in tqdm.tqdm(enumerate(promoters.iterrows()), total = promoters.shape[0]):
    # have to add a +1 here because the genome annotation starts from 1 while python starts from 0
    sequence = genome[promoter["chr"]][promoter["start"]+1:promoter["end"]+1]
    
    # flip sequence if strand is negative
    if promoter["strand"] == -1:
        sequence = sequence[::-1]
    
    onehot_promoters[promoter_ix] = create_onehot(sequence)

# %% [markdown]
# ## Motif scanning in promoters

# %%
pwms = pickle.load((folder_motifs / "pwms.pkl").open("rb"))
motifs = pd.read_pickle(folder_motifs / "motifs.pkl")

# %%
# motifs_oi = motifs.loc[motifs["gene_label"].isin(["TCF7", "GATA3", "IRF4"])]
# motifs_oi = motifs.iloc[:20]
# motifs_oi = motifs.loc[motifs["gene_label"].isin(["TCF7", ])]
motifs_oi = motifs

# %%
motif_oi = "ZN250_HUMAN.H11MO.0.C"
print(motif_oi)
fig, ax = plt.subplots()
pd.DataFrame(pwms[motif_oi].numpy()).plot(ax = ax)
ax.axhline(0, color = "#333333")


# %%
def scan(onehot, pwm):
    n = onehot.shape[-2]
    k = pwm.shape[-2]
    x = torch.matmul(onehot, pwm.T)
    positive = torch.zeros(((*onehot.shape[:-2], n - k+1)), device = onehot.device)
    for i in range(k):
        positive += x[..., i:n-k+i+1, i]
    del x
        
    onehot_comp = onehot[..., [3, 2, 1, 0]]
    pwm_rev = pwm.flip(0)
    x = torch.matmul(onehot_comp, pwm_rev.T)
    negative = torch.zeros(((*onehot.shape[:-2], n - k+1)), device = onehot.device)
    for i in range(k):
        negative += x[..., i:n-k+i+1, i]
    del x
        
    # return positive
    return torch.maximum(positive, negative)

onehot = torch.tensor(np.eye(4, dtype = np.float32)[np.array([0, 1, 2, 3, 3, 2, 1, 0])])
pwm = torch.tensor([[1, 0., 0., 0.], [0., 1, 0., 0.]])
motifscore = scan(onehot, pwm)
assert len(motifscore) == 8 - 2 + 1
# assert (motifscore == torch.tensor([2., 0., 2., 1., 0., 1., 0.])).all()

# %%
motifscore_cutoff = np.log(100)

# %%
# memory consumption when put on cuda
str(np.prod(onehot_promoters.shape)*32/8/1024/1024/1024) + " GiB"

# %% [markdown]
# Running this on GPU generally gives a 10x time improvement (40 minutes to 3 minutes)

# %%
cutoff_col = "cutoff_0001"

# %% tags=[]
position_ixs = []
motif_ixs = []
scores = []

onehot_promoters = onehot_promoters.to("cuda")
   
for motif_ix, motif in enumerate(tqdm.tqdm(motifs_oi.index)):
    cutoff = motifs_oi.loc[motif, cutoff_col]
    pwm = pwms[motif].to(onehot_promoters.device)
    score = scan(onehot_promoters, pwm)
    pad = (onehot_promoters.shape[-2] - score.shape[-1])
    pad_left = math.ceil(pad/2)
    pad_right = math.floor(pad/2)
    shape_left = (*score.shape[:-1], pad_left)
    shape_right = (*score.shape[:-1], pad_right)
    score = torch.cat([torch.zeros(shape_left, device = score.device), score, torch.zeros(shape_right, device = score.device)], dim = -1)
    
    # add to sparse matrix
    
    # position in this case refers to gene x position (based on promoter window)
    position_ix = torch.where(score.flatten() > cutoff)[0].cpu().numpy()
    position_ixs.extend(position_ix)
    motif_ixs.extend(np.repeat(motif_ix, len(position_ix)))
    scores.extend(score.flatten().cpu().numpy()[position_ix])
    
onehot_promoters = onehot_promoters.to("cpu")

# %%
import scipy.sparse

# %%
# convert to csr, but using coo as input
motifscores = scipy.sparse.csr_matrix((scores, (position_ixs, motif_ixs)), shape = (np.prod(onehot_promoters.shape[:2]), motifs_oi.shape[0]))

# %% [markdown]
# Assess whether motif scanning worked correctly

# %%
motif_ix = 0
# motif_ix = motifs_oi.index.tolist().index("ZN250_HUMAN.H11MO.0.C")

# %%
gene_ix = 20
pwm = pwms[motifs_oi.iloc[motif_ix].name]

# %%
chr, start, end, strand = promoters.iloc[gene_ix][["chr", "start", "end", "strand"]]

# %%
window_length = window[1] - window[0]

# %%
max_pos = motifscores[(gene_ix * window_length):((gene_ix + 1) * window_length), motif_ix].argmax()
max_score = motifscores[(gene_ix * window_length):((gene_ix + 1) * window_length), motif_ix].max()

# %%
# score = scan(onehot_promoters, pwm)
# max = score[gene_ix].max(0)
# max

# %%
# maximum score
pwm.max(1)[0].sum()

# %%
local_start = max_pos - math.floor(pwm.shape[0] / 2)
local_end = max_pos + math.ceil(pwm.shape[0] / 2)

# %%
# check score using a manual multiplication
forward_score = (onehot_promoters[gene_ix, local_start:local_end].numpy() * pwm.numpy()).sum()
reverse_score = ((onehot_promoters[gene_ix, local_start:local_end].numpy()[::-1, [3, 2, 1, 0]] * pwm.numpy()).sum())

assert np.isclose(np.max([forward_score, reverse_score]), max_score)
forward_score, reverse_score

# %%
locus_start = start + max_pos - window[0]
locus_end = start + max_pos - window[0] + pwm.shape[0]

# %%
onehot = onehot_promoters[gene_ix, local_start:local_end]

# %%
fig, (ax_score, ax_onehot, ax_pwm, ax_onehotrev, ax_scorerev) = plt.subplots(5, 1, figsize = (3, 4), sharex = True)

ntscores = pwm.flatten()[onehot.flatten().to(bool)]
ax_score.fill_between(np.arange(onehot.shape[0]), ntscores, color = "#55555533")
ax_score.scatter(np.arange(onehot.shape[0]), ntscores, c = np.array(sns.color_palette(n_colors = 4))[onehot.argmax(1)])
ax_score.set_ylabel("Forward scores", rotation = 0, ha = "right", va = "center")

pd.DataFrame(onehot.numpy()).plot(ax = ax_onehot, legend = False)
ax_onehot.set_ylabel("Forward sequence", rotation = 0, ha = "right", va = "center")

pd.DataFrame(pwm.numpy()).plot(ax = ax_pwm, legend = False)
ax_pwm.set_ylabel("PWM", rotation = 0, ha = "right", va = "center")

pd.DataFrame(onehot.numpy()[::-1, [3, 2, 1, 0]]).plot(ax = ax_onehotrev, legend = False)
ax_onehotrev.set_ylabel("Reverse sequence", rotation = 0, ha = "right", va = "center")

onehot_rev = onehot.numpy()[::-1, [3, 2, 1, 0]]
ntscores = pwm.flatten()[onehot_rev.flatten().astype(bool)]
ax_scorerev.fill_between(np.arange(onehot.shape[0]), ntscores, color = "#55555533")
ax_scorerev.scatter(np.arange(onehot.shape[0]), ntscores, c = np.array(sns.color_palette(n_colors = 4))[onehot_rev.argmax(1)])
ax_scorerev.set_ylabel("Reverse scores", rotation = 0, ha = "right", va = "center")

# %% [markdown]
# ### Save

# %%
motifscan_folder = pfa.get_output() / "motifscans" / "pbmc10k" / promoter_name
motifscan_folder.mkdir(parents=True, exist_ok=True)

# %%
# motifscan_folder2 = pfa.get_output() / "motifscans" / "pbmc10k_clustered" / promoter_name
# if motifscan_folder2.exists():
#     if motifscan_folder2.is_symlink():
#         motifscan_folder2.unlink()
#     else:
#         motifscan_folder2.rmdir()
# motifscan_folder2.symlink_to(motifscan_folder)

# %%
pickle.dump(motifs_oi, open(motifscan_folder / "motifs.pkl", "wb"))

# %%
pickle.dump(motifscores, open(motifscan_folder / "motifscores.pkl", "wb"))

# %%
# !ls -lh {motifscan_folder}/motifscores.pkl

# %%
# !ln -s 

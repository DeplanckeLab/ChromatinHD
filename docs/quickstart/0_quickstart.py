# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: chromatinhd
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quickstart - python

# %% [markdown]
# ## Install

# %% [markdown]
# ```
# # using pip
# pip install chromatinhd
#
# # (soon) using conda
# conda install -c bioconda chromatinhd
# ```
#
# To use the GPU, ensure that a PyTorch version was installed with cuda enabled:
#
# ```
# # within python
# >>> import torch
# >>> torch.cuda.is_available() # should return True
# True
# >>> torch.cuda.device_count() # should be >= 1
# 1
# ```
#
# If not, follow the instructions at https://pytorch.org/get-started/locally/. You may have to re-install PyTorch.

# %% [markdown]
# ## Prepare data

# %%
import chromatinhd as chd

# %% [markdown]
# For this quickstart, we will use a tiny example dataset. You can of course use your own data.

# %%
dataset_folder = "example"
chd.

# %% [markdown]
#
# ### Transcriptomics
#

# %%
import scanpy as sc

# %% [markdown]
# <div class="admonition note">
#   <p class="admonition-title">Batch effects</p>
#   <p>
#     Currently, none of the ChromatinHD models directly supports batch effects, although this will likely be added in the future. If you have batch effects, the current recommended workflow depends on the source of the batch effect:
#     <ul>
#     <li>If it mainly comes from ambient mRNA, we recommend to use the corrected data. The reason is that this batch effect will likely not be present in the ATAC-seq data.</li>
#     <li>If it mainly comes from biological differences (e.g. cell stress, patient differences, ...), we recommend to use the uncorrected data. The reason is that this batch effect will likely be reflected in the ATAC-seq data as well, given that the genes are truly differentially regulated between the cells.</li>
#     </ul>
#   </p>
# </div>   

# %% [markdown]
#
# ### ATAC-seq
#
# ChromatinHD simply requires a `fragments.tsv` file. This contains for each fragment its chromosome, start, end and cell barcode.
#
# - When using Cellranger, this file will be produced by the pipeline.
# - If you have a bam file, [you can use sinto to create the fragment file](https://timoast.github.io/sinto/basic_usage.html)
#

# %% [markdown]
# ### Genome
#

# %% [markdown]
# ### Cell type/state

# %% [markdown]
# ## ChromatinHD-<i>pred</i>

# %%
import chromatinhd as chd

# %%

# %% [markdown]
# ## ChromatinHD-<i>diff</i>

# %%

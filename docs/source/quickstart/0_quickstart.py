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

# %%
# autoreload
import IPython
if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic('load_ext',   'autoreload')
    IPython.get_ipython().run_line_magic('autoreload', '2')

# %%
import chromatinhd as chd

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

# To speed up training and inference, ChromatinHD stores several intermediate files to disk.

# %%
import pathlib
import shutil

dataset_folder = pathlib.Path("example")
dataset_folder.mkdir(exist_ok=True)

for file in dataset_folder.iterdir():
    if file.is_file():
        file.unlink()
    else:
        shutil.rmtree(file)

# %% [markdown]
# For this quickstart, we will use a tiny example dataset. You can of course use your own data.

# %%
import pkg_resources
import shutil

DATA_PATH = pathlib.Path(
    pkg_resources.resource_filename("chromatinhd", "data/examples/pbmc10ktiny/")
)

# copy all files from data path to dataset folder
for file in DATA_PATH.iterdir():
    shutil.copy(file, dataset_folder / file.name)

# %%
!ls {dataset_folder}

# %% [markdown]
#
# ### Transcriptomics

# %%
import scanpy as sc

adata = sc.read(dataset_folder / "transcriptome.h5ad")

# %%
transcriptome = chd.data.Transcriptome.from_adata(adata, path = dataset_folder / "transcriptome")

# %%
transcriptome.layers["X"]

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

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: chromatinhd
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Installation

# %% tags=["hide_code"]
# autoreload
import IPython

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")

# %% [markdown]
# <pre class="bash">
# # from github
# <b>pip install git+https://github.com/DeplanckeLab/ChromatinHD</b>
#
# # (soon) using pip
# pip install chromatinhd
#
# # (soon) using conda
# conda install -c bioconda chromatinhd
# </pre>

# %% [markdown]
#
# To use the GPU, ensure that a PyTorch version was installed with cuda enabled:
#

# %%
import torch

torch.cuda.is_available()  # should return True
torch.cuda.device_count()  # should be >= 1

# %% [markdown]
#
# If not, follow the instructions at https://pytorch.org/get-started/locally/. You may have to re-install PyTorch.

# %% tags=["hide_output"]
import chromatinhd as chd

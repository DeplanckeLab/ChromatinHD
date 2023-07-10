# Quickstart

## Hardware requirements

To use ChromatinHD, it is highly recommended to have a GPU available. 


## Install

```
# using pip
pip install chromatinhd

# (soon) using conda
conda install -c bioconda chromatinhd
```

To use the GPU, ensure that a PyTorch version was installed with cuda enabled:

```
# within python
>>> import torch
>>> torch.cuda.is_available() # should return True
True
>>> torch.cuda.device_count() # should be >= 1
1
```

If not, follow the instructions at https://pytorch.org/get-started/locally/

## Prepare data

### Transcriptomics

### ATAC-seq

ChromatinHD simply requires a `fragments.tsv` file, which, is typically produced by the Cellranger pipeline.
If you have a bam file, [you can use sinto to create the `fragments.tsv` file](https://timoast.github.io/sinto/basic_usage.html)

### Genome

### Cell type/state

## ChromatinHD-<i>pred</i>

## ChromatinHD-<i>diff</i>
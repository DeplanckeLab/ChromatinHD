# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Data preparation

# %% tags=["hide_code", "hide_output"]
# autoreload
import IPython

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")

# %% tags=["hide_output"]
import chromatinhd as chd

# %% [markdown]
# To speed up training and inference, ChromatinHD stores several intermediate files to disk. This includes preprocessed data and models. These will be stored in the example folder.

# %%
import pathlib

dataset_folder = pathlib.Path("example")
dataset_folder.mkdir(exist_ok=True)

# %% tags=["hide_code", "hide_output"]
import shutil

for file in dataset_folder.iterdir():
    if file.is_file():
        file.unlink()
    else:
        shutil.rmtree(file)

# %% [markdown]
# For this quickstart, we will use a tiny example dataset extracted from the 10X multiome PBMC example data. We'll copy over both the h5ad for the transcriptomics data, and the fragments.tsv for the accessibility data.

# %%
import pkg_resources
import shutil

DATA_PATH = pathlib.Path(pkg_resources.resource_filename("chromatinhd", "data/examples/pbmc10ktiny/"))

# # copy all files from data path to dataset folder
for file in DATA_PATH.iterdir():
    shutil.copy(file, dataset_folder / file.name)

# %%
# !ls {dataset_folder}

# %% [markdown]
#
# ### Transcriptomics

# %%
import scanpy as sc

adata = sc.read(dataset_folder / "transcriptome.h5ad")

# %%
transcriptome = chd.data.Transcriptome.from_adata(adata, path=dataset_folder / "transcriptome")

# %%
# !ls {dataset_folder}/*

# %% [markdown]
# <div class="result">
#   <details class="warning">
#   <summary><strong>Batch effects</strong></summary>
#   <p>
#       Currently, none of the ChromatinHD models directly supports batch effects, although this will likely be added in the future. If you have batch effects, the current recommended workflow depends on the source of the batch effect:
#     <ul>
#     <li>If it mainly comes from ambient mRNA, we recommend to use the corrected data. The reason is that this batch effect will likely not be present in the ATAC-seq data.</li>
#     <li>If it mainly comes from biological differences (e.g. cell stress, patient differences, ...), we recommend to use the uncorrected data. The reason is that this batch effect will likely be reflected in the ATAC-seq data as well, given that the genes are truly differentially regulated between the cells.</li>
#     </ul>
#   </p>
#   </details>
# </div>

# %% [markdown]
# ### Regions of interest

# %% [markdown]
# ChromatinHD defines a set of regions of interest, typically surrounding transcription start sites of a gene. Since we typically do not know which transcription start sites are used, we can either use the canonical ones (as determined by e.g. ENCODE) or use the ATAC-seq data to select the one that is most open. We will use the latter option here.

# %% [markdown]
# We first get the transcripts for each gene.

# %%
transcripts = chd.biomart.get_transcripts(chd.biomart.Dataset.from_genome("GRCh38"), gene_ids=transcriptome.var.index)
fragments_file = dataset_folder / "fragments.tsv.gz"
transcripts = chd.data.regions.select_tss_from_fragments(transcripts, fragments_file)

# %% [markdown]
# Now we can define the regions around the TSS. In this case we choose -10kb and +10kb around a TSS, although in real situations this will typically be much bigger (e.g. -100kb - +100kb)

# %%
regions = chd.data.Regions.from_transcripts(
    transcripts,
    path=dataset_folder / "regions",
    window=[-10000, 10000],
)

# %%
# !ls {dataset_folder}/*

# %% [markdown]
# <div class="result">
#   <details class="note">
#   <summary><strong>Gene vs TSS coordinates</strong></summary>
#   <p>The coordinates of the canonical transcript often do not correspond to the gene annotation that are used for e.g. RNA-seq analysis. The reason is that gene coordinates are defined based on the largest transcript in both ends.
#   </p>
#   </details>
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
# The fragment file should be indexed with tabix:

# %%
if not (dataset_folder / "fragments.tsv.gz.tbi").exists():
    import subprocess

    subprocess.run(
        [
            "tabix",
            dataset_folder / "fragments.tsv.gz",
        ]
    )

# %%
fragments = chd.data.Fragments.from_fragments_tsv(
    dataset_folder / "fragments.tsv.gz",
    regions,
    obs=transcriptome.obs,
    path=dataset_folder / "fragments",
)

# %%
fragments.create_cellxgene_indptr()

# %%
# !ls {dataset_folder}/*

# %% [markdown]
# ### Training folds

# %% [markdown]
# The final set of data are the training folds that will be used to train - and test - the model. For basic models this is simply done by randomly sampling cells.

# %%
folds = chd.data.folds.Folds(dataset_folder / "folds" / "5x1").sample_cells(fragments, 5, 1)

# %% [markdown]
# ## Optional data¶

# %% [markdown]
# ### Clusters

# %% [markdown]
# Although only needed for some models, e.g. ChromatinHD-diff, for interpretation it can be helpful to store some clustering.

# %%
clustering = chd.data.Clustering.from_labels(adata.obs["celltype"], path=dataset_folder / "clustering")

# %%
# !ls {clustering.path}

# %% [markdown]
# ### Motif scan¶

# %% [markdown]
# <details class="note" open="">
#   <summary><strong>This functionality is still under construction and is only used for visualization/interpretation use cases for now.</strong></summary>
# </details>

# %% [markdown]
# Let's first download the HOCOMOCO motif data:

# %%
motifs_folder = dataset_folder / "motifs"
motifs_folder.mkdir(exist_ok=True, parents=True)

# download cutoffs, pwms and annotations
# !wget https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_standard_thresholds_HUMAN_mono.txt -O {motifs_folder}/pwm_cutoffs.txt
# !wget https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_pwms_HUMAN_mono.txt -O {motifs_folder}/pwms.txt
# !wget https://hocomoco11.autosome.org/final_bundle/hocomoco11/core/HUMAN/mono/HOCOMOCOv11_core_annotation_HUMAN_mono.tsv -O {motifs_folder}/annot.txt

# %%
import numpy as np
import pandas as pd

# %% [markdown]
# We read in all PWMs:

# %%
pwms = {}
motif = None
for line in (motifs_folder / "pwms.txt").open():
    if line.startswith(">"):
        if motif is not None:
            pwms[motif_id] = motif
        motif_id = line[1:].strip("\n")
        motif = []
    else:
        motif.append([float(x) for x in line.split("\t")])
pwms = {motif_id: np.array(pwm) for motif_id, pwm in pwms.items()}

# %% [markdown]
# And gather some metadata about each motif

# %%
motifs = pd.DataFrame({"motif": pwms.keys()}).set_index("motif")
motif_cutoffs = pd.read_table(
    motifs_folder / "pwm_cutoffs.txt",
    names=["motif", "cutoff_001", "cutoff_0005", "cutoff_0001"],
    skiprows=1,
).set_index("motif")
motifs = motifs.join(motif_cutoffs)
annot = (
    pd.read_table(motifs_folder / "annot.txt")
    .rename(columns={"Model": "motif", "Transcription factor": "gene_label"})
    .set_index("motif")
)
motifs = motifs.join(annot)

# %% [markdown]
# You also need to provide the location where the genome fasta file is stored. In our case this is located at /data/genome/GRCh38/, which was installed using `genomepy.install_genome("GRCh38", genomes_dir = "/data/genome/")`.

# %%
import genomepy

genomepy.install_genome("GRCh38", genomes_dir="/data/genome/")

fasta_file = "/data/genome/GRCh38/GRCh38.fa"

# %% [markdown]
# Motifs can than be scanned within the regions as follows:

# %%
motifscan = chd.data.Motifscan.from_pwms(
    pwms,
    regions,
    motifs=motifs,
    cutoff_col="cutoff_0001",
    fasta_file=fasta_file,
    path=dataset_folder / "motifscan",
    device="cuda",
)

# %%
# !ls {motifscan.path}

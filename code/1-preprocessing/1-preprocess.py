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
# # Preprocess

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn as sns
sns.set_style('ticks')

import torch

import pickle

import scanpy as sc

import tqdm.auto as tqdm
import io

# %%
import peakfreeatac as pfa

# %%
# https://cf.10xgenomics.com/samples/cell-arc/2.0.1/Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote/Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote_filtered_feature_bc_matrix.h5

# %%
folder_root = pfa.get_output()
folder_data = folder_root / "data"

# dataset_name = "pbmc10k"; main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k"; genome = "GRCh38.107"; organism = "hs"
# dataset_name = "pbmc10k_gran"; main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_unsorted_10k/pbmc_unsorted_10k"; genome = "GRCh38.107"; organism = "hs"
# dataset_name = "pbmc3k"; main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k"; genome = "GRCh38.107"; organism = "hs"
# dataset_name = "lymphoma"; main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/lymph_node_lymphoma_14k/lymph_node_lymphoma_14k"; genome = "GRCh38.107"; organism = "hs"
# dataset_name = "e18brain"; main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.0/e18_mouse_brain_fresh_5k/e18_mouse_brain_fresh_5k";  genome = "mm10"; organism = "mm"
# dataset_name = "alzheimer"; main_url = "https://cf.10xgenomics.com/samples/cell-arc/2.0.1/Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote/Multiome_RNA_ATAC_Mouse_Brain_Alzheimers_AppNote";  genome = "mm10"; organism = "mm"
dataset_name = "brain"; main_url = "https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-arc/2.0.0/human_brain_3k/human_brain_3k"; genome = "GRCh38.107"; organism = "hs"

folder_data_preproc = folder_data / dataset_name
folder_data_preproc.mkdir(exist_ok = True, parents = True)

if organism == "mm":
    chromosomes = ["chr" + str(i) for i in range(20)] + ["chrX", "chrY"]
elif organism == "hs":
    chromosomes = ["chr" + str(i) for i in range(24)] + ["chrX", "chrY"]

# %% [markdown]
# ## Download

# %% [markdown]
# For an overview on the output data format, see:
# https://support.10xgenomics.com/single-cell-atac/software/pipelines/latest/algorithms/overview

# %%
# ! echo mkdir -p {folder_data_preproc}
# ! echo mkdir -p {folder_data_preproc}/bam

# %%
# # ! wget {main_url}_atac_possorted_bam.bam -O {folder_data_preproc}/bam/atac_possorted_bam.bam
# # ! wget {main_url}_atac_possorted_bam.bam.bai -O {folder_data_preproc}/bam/atac_possorted_bam.bam.bai
# ! echo wget {main_url}_atac_possorted_bam.bam -O {folder_data_preproc}/bam/atac_possorted_bam.bam
# ! echo wget {main_url}_atac_possorted_bam.bam.bai -O {folder_data_preproc}/bam/atac_possorted_bam.bam.bai
# ! echo wget {main_url}_atac_fragments.tsv.gz -O {folder_data_preproc}/bam/atac_fragments.tsv.gz

# %%
# ! wget {main_url}_filtered_feature_bc_matrix.h5 -O {folder_data_preproc}/filtered_feature_bc_matrix.h5

# %%
# !wget {main_url}_atac_fragments.tsv.gz -O {folder_data_preproc}/atac_fragments.tsv.gz

# %%
# !wget {main_url}_atac_fragments.tsv.gz.tbi -O {folder_data_preproc}/atac_fragments.tsv.gz.tbi

# %%
# !wget {main_url}_atac_peaks.bed -O {folder_data_preproc}/atac_peaks.bed

# %%
# !cat {folder_data_preproc}/atac_peaks.bed | sed '/^#/d' > {folder_data_preproc}/peaks.tsv

# %%
# !wget {main_url}_atac_peak_annotation.tsv -O {folder_data_preproc}/peak_annot.tsv

# %%
# # !wget {main_url}_atac_cut_sites.bigwig -O {folder_data_preproc}/atac_cut_sites.bigwig

# %%
# !wget http://ftp.ensembl.org/pub/release-107/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_rm.primary_assembly.fa.gz -O {folder_data_preproc}/dna.fa.gz

# %%
# !ls -lh {folder_data_preproc}/

# %%
# !ls -lh {folder_data_preproc}/../pbmc10k

# %%
if genome == "GRCh38.107":  
    # !wget http://ftp.ensembl.org/pub/release-107/gff3/homo_sapiens/Homo_sapiens.GRCh38.107.gff3.gz -O {folder_data_preproc}/genes.gff.gz
    # !wget http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes -O  {folder_data_preproc}/chromosome.sizes
    
    # to reuse from lymphoma
    # !ln -s {folder_data_preproc}/../lymphoma/dna.fa.gz {folder_data_preproc}/dna.fa.gz
    # !ln -s {folder_data_preproc}/../lymphoma/genome.pkl.gz {folder_data_preproc}/genome.pkl.gz
    
    # # !wget http://ftp.ensembl.org/pub/release-107/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.toplevel.fa.gz -O {folder_data_preproc}/dna.fa.gz
elif genome == "mm10":
    # # !wget http://ftp.ensembl.org/pub/release-98/gff3/mus_musculus/Mus_musculus.GRCm38.98.gff3.gz -O {folder_data_preproc}/genes.gff.gz
    # # !wget http://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.chrom.sizes -O  {folder_data_preproc}/chromosome.sizes
    
    # to reuse from e18brain
    # # !ln -s {folder_data_preproc}/../e18brain/dna.fa.gz {folder_data_preproc}/dna.fa.gz
    # # !ln -s {folder_data_preproc}/../e18brain/genome.pkl.gz {folder_data_preproc}/genome.pkl.gz
    
    # # !wget http://ftp.ensembl.org/pub/release-98/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna_sm.toplevel.fa.gz -O {folder_data_preproc}/dna.fa.gz

# %%
# !zcat {folder_data_preproc}/genes.gff.gz | grep -vE "^#" | awk '$3 == "gene"' > {folder_data_preproc}/genes.gff

# %% [markdown]
# ### Genome

# %%
if (organism == "hs") and (dataset_name != "pbmc10k"):
    # !rm {folder_data_preproc}/fasta.fa
    # !rm {folder_data_preproc}/genome.pkl.gz
    # !ln -s {folder_data_preproc}/../pbmc10k/fasta.fa {folder_data_preproc}/
    # !ln -s {folder_data_preproc}/../pbmc10k/genome.pkl.gz {folder_data_preproc}/
elif (organism == "mm") and (dataset_name != "e18brain"):
    # !rm {folder_data_preproc}/fasta.fa
    # !rm {folder_data_preproc}/genome.pkl.gz
    # !ln -s {folder_data_preproc}/../e18brain/fasta.fa {folder_data_preproc}/
    # !ln -s {folder_data_preproc}/../e18brain/genome.pkl.gz {folder_data_preproc}/

# %%
# from operator import xor
# y = lambda bp: 0b11 & xor((ord(bp) >> 2), (ord(bp) >> 1))

# %%
import gzip
genome = {}
chromosome = None
translate_table = {"A":0, "C":1, "G":2, "T":3, "N":4} # alphabetic order
for i, line in enumerate(gzip.GzipFile(folder_data_preproc / "dna.fa.gz")):
    line = str(line,'utf-8')
    if line.startswith(">"):
        if chromosome is not None:
            genome[chromosome] = np.array(genome_chromosome, dtype = np.int8)
        chromosome = "chr" + line[1:line.find(" ")]
        genome_chromosome = []
        
        print(chromosome)
        
        if chromosome not in chromosomes:
            break
    else:
        genome_chromosome += [translate_table[x] for x in line.strip("\n").upper()]

# %%
# to link between datasets with the same genome/organism
# # !ln -s {folder_data_preproc}/../pbmc10k/genome.pkl.gz {folder_data_preproc}/

pickle.dump(genome, gzip.GzipFile((folder_data_preproc / "genome.pkl.gz"), "wb", compresslevel = 3))

# %%
# !ls -lh {folder_data_preproc}

# %% [markdown]
# ### Genes

# %%
biomart_dataset_name = "mmusculus_gene_ensembl" if organism == "mm" else "hsapiens_gene_ensembl"

# %%
if genome == "GRCh38.107":
    query = f"""<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE Query>
    <Query  virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >

        <Dataset name = "{biomart_dataset_name}" interface = "default" >
            <Filter name = "transcript_is_canonical" excluded = "0"/>
            <Filter name = "transcript_biotype" value = "protein_coding"/>
            <Attribute name = "ensembl_gene_id" />
            <Attribute name = "transcript_start" />
            <Attribute name = "transcript_end" />
            <Attribute name = "end_position" />
            <Attribute name = "start_position" />
            <Attribute name = "ensembl_transcript_id" />
            <Attribute name = "chromosome_name" />
            <Attribute name = "strand" />
            <Attribute name = "external_gene_name" />
        </Dataset>
    </Query>"""
    url = "http://www.ensembl.org/biomart/martservice?query=" + query.replace("\t", "").replace("\n", "")
    from io import StringIO
    import requests
    session = requests.Session()
    session.headers.update({'User-Agent': 'Custom user agent'})
    r = session.get(url)
    result = pd.read_table(StringIO(r.content.decode("utf-8")))
else:
    query = f"""<?xml version="1.0" encoding="UTF-8"?>
    <!DOCTYPE Query>
    <Query  virtualSchemaName = "default" formatter = "TSV" header = "1" uniqueRows = "0" count = "" datasetConfigVersion = "0.6" >

        <Dataset name = "{biomart_dataset_name}" interface = "default" >
            <Filter name = "transcript_biotype" value = "protein_coding"/>
            <Attribute name = "ensembl_gene_id" />
            <Attribute name = "transcript_start" />
            <Attribute name = "transcript_end" />
            <Attribute name = "end_position" />
            <Attribute name = "start_position" />
            <Attribute name = "ensembl_transcript_id" />
            <Attribute name = "chromosome_name" />
            <Attribute name = "strand" />
            <Attribute name = "external_gene_name" />
        </Dataset>
    </Query>"""
    url = "https://nov2020.archive.ensembl.org:443/biomart/martservice?query=" + query.replace("\t", "").replace("\n", "")
    from io import StringIO
    import requests
    session = requests.Session()
    session.headers.update({'User-Agent': 'Custom user agent'})
    r = session.get(url)
    result = pd.read_table(StringIO(r.content.decode("utf-8")))

# %%
genes = result.rename(columns = {
    "Gene stable ID":"gene",
    "Transcript start (bp)":"start",
    "Transcript end (bp)":"end",
    "Chromosome/scaffold name":"chr",
    "Gene name":"symbol",
    "Strand":"strand"
})
genes["chr"] = "chr" + genes["chr"].astype(str)
genes = genes.groupby("gene").first()

# %%
genes = genes.loc[genes["chr"].isin(chromosomes)]

# %%
assert genes.groupby(level = 0).size().mean() == 1, "For each gene, there should only be one transcript"

# %%
genes.to_csv(folder_data_preproc / "genes.csv")

# %% [markdown]
# ## Create transcriptome

# %%
genes = pd.read_csv(folder_data_preproc / "genes.csv", index_col = 0)

# %%
import peakfreeatac.data

# %%
transcriptome = peakfreeatac.data.Transcriptome(folder_data_preproc / "transcriptome")

# %% [markdown]
# ### Read and process

# %%
adata = sc.read_10x_h5(folder_data_preproc / "filtered_feature_bc_matrix.h5")

# %%
adata.var.index.name = "symbol"
adata.var = adata.var.reset_index()
adata.var.index = adata.var["gene_ids"]
adata.var.index.name = "gene"

all_gene_ids = sorted(list(set(genes.loc[genes["chr"].isin(chromosomes)].index) & set(adata.var.index)))

adata = adata[:, all_gene_ids]

# %%
adata.var_names_make_unique()

# %%
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_counts = 1000)
print(adata.obs.shape[0])
sc.pp.filter_cells(adata, min_genes= 200)
print(adata.obs.shape[0])
sc.pp.filter_genes(adata, min_cells=3)
print(adata.var.shape[0])

# %%
sc.external.pp.scrublet(adata)

# %%
adata.obs["doublet_score"].plot(kind = "hist")

# %%
adata.obs["doublet"] = (adata.obs["doublet_score"] > 0.1).astype("category")

print(adata.obs.shape[0])
adata = adata[~adata.obs["doublet"].astype(bool)]
print(adata.obs.shape[0])

# %%
size_factor = np.median(np.array(adata.X.sum(1)))
adata.uns["size_factor"] = size_factor

# %%
sc.pp.normalize_total(adata, size_factor)

# %%
sc.pp.log1p(adata)

# %%
sc.pp.pca(adata)

# %%
sc.pp.highly_variable_genes(adata)

# %%
adata.var["n_cells"] = np.array((adata.X > 0).sum(0))[0]

# %%
# adata = adata[:, adata.var["dispersions_norm"].sort_values(ascending = False)[:5000].index]
print(adata.var.shape[0])

genes_oi = adata.var.query("n_cells > 100")["dispersions_norm"].sort_values(ascending = False)[:5000].index.tolist()

if dataset_name == "pbmc10k":
    genes_oi = genes_oi[:-1] + ["ENSG00000115977"]

adata = adata[:, genes_oi]
print(adata.var.shape[0])

all_gene_ids = adata.var.index

# %%
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# %%
adata.var["chr"] = genes["chr"]

# %%
sc.tl.leiden(adata, resolution = 1)

# %%
sc.pl.umap(adata, color = "leiden")

# %%
if dataset_name == "pbmc10k":
    adata = adata[~adata.obs["leiden"].isin(["16", "21", "20", "23"])]

# %%
sc.pl.umap(adata, color = "leiden")

# %%
transcriptome.adata = adata
transcriptome.var = adata.var
transcriptome.obs = adata.obs

# %%
transcriptome.create_X()

# %%
fig, ax = plt.subplots()
sns.scatterplot(
    adata.var,
    x = "means", y = "dispersions_norm"
)
ax.set_xscale("log")

# %%
# adata.var['mt'] = adata.var["symbol"].str.startswith('MT')  # annotate the group of mitochondrial genes as 'mt'
# sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# %%
genes_oi = adata.var.sort_values("dispersions_norm", ascending = False).index[:10]
# genes_oi = transcriptome.gene_id(["Nrg1", "Nrg3", "Cdk1"])
sc.pl.umap(adata, color=genes_oi, title = transcriptome.symbol(genes_oi))

# %%
sc.pl.umap(adata, color="leiden", legend_loc="on data")

# %% [markdown]
# ### Interpret E18 brain

# %%
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, gene_symbols = "symbol")

# %%
cluster_annotation = pd.read_table(io.StringIO("""leiden  celltype
14  Cycling
17,5  Whatever
"""), sep = "  ", engine = "python").set_index("celltype")
cluster_annotation = cluster_annotation["leiden"].str.split(",").explode().to_frame().reset_index().set_index("leiden")["celltype"]
cluster_annotation = cluster_annotation.reindex(adata.obs["leiden"].unique())

adata.obs["celltype"] = cluster_annotation.loc[adata.obs["leiden"]].values

# %%
# genes_oi = adata.var.sort_values("dispersions_norm", ascending = False).index[:10]

# genes_oi = transcriptome.gene_id(["Pax6", "Nhlh2", "Gad2", "Slc32a1", "Dlx5", "Ebf1"])

# genes_oi = transcriptome.gene_id(["Eomes", "Pax6", "Slc17a6", "Slc17a7", "Satb2", "Tbr1"])

genes_oi = transcriptome.gene_id(["Tnc", "Egfr"])
sc.pl.umap(adata, color=genes_oi, title = transcriptome.symbol(genes_oi))

# %%
sc.pl.rank_genes_groups_matrixplot(adata, ["5"], gene_symbols = "symbol")

# %% [markdown]
# ### Interpret Lymphoma

# %%
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, gene_symbols = "symbol")

# %%
sc.pl.rank_genes_groups_matrixplot(adata, ["22"], gene_symbols = "symbol")

# %%
cluster_annotation = pd.read_table(io.StringIO("""leiden  celltype
0  T cycling
7,5,19,14,2  Lymphoma
17,23  Lymphoma cycling
21  B
11  NK
27,29  cDCs
1,9  Monocytes
26  Stromal
30  Plasma
28  pDCs
22,18,4,3,20,6,12,8,10,25,13,15,16,24  T
"""), sep = "  ", engine = "python").set_index("celltype")
cluster_annotation = cluster_annotation["leiden"].str.split(",").explode().to_frame().reset_index().set_index("leiden")["celltype"]

adata.obs["celltype"] = cluster_annotation.loc[adata.obs["leiden"]].values

# %%
sc.pl.umap(adata, color="celltype", legend_loc="on data")

# %%
transcriptome.obs = adata.obs
transcriptome.adata = adata

# %% [markdown]
# ### Interpret PBMC10K

# %%
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
# sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, gene_symbols = "symbol")

# %%
import io

# %%
marker_annotation = pd.read_table(io.StringIO("""ix	symbols	celltype
0	IL7R, CD3D	CD4 naive T
0	IL7R, CD3D, ITGB1	CD4 memory T
1	CD14, LYZ	CD14+ Monocytes
2	MS4A1, IL4R, CD79A	naive B
2	MS4A1, CD79A, TNFRSF13B	memory B
3	CD8A, CD3D	CD8 naive T
4	GNLY, NKG7, GZMA, GZMB, NCAM1	NK
4	GNLY, NKG7, CD3D, CCL5, GZMA, CD8A	CD8 activated T
4	SLC4A10	MAIT
5	FCGR3A, MS4A7	FCGR3A+ Monocytes
5	CD27, JCHAIN	Plasma
6	TCF4	pDCs
6	FCER1A, CST3	cDCs
""")).set_index("celltype")
marker_annotation["symbols"] = marker_annotation["symbols"].str.split(", ")
# marker_annotation = marker_annotation.explode("symbols")

# %%
marker_annotation = pd.read_table(io.StringIO("""ix	symbols	celltype
0	IL7R, CD3D	CD4 naive T
0	IL7R, CD3D, ITGB1	CD4 memory T
1	CD14, LYZ	CD14+ Monocytes
2	MS4A1, IL4R, CD79A	naive B
2	MS4A1, CD79A, TNFRSF13B	memory B
3	CD8A, CD3D	CD8 naive T
4	GNLY, NKG7, GZMA, GZMB, NCAM1	NK
4	GNLY, NKG7, CD3D, CCL5, GZMA, CD8A	CD8 activated T
4	SLC4A10	MAIT
5	FCGR3A, MS4A7	FCGR3A+ Monocytes
5	CD27, JCHAIN	Plasma
6	TCF4	pDCs
6	CST3	cDCs
""")).set_index("celltype")
marker_annotation["symbols"] = marker_annotation["symbols"].str.split(", ")
# marker_annotation = marker_annotation.explode("symbols")

# %%
sc.pl.umap(
    adata,
    color = transcriptome.gene_id(marker_annotation.query("celltype == 'pDCs'")["symbols"].explode())
)


# %%
#Define cluster score for all markers
def evaluate_partition(anndata, marker_dict, gene_symbol_key=None, partition_key='louvain_r1'):
    # Inputs:
    #    anndata         - An AnnData object containing the data set and a partition
    #    marker_dict     - A dictionary with cell-type markers. The markers should be stores as anndata.var_names or 
    #                      an anndata.var field with the key given by the gene_symbol_key input
    #    gene_symbol_key - The key for the anndata.var field with gene IDs or names that correspond to the marker 
    #                      genes
    #    partition_key   - The key for the anndata.obs field where the cluster IDs are stored. The default is
    #                      'louvain_r1' 

    #Test inputs
    if partition_key not in anndata.obs.columns.values:
        print('KeyError: The partition key was not found in the passed AnnData object.')
        print('   Have you done the clustering? If so, please tell pass the cluster IDs with the AnnData object!')
        raise

    if (gene_symbol_key != None) and (gene_symbol_key not in anndata.var.columns.values):
        print('KeyError: The provided gene symbol key was not found in the passed AnnData object.')
        print('   Check that your cell type markers are given in a format that your anndata object knows!')
        raise
        

    if gene_symbol_key:
        gene_ids = anndata.var[gene_symbol_key]
    else:
        gene_ids = anndata.var_names

    clusters = np.unique(anndata.obs[partition_key])
    n_clust = len(clusters)
    n_groups = len(marker_dict)
    
    marker_res = np.zeros((n_groups, n_clust))
    z_scores = sc.pp.scale(anndata, copy=True)

    i = 0
    for group in marker_dict:
        # Find the corresponding columns and get their mean expression in the cluster
        j = 0
        for clust in clusters:
            cluster_cells = np.in1d(z_scores.obs[partition_key], clust)
            marker_genes = np.in1d(gene_ids, marker_dict[group])
            marker_res[i,j] = z_scores.X[np.ix_(cluster_cells,marker_genes)].mean()
            j += 1
        i+=1

    variances = np.nanvar(marker_res, axis=0)
    if np.all(np.isnan(variances)):
        print("No variances could be computed, check if your cell markers are in the data set.")
        print("Maybe the cell marker IDs do not correspond to your gene_symbol_key input or the var_names")
        raise

    marker_res_df = pd.DataFrame(marker_res, columns=clusters, index=marker_dict.keys())

    #Return the median of the variances over the clusters
    return(marker_res_df)

# %%
cluster_celltypes = evaluate_partition(adata, marker_annotation["symbols"].to_dict(), "symbol", partition_key="leiden").idxmax()

# %%
adata.obs["celltype"] = cluster_celltypes[adata.obs["leiden"]].values
adata.obs["celltype"] = adata.obs["celltype"].astype(str)
# adata.obs.loc[adata.obs["leiden"] == "4", "celltype"] = "NKT"

# %%
transcriptome.adata.obs["log_n_counts"] = np.log(transcriptome.adata.obs["n_counts"])

# %%
sc.pl.umap(
    adata,
    color = ["celltype", "log_n_counts", "leiden"]
)
sc.pl.umap(
    adata,
    color = transcriptome.gene_id(marker_annotation["symbols"].explode()),
    title = marker_annotation["symbols"].explode()
)

# %%
transcriptome.obs = adata.obs
transcriptome.adata = adata

# %% [markdown]
# ## Create cell type pseudobulk dataset

# %%
import peakfreeatac.transcriptome

# %%
dataset_name_original = "pbmc10k"
dataset_name = dataset_name_original + "_clustered"

# %%
folder_data_preproc_original = folder_data_preproc.parent / (dataset_name_original)

# %%
transcriptome_original = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc_original / "transcriptome")

# %%
folder_data_preproc = folder_data_preproc.parent / dataset_name
transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")

# %%
sc.tl.leiden(transcriptome_original.adata, resolution = 100)

# %%
transcriptome_original.adata.obs.groupby("leiden").size().plot(kind = "hist")

# %%
# transcriptome_original.adata.obs["leiden"] = np.arange(transcriptome_original.adata.obs.shape[0])

# %%
groupby = transcriptome_original.adata.obs.reset_index().groupby("leiden")
obs = groupby["cell"].apply(list).to_frame()
obs.columns = ["cell_original"]
obs.index.name = "cell"
obs.index = obs.index.astype(str)

# %%
X = {}
for celltype, group in groupby:
    X[celltype] = np.array(transcriptome_original.adata[group.index].X.mean(0))[0]
X = pd.DataFrame(X).T.values

# %%
transcriptome.var = transcriptome_original.var
transcriptome.obs = obs

# %%
adata = sc.AnnData(X, obs = transcriptome.obs, var = transcriptome.var)

# %%
transcriptome.adata = adata

# %%
transcriptome.create_X()

# %%
# !ln -s {folder_data_preproc_original}/promoters_10k10k.csv {folder_data_preproc}/promoters_10k10k.csv

# %%
# !ln -s {folder_data_preproc_original}/genes.gff {folder_data_preproc}/genes.gff

# %% [markdown]
# ## Create windows

# %% [markdown]
# ### Creating promoters

# %%
import tabix

# %%
fragments_tabix = tabix.open(str(folder_data_preproc / "atac_fragments.tsv.gz"))

# %% [markdown]
# #### Define promoters

# %%
promoter_name, (padding_negative, padding_positive) = "4k2k", (2000, 4000)
promoter_name, (padding_negative, padding_positive) = "10k10k", (10000, 10000)
# promoter_name, (padding_negative, padding_positive) = "20kpromoter", (10000, 0)
# promoter_name, (padding_negative, padding_positive) = "1k1k", (1000, 1000)

# %%
import pybedtools

# %%
all_gene_ids = transcriptome.var.index

# %%
promoters = pd.DataFrame(index = all_gene_ids)

# %%
promoters["tss"] = [genes_row["start"] if genes_row["strand"] == +1 else genes_row["end"] for _, genes_row in genes.loc[promoters.index].iterrows()]
promoters["strand"] = genes["strand"]
promoters["positive_strand"] = (promoters["strand"] == 1).astype(int)
promoters["negative_strand"] = (promoters["strand"] == -1).astype(int)
promoters["chr"] = genes.loc[promoters.index, "chr"]

# %%
promoters["start"] = promoters["tss"] - padding_negative * promoters["positive_strand"] - padding_positive * promoters["negative_strand"]
promoters["end"] = promoters["tss"] + padding_negative * promoters["negative_strand"] + padding_positive * promoters["positive_strand"]

# %%
promoters = promoters.drop(columns = ["positive_strand", "negative_strand"], errors = "ignore")

# %%
promoters

# %%
promoters.to_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"))

# %% [markdown]
# #### Create fragments

# %%
promoters = pd.read_csv(folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col = 0)

# %%
transcriptome = peakfreeatac.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
import pathlib
import peakfreeatac.data
fragments = pfa.data.Fragments(folder_data_preproc / "fragments" / promoter_name)

# %%
var = pd.DataFrame(index = promoters.index)
var["ix"] = np.arange(var.shape[0])

n_genes = var.shape[0]

# %%
obs = transcriptome.adata.obs[[]].copy()
obs.index.name = "cell"
obs["ix"] = np.arange(obs.shape[0])

if "cell_original" in transcriptome.adata.obs.columns:
    cell_ix_to_cell = transcriptome.adata.obs["cell_original"].explode()
    cell_to_cell_ix = pd.Series(cell_ix_to_cell.index.astype(int), cell_ix_to_cell.values)
else:
    cell_to_cell_ix = obs["ix"].to_dict()

n_cells = obs.shape[0]

# %%
gene_to_fragments = [[] for i in var["ix"]]
cell_to_fragments = [[] for i in obs["ix"]]

# %%
coordinates_raw = []
mapping_raw = []

for i, (gene, promoter_info) in tqdm.tqdm(enumerate(promoters.iterrows()), total = promoters.shape[0]):
    gene_ix = var.loc[gene, "ix"]
    fragments_promoter = fragments_tabix.query(*promoter_info[["chr", "start", "end"]])
    
    for fragment in fragments_promoter:
        cell = fragment[3]
        
        # only store the fragment if the cell is actually of interest
        if cell in cell_to_cell_ix:
            # add raw data of fragment relative to tss
            coordinates_raw.append([
                (int(fragment[1]) - promoter_info["tss"]) * promoter_info["strand"],
                (int(fragment[2]) - promoter_info["tss"]) * promoter_info["strand"]
            ][::promoter_info["strand"]])

            # add mapping of cell/gene
            mapping_raw.append([
                cell_to_cell_ix[fragment[3]],
                gene_ix
            ])

# %%
fragments.var = var
fragments.obs = obs

# %% [markdown]
# Create fragments tensor

# %%
coordinates = torch.tensor(np.array(coordinates_raw, dtype = np.int64))
mapping = torch.tensor(np.array(mapping_raw), dtype = torch.int64)

# %% [markdown]
# Sort `coordinates` and `mapping` according to `mapping`

# %%
sorted_idx = torch.argsort((mapping[:, 0] * var.shape[0] + mapping[:, 1]))
mapping = mapping[sorted_idx]
coordinates = coordinates[sorted_idx]

# %% [markdown]
# Check size

# %%
np.product(mapping.size()) * 64 / 8 / 1024 / 1024

# %%
np.product(mapping.size()) * 64 / 8 / 1024 / 1024

# %%
np.product(coordinates.size()) * 64 / 8 / 1024 / 1024

# %% [markdown]
# Store

# %%
fragments.mapping = mapping
fragments.coordinates = coordinates

# %% [markdown]
# Create cellxgene index pointers

# %%
fragments.create_cellxgene_indptr()

# %% [markdown]
# #### Create training folds

# %%
n_bins = 5

# %%
# train/test split
cells_all = np.arange(fragments.n_cells)
genes_all = np.arange(fragments.n_genes)

cell_bins = np.floor((np.arange(len(cells_all))/(len(cells_all)/n_bins)))

chromosome_gene_counts = transcriptome.var.groupby("chr").size().sort_values(ascending = False)
chromosome_bins = np.cumsum(((np.cumsum(chromosome_gene_counts) % (chromosome_gene_counts.sum() / n_bins + 1)).diff() < 0))

gene_bins = chromosome_bins[transcriptome.var["chr"]].values

n_folds = 5
folds = []
for i in range(n_folds):
    cells_train = cells_all[cell_bins != i]
    cells_validation = cells_all[cell_bins == i]

    chromosomes_train = chromosome_bins.index[~(chromosome_bins == i)]
    chromosomes_validation = chromosome_bins.index[chromosome_bins == i]
    genes_train = fragments.var["ix"][transcriptome.var.index[transcriptome.var["chr"].isin(chromosomes_train)]].values
    genes_validation = fragments.var["ix"][transcriptome.var.index[transcriptome.var["chr"].isin(chromosomes_validation)]].values
    
    folds.append({
        "cells_train":cells_train,
        "cells_validation":cells_validation,
        "genes_train":genes_train,
        "genes_validation":genes_validation
    })
pickle.dump(folds, (fragments.path / "folds.pkl").open("wb"))

# %% [markdown]
# ## Fragment distribution

# %%
import gzip

# %%
sizes = []
with gzip.GzipFile(folder_data_preproc / "atac_fragments.tsv.gz", "r") as fragment_file:
    i = 0
    for line in fragment_file:
        line = line.decode("utf-8")
        if line.startswith("#"):
            continue
        split = line.split("\t")
        sizes.append(int(split[2]) - int(split[1]))
        i += 1
        if i > 1000000:
            break

# %%
sizes = np.array(sizes)

# %%
np.isnan(sizes).sum()

# %%
fig, ax = plt.subplots()
ax.hist(sizes, range = (0, 1000), bins = 100)
ax.set_xlim(0, 1000)

# %%
import scipy

# %%
gamma_params = scipy.stats.gamma.fit(sizes)

# %%
dist = scipy.stats.gamma(*gamma_params)

# %%
xs = np.linspace(0, 1000)
ys = dist.pdf(xs)

# %%
fig, ax = plt.subplots()
ax.plot(xs, ys)
ax.hist(sizes, range = (0, 1000), bins = 100, density = True)
ax.set_xlim(0, 1000)

# %% [markdown]
# ## Fragment distributions across datasets

# %%
import gzip

# %%
dataset_names = ["pbmc10k_gran", "pbmc10k", "pbmc3k", "lymphoma", "e18brain"]

# %%
sizes = {}
for dataset_name in dataset_names:
    sizes_dataset = []
    with gzip.GzipFile(folder_data / dataset_name / "atac_fragments.tsv.gz", "r") as fragment_file:
        i = 0
        for line in fragment_file:
            line = line.decode("utf-8")
            if line.startswith("#"):
                continue
            split = line.split("\t")
            sizes_dataset.append(int(split[2]) - int(split[1]))
            i += 1
            if i > 1000000:
                break
    sizes[dataset_name] = sizes_dataset

# %%
bins = np.linspace(0, 1000, 100+1)

# %%
bincounts = {dataset_name:np.histogram(x, bins, density = True)[0] for dataset_name, x in sizes.items()}


# %%
def ecdf(a):
    x = np.sort(a)
    y = np.arange(len(x))/float(len(x))
    return y


# %%
fig, ax = plt.subplots()
for dataset_name, bincounts_dataset in bincounts.items():
    x = np.sort(sizes[dataset_name])
    x_ecdf = ecdf(x)
    ax.plot(x, x_ecdf, label = dataset_name)
    ax.set_xscale("log")
ax.legend()
ax.set_xlabel("fragment length")
ax.set_ylabel("ECDF", rotation = 0, ha = "right", va = "center")

# %%
fig, ax = plt.subplots()
for dataset_name in sizes.keys():
    ax.hist(sizes[dataset_name], range = (0, 1000), bins = 100, histtype = "step", label = dataset_name)
    ax.set_xlim(0, 1000)
plt.legend()
ax.set_xlabel("fragment length")
ax.set_ylabel("# fragments", rotation = 0, ha = "right", va = "center")

# %%

# %% [markdown]
# ## Create latent space

# %%
import peakfreeatac.data

# %%
transcriptome = peakfreeatac.data.Transcriptome(folder_data_preproc / "transcriptome")

# %%
sc.pp.neighbors(transcriptome.adata)

# %%
resolution = 0.1

# %%
sc.tl.leiden(transcriptome.adata, resolution = resolution)

# %%
latent = pd.get_dummies(transcriptome.adata.obs["leiden"])
latent.columns = pd.Series("leiden_" + latent.columns.astype(str))

# %%
latent_folder = folder_data_preproc / "latent"
latent_folder.mkdir(exist_ok = True)

# %%
latent_name = "leiden_" + str(resolution)

# %%
latent.to_pickle(latent_folder / (latent_name + ".pkl"))

# %% [markdown]
# ## Store transcripts and exons

# %%
# !zcat {folder_data_preproc}/genes.gff.gz | grep -vE "^#" | awk '$3 == "exon"' > {folder_data_preproc}/exons.gff
# !zcat {folder_data_preproc}/genes.gff.gz | grep -vE "^#" | awk '$3 == "protein_coding"' > {folder_data_preproc}/transcript.gff

# %%
transcripts = pd.read_table(folder_data_preproc / "transcript.gff", sep = "\t", names = ["chr", "type", "__", "start", "end", "dot", "strand", "dot2", "info"])

# %%
transcripts

# %%
# !zcat {folder_data_preproc}/genes.gff.gz

# %%
# !zcat {folder_data_preproc}/genes.gff.gz | grep -vE "^#" | head -n 100

# %%
transcripts

# %%
# !head {folder_data_preproc}/transcript.gff

# %%
gff = pd.read_table(folder_data_preproc / "genes.gff", sep = "\t", names = ["chr", "type", "__", "start", "end", "dot", "strand", "dot2", "info"])
genes = gff.copy()
genes["chr"] = "chr" + genes["chr"]
genes["symbol"] = genes["info"].str.split(";").str[1].str[5:]
genes["gene"] = genes["info"].str.split(";").str[0].str[8:]
genes = genes.set_index("gene", drop = False)

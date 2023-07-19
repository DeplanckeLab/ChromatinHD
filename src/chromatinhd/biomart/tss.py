import pandas as pd
from .dataset import Dataset


def get_canonical_transcripts(biomart_dataset: Dataset, gene_ids):
    genes = biomart_dataset.get(
        [
            biomart_dataset.attribute("ensembl_gene_id"),
            biomart_dataset.attribute("transcript_start"),
            biomart_dataset.attribute("transcript_end"),
            biomart_dataset.attribute("end_position"),
            biomart_dataset.attribute("start_position"),
            biomart_dataset.attribute("ensembl_transcript_id"),
            biomart_dataset.attribute("chromosome_name"),
            biomart_dataset.attribute("strand"),
            biomart_dataset.attribute("external_gene_name"),
            biomart_dataset.attribute("transcript_is_canonical"),
            biomart_dataset.attribute("transcript_biotype"),
        ],
        filters=[
            biomart_dataset.filter("transcript_biotype", value="protein_coding"),
            biomart_dataset.filter("ensembl_gene_id", value=",".join(gene_ids)),
        ],
    )
    genes["chrom"] = "chr" + genes["chromosome_name"].astype(str)
    genes = (
        genes.sort_values("transcript_is_canonical").groupby("ensembl_gene_id").first()
    )
    genes = genes.rename(
        columns={
            "transcript_start": "start",
            "transcript_end": "end",
            "external_gene_name": "symbol",
        }
    )
    genes.index.name = "gene"
    genes = genes.drop(
        [
            "chromosome_name",
            "transcript_biotype",
            "transcript_is_canonical",
            "start_position",
            "end_position",
        ],
        axis=1,
    )

    # remove genes without a symbol
    genes = genes.loc[~pd.isnull(genes["symbol"])]

    # remove genes on alternative contigs
    genes = genes.loc[~genes["chrom"].str.contains("_")]

    # remove genes on mitochondrial DNA
    genes = genes.loc[~genes["chrom"].isin(["chrMT", "chrM"])]

    # order according to gene_ids
    genes = genes.reindex(gene_ids)

    return genes

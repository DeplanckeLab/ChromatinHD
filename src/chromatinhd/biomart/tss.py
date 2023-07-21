import pandas as pd
from .dataset import Dataset


def get_canonical_transcripts(
    biomart_dataset: Dataset, gene_ids=None, chrom=None, start=None, end=None
):
    filters = []
    if gene_ids is not None:
        filters.append(
            biomart_dataset.filter("ensembl_gene_id", value=",".join(gene_ids))
        )
    if chrom is not None:
        filters.append(
            biomart_dataset.filter(
                "chromosome_name", value=str(chrom).replace("chr", "")
            )
        )
    if start is not None:
        filters.append(biomart_dataset.filter("start", value=str(start)))
    if end is not None:
        filters.append(biomart_dataset.filter("end", value=str(end)))
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
        filters=filters,
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

    # add ensembl gene id
    genes["ensembl_gene_id"] = genes.index

    return genes


def get_exons(biomart_dataset: Dataset, chrom, start, end):
    canonical_transcripts = get_canonical_transcripts(
        biomart_dataset, chrom=chrom, start=start, end=end
    )
    exons = biomart_dataset.get(
        [
            biomart_dataset.attribute("ensembl_gene_id"),
            biomart_dataset.attribute("exon_chrom_start"),
            biomart_dataset.attribute("exon_chrom_end"),
            biomart_dataset.attribute("genomic_coding_start"),
            biomart_dataset.attribute("genomic_coding_end"),
            biomart_dataset.attribute("ensembl_transcript_id"),
        ],
        filters=[
            biomart_dataset.filter("start", value=str(start)),
            biomart_dataset.filter("end", value=str(end)),
            biomart_dataset.filter(
                "chromosome_name", value=str(chrom).replace("chr", "")
            ),
            biomart_dataset.filter(
                "ensembl_transcript_id",
                value=",".join(canonical_transcripts["ensembl_transcript_id"]),
            ),
        ],
    )

    return exons

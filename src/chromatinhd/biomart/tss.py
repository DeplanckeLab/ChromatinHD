import pandas as pd
from .dataset import Dataset


def get_transcripts(
    biomart_dataset: Dataset,
    gene_ids=None,
    chrom=None,
    start=None,
    end=None,
    filter_chromosomes=True,
    filter_protein_coding=True,
) -> pd.DataFrame:
    """
    Get all canonical transcripts
    """
    batch_size = 100
    transcripts_ = []

    for i in range(0, len(gene_ids), batch_size):
        filters = []
        if gene_ids is not None:
            filters.append(biomart_dataset.filter("ensembl_gene_id", value=",".join(gene_ids[i : i + batch_size])))
        if chrom is not None:
            filters.append(biomart_dataset.filter("chromosome_name", value=str(chrom).replace("chr", "")))
        if start is not None:
            filters.append(biomart_dataset.filter("start", value=str(start)))
        if end is not None:
            filters.append(biomart_dataset.filter("end", value=str(end)))
        transcripts_.append(
            biomart_dataset.get(
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
        )
    transcripts = pd.concat(transcripts_)
    transcripts["chrom"] = "chr" + transcripts["chromosome_name"].astype(str)
    transcripts = transcripts.set_index("ensembl_transcript_id")
    transcripts = transcripts.rename(
        columns={
            "transcript_start": "start",
            "transcript_end": "end",
            "external_gene_name": "symbol",
        }
    )
    transcripts["tss"] = transcripts["start"] * (transcripts["strand"] == 1) + transcripts["end"] * (
        transcripts["strand"] == -1
    )

    transcripts = transcripts.drop(
        [
            "chromosome_name",
            "start_position",
            "end_position",
        ],
        axis=1,
    )

    # filter on protein coding
    if filter_protein_coding:
        transcripts = transcripts.loc[
            transcripts["transcript_biotype"].isin(["protein_coding", "protein_coding_CDS_not_defined"])
        ]

    if filter_chromosomes:
        transcripts = transcripts.loc[~transcripts["chrom"].isin(["chrM", "chrMT"])]
        transcripts = transcripts.loc[~transcripts["chrom"].str.contains("_")]
        transcripts = transcripts.loc[~transcripts["chrom"].str.contains("\.")]

    return transcripts


def get_canonical_transcripts(
    biomart_dataset: Dataset,
    gene_ids=None,
    chrom=None,
    start=None,
    end=None,
    filter_chromosomes: bool = True,
    filter_protein_coding: bool = True,
) -> pd.DataFrame:
    """
    Get all canonical transcripts
    """
    filters = []
    if gene_ids is not None:
        filters.append(biomart_dataset.filter("ensembl_gene_id", value=",".join(gene_ids)))
    if chrom is not None:
        filters.append(biomart_dataset.filter("chromosome_name", value=str(chrom).replace("chr", "")))
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
    genes = genes.sort_values("transcript_is_canonical").groupby("ensembl_gene_id").first()
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
            "start_position",
            "end_position",
        ],
        axis=1,
    )

    # add tss
    genes["tss"] = genes["start"] * (genes["strand"] == 1) + genes["end"] * (genes["strand"] == -1)

    # remove genes without a symbol
    genes = genes.loc[~pd.isnull(genes["symbol"])]

    if filter_chromosomes:
        genes = genes.loc[~genes["chrom"].str.contains("_")]
        genes = genes.loc[~genes["chrom"].isin(["chrMT", "chrM"])]

    # filter on protein coding
    if filter_protein_coding:
        genes = genes.loc[genes["transcript_biotype"].isin(["protein_coding", "protein_coding_CDS_not_defined"])]

    # order according to gene_ids
    genes = genes.reindex(gene_ids)

    # add ensembl gene id
    genes["ensembl_gene_id"] = genes.index

    return genes


def get_exons(biomart_dataset: Dataset, chrom, start, end):
    canonical_transcripts = get_canonical_transcripts(biomart_dataset, chrom=chrom, start=start, end=end)
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
            biomart_dataset.filter("chromosome_name", value=str(chrom).replace("chr", "")),
            biomart_dataset.filter(
                "ensembl_transcript_id",
                value=",".join(canonical_transcripts["ensembl_transcript_id"]),
            ),
        ],
    )

    return exons

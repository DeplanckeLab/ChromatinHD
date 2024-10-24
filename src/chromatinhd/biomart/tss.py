import pandas as pd
from .dataset import Dataset


def map_symbols(biomart_dataset: Dataset, symbols):
    """
    Map symbols to ensembl gene ids
    """
    mapping = biomart_dataset.get_batched(
        [
            biomart_dataset.attribute("ensembl_gene_id"),
            biomart_dataset.attribute("external_gene_name"),
        ],
        filters=[
            biomart_dataset.filter("external_gene_name", value=symbols),
        ],
        batch_size=500,
    )
    mapping = mapping.set_index("external_gene_name")
    return mapping


def get_genes(
    biomart_dataset: Dataset,
) -> pd.DataFrame:
    """
    Get all canonical transcripts
    """
    genes = biomart_dataset.get(
        [
            biomart_dataset.attribute("ensembl_gene_id"),
            biomart_dataset.attribute("external_gene_name"),
        ],
    )

    return genes


def get_transcripts(
    biomart_dataset: Dataset,
    gene_ids=None,
    symbols=None,
    chrom=None,
    start=None,
    end=None,
    filter_chromosomes=True,
    filter_protein_coding=True,
    batch_size=100,
) -> pd.DataFrame:
    """
    Get all canonical transcripts

    Parameters:
        dataset:
            A biomart dataset
        gene_ids:
            List of ensembl gene ids
        symbols:
            List of gene symbols
        chrom:
            Chromosome
        start:
            Start position
        end:
            End position
        filter_chromosomes:
            Filter out irregular chromosomes
        filter_protein_coding:
            Filter out non-protein coding transcripts
        batch_size:
            Batch size for fetching. Reduce if you get a timeout error
    """

    filters = []
    if gene_ids is not None:
        filters.append(biomart_dataset.filter("ensembl_gene_id", value=gene_ids))
    if symbols is not None:
        filters.append(biomart_dataset.filter("external_gene_name", value=symbols))
    if chrom is not None:
        filters.append(biomart_dataset.filter("chromosome_name", value=str(chrom).replace("chr", "")))
    if start is not None:
        filters.append(biomart_dataset.filter("start", value=str(start)))
    if end is not None:
        filters.append(biomart_dataset.filter("end", value=str(end)))

    attributes = [
        biomart_dataset.attribute("ensembl_gene_id"),
        biomart_dataset.attribute("transcript_start"),
        biomart_dataset.attribute("transcript_end"),
        biomart_dataset.attribute("end_position"),
        biomart_dataset.attribute("start_position"),
        biomart_dataset.attribute("ensembl_transcript_id"),
        biomart_dataset.attribute("chromosome_name"),
        biomart_dataset.attribute("strand"),
        biomart_dataset.attribute("external_gene_name"),
        biomart_dataset.attribute("transcript_biotype"),
    ]

    if len(filters) == 1:
        transcripts = biomart_dataset.get_batched(
            attributes,
            filters=filters,
        )
    else:
        transcripts = biomart_dataset.get(attributes, filters=filters)
    transcripts["chrom"] = "chr" + transcripts["chromosome_name"].astype(str)
    transcripts = transcripts.set_index("ensembl_transcript_id")
    transcripts = transcripts.rename(
        columns={
            "transcript_start": "start",
            "transcript_end": "end",
            "external_gene_name": "symbol",
            "ensembl_transcript_id": "transcript",
        }
    )
    transcripts["tss"] = transcripts["start"] * (transcripts["strand"] == 1) + transcripts["end"] * (transcripts["strand"] == -1)

    transcripts = transcripts.drop(
        [
            "chromosome_name",
            "start_position",
            "end_position",
        ],
        axis=1,
    )
    transcripts.index.name = "transcript"

    # filter on protein coding
    if filter_protein_coding:
        transcripts = transcripts.loc[transcripts["transcript_biotype"].isin(["protein_coding", "protein_coding_CDS_not_defined"])]

    if filter_chromosomes:
        transcripts = transcripts.loc[~transcripts["chrom"].isin(["chrM", "chrMT"])]
        transcripts = transcripts.loc[~transcripts["chrom"].str.contains("_")]
        transcripts = transcripts.loc[~transcripts["chrom"].str.contains(r"\.")]

    # sort by gene_ids
    transcripts["ensembl_gene_id"] = pd.Categorical(transcripts["ensembl_gene_id"], categories=gene_ids, ordered=True)

    return transcripts


def get_canonical_transcripts(
    biomart_dataset: Dataset,
    gene_ids=None,
    symbols=None,
    chrom=None,
    start=None,
    end=None,
    filter_canonical: bool = True,
    filter_chromosomes: bool = True,
    filter_protein_coding: bool = True,
    use_cache=True,
) -> pd.DataFrame:
    """
    Get all canonical transcripts
    """
    filters = []
    if gene_ids is not None:
        filters.append(biomart_dataset.filter("ensembl_gene_id", value=gene_ids))
    if symbols is not None:
        filters.append(biomart_dataset.filter("external_gene_name", value=symbols))
    if chrom is not None:
        filters.append(biomart_dataset.filter("chromosome_name", value=str(chrom).replace("chr", "")))
    if start is not None:
        filters.append(biomart_dataset.filter("start", value=str(start)))
    if end is not None:
        filters.append(biomart_dataset.filter("end", value=str(end)))

    # do not filter on canonical if not available
    if filter_canonical and "transcript_is_canonical" not in biomart_dataset.list_attributes().index:
        filter_canonical = False

    attributes = [
        biomart_dataset.attribute("ensembl_gene_id"),
        biomart_dataset.attribute("transcript_start"),
        biomart_dataset.attribute("transcript_end"),
        biomart_dataset.attribute("end_position"),
        biomart_dataset.attribute("start_position"),
        biomart_dataset.attribute("ensembl_transcript_id"),
        biomart_dataset.attribute("chromosome_name"),
        biomart_dataset.attribute("strand"),
        biomart_dataset.attribute("external_gene_name"),
        biomart_dataset.attribute("transcript_biotype"),
    ]
    if filter_canonical:
        attributes.append(biomart_dataset.attribute("transcript_is_canonical"))

    if len(filters) == 1:
        genes = biomart_dataset.get_batched(
            attributes,
            filters=filters,
            use_cache=use_cache,
        )
    else:
        genes = biomart_dataset.get(
            attributes,
            filters=filters,
            use_cache=use_cache,
        )
    genes["chrom"] = "chr" + genes["chromosome_name"].astype(str)

    # filter on irregular chromosomes
    if filter_chromosomes:
        genes = genes.loc[~genes["chrom"].str.contains("_")]
        genes = genes.loc[~genes["chrom"].str.contains(".", regex=False)]
        genes = genes.loc[~genes["chrom"].isin(["chrMT", "chrM"])]

    # filter on protein coding
    if filter_protein_coding:
        genes = genes.loc[genes["transcript_biotype"].isin(["protein_coding", "protein_coding_CDS_not_defined"])]

    # filter canonical
    # if we do not have the canonical attribute, we use the largest (protein coding) transcript
    if filter_canonical:
        genes = genes.sort_values("transcript_is_canonical").groupby("ensembl_gene_id").first()
    else:
        genes["length"] = genes["transcript_end"] - genes["transcript_start"]
        genes = genes.sort_values("length").groupby("ensembl_gene_id").last()
    genes = genes.rename(
        columns={
            "transcript_start": "start",
            "transcript_end": "end",
            "external_gene_name": "symbol",
            "ensembl_transcript_id": "transcript",
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

    # order according to gene_ids
    genes = genes.reindex(gene_ids)

    # add ensembl gene id
    genes["ensembl_gene_id"] = genes.index

    return genes


def get_exons(biomart_dataset: Dataset, chrom, start, end):
    canonical_transcripts = get_canonical_transcripts(biomart_dataset, chrom=chrom, start=start, end=end)
    if len(canonical_transcripts) == 0:
        return pd.DataFrame()
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
                value=canonical_transcripts["transcript"],
            ),
        ],
    )

    return exons

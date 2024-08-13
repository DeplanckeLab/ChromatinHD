from .dataset import Dataset
import numpy as np


def get_orthologs(biomart_dataset: Dataset, gene_ids, organism="mmusculus"):
    """
    Map ensembl gene ids to orthologs in another organism
    """

    gene_ids_to_map = np.unique(gene_ids)
    mapping = biomart_dataset.get_batched(
        [
            biomart_dataset.attribute("ensembl_gene_id"),
            biomart_dataset.attribute("external_gene_name"),
            biomart_dataset.attribute(f"{organism}_homolog_ensembl_gene"),
            biomart_dataset.attribute(f"{organism}_homolog_associated_gene_name"),
        ],
        filters=[
            biomart_dataset.filter("ensembl_gene_id", value=gene_ids_to_map),
        ],
    )
    mapping = mapping.groupby("ensembl_gene_id").first()

    return mapping[f"{organism}_homolog_ensembl_gene"].reindex(gene_ids).values

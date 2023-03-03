import numpy as np
import dataclasses
import torch


@dataclasses.dataclass
class Minibatch:
    genes_oi: np.ndarray


@dataclasses.dataclass
class Data:
    variantxgene_to_gene: torch.Tensor
    variantxgene_ixs: torch.Tensor
    local_variant_to_local_variantxgene_selector: torch.Tensor
    variantxgene_to_local_gene: torch.Tensor

    expression: torch.Tensor
    genotypes: torch.Tensor

    variants_oi: np.ndarray
    genes_oi: np.ndarray
    clusters_oi: np.ndarray

    def to(self, device):
        for field_name, field in self.__dataclass_fields__.items():
            if field.type is torch.Tensor:
                self.__setattr__(
                    field_name, self.__getattribute__(field_name).to(device)
                )
        return self

    @property
    def n_clusters(self):
        return len(self.clusters_oi)

    @property
    def n_variants(self):
        return len(self.variants_oi)


class Loader:
    def __init__(self, transcriptome, genotype, gene_variants_mapping):
        # map genes to variantxgene
        self.gene_variants_mapping = gene_variants_mapping
        self.gene_variantxgene_ix_mapping = []
        i = 0
        for variants in gene_variants_mapping:
            self.gene_variantxgene_ix_mapping.append(np.arange(i, i + len(variants)))
            i += len(variants)
        self.n_variantxgenes = i

        self.variant_ix_to_local_variant_ix = np.zeros(
            len(genotype.variants_info), dtype=int
        )

        self.expression = transcriptome.X
        self.genotypes = genotype.genotypes

    def load(self, minibatch: Minibatch):
        # this will map a variant_ix to a local_variant_ix
        # initially all -1, given that the variant is not (yet) in the variants_oi
        self.variant_ix_to_local_variant_ix[:] = -1

        # this contains all variants that are selected, in order of their local_variant_ix
        variants_oi = []

        # this maps each variantxgene combination to the (local)_gene_ix
        variantxgene_to_gene = []
        variantxgene_to_local_gene = []

        # contains the variantxgene_ixs
        variantxgene_ixs = []

        # this will map a local_variant_ix to a local_variantxgene_x
        # e.g. if we have calculated something for all local variants, we can then easily reshape to have all variantxgene combinations
        # variantxgene_to_gene can then be used to retrieve the exact gene to which this variantxgene maps
        local_variant_to_local_variantxgene_selector = []

        for local_gene_ix, gene_ix in enumerate(minibatch.genes_oi):
            gene_variant_ixs = self.gene_variants_mapping[gene_ix]
            unknown_variant_ixs = gene_variant_ixs[
                self.variant_ix_to_local_variant_ix[gene_variant_ixs] == -1
            ]
            self.variant_ix_to_local_variant_ix[unknown_variant_ixs] = np.arange(
                len(unknown_variant_ixs)
            ) + len(variants_oi)

            variants_oi.extend(unknown_variant_ixs)

            local_variant_to_local_variantxgene_selector.extend(
                self.variant_ix_to_local_variant_ix[gene_variant_ixs]
            )
            variantxgene_to_gene.extend([gene_ix] * len(gene_variant_ixs))
            variantxgene_to_local_gene.extend([local_gene_ix] * len(gene_variant_ixs))
            variantxgene_ixs.extend(self.gene_variantxgene_ix_mapping[gene_ix])

        variants_oi = np.array(variants_oi)
        local_variant_to_local_variantxgene_selector = np.array(
            local_variant_to_local_variantxgene_selector
        )
        variantxgene_to_gene = np.array(variantxgene_to_gene)
        variantxgene_to_local_gene = np.array(variantxgene_to_local_gene)
        variantxgene_ixs = np.array(variantxgene_ixs)

        # expression
        expression = self.expression[:, :, minibatch.genes_oi]

        # genotypes
        genotypes = self.genotypes[:, variants_oi]

        return Data(
            variantxgene_to_gene=torch.from_numpy(variantxgene_to_gene),
            variantxgene_ixs=torch.from_numpy(variantxgene_ixs),
            expression=torch.from_numpy(expression),
            genotypes=torch.from_numpy(genotypes),
            variants_oi=torch.from_numpy(variants_oi),
            genes_oi=torch.from_numpy(minibatch.genes_oi),
            local_variant_to_local_variantxgene_selector=torch.from_numpy(
                local_variant_to_local_variantxgene_selector
            ),
            variantxgene_to_local_gene=torch.from_numpy(variantxgene_to_local_gene),
            clusters_oi=None,
        )

import numpy as np
import dataclasses
import torch
import copy

from chromatinhd.utils import indices_to_indptr


@dataclasses.dataclass
class Minibatch:
    genes_oi: np.ndarray


def create_bins_ordered(
    genes,
    n_genes_step=300,
    use_all=True,
    rg=None,
    permute_genes=False,
):
    """
    Creates bins of genes
    A number of cell and gene bins are created first, and all combinations of these bins make up the
    """
    if rg is None:
        rg = np.random.RandomState()
    if permute_genes:
        genes = rg.permutation(genes)
    genes = np.array(genes)

    gene_cuts = [*np.arange(0, len(genes), step=n_genes_step)]
    if use_all:
        gene_cuts.append(len(genes))
    gene_bins = [genes[a:b] for a, b in zip(gene_cuts[:-1], gene_cuts[1:])]

    bins = []
    for genes_oi in gene_bins:
        bins.append(
            Minibatch(
                genes_oi=genes_oi,
            )
        )
    return bins


@dataclasses.dataclass
class Data:
    relative_coordinates: torch.Tensor
    local_cluster_ixs: torch.Tensor
    local_variant_to_local_variantxgene_selector: torch.Tensor
    variantxgene_to_gene: torch.Tensor
    variantxgene_to_local_gene: torch.Tensor
    variantxgene_ixs: torch.Tensor
    local_clusterxvariant_indptr: torch.Tensor

    expression: torch.Tensor
    genotypes: torch.Tensor

    window_size: np.ndarray
    variants_oi: np.ndarray
    genes_oi: np.ndarray
    clusters_oi: np.ndarray

    variantxgene_tss_distances: torch.Tensor

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
    def __init__(
        self,
        transcriptome,
        genotype,
        fragments,
        gene_variants_mapping,
        variantxgenes_info,
        window_size=5000,
    ):
        # map genes to variantxgene
        self.gene_variants_mapping = gene_variants_mapping
        self.gene_variantxgene_ix_mapping = []
        i = 0
        for variants in gene_variants_mapping:
            self.gene_variantxgene_ix_mapping.append(np.arange(i, i + len(variants)))
            i += len(variants)
        self.n_variantxgenes = i

        assert self.n_variantxgenes == len(variantxgenes_info)

        self.variantxgene_tss_distances = variantxgenes_info["tss_distance"].values

        self.variant_ix_to_local_variant_ix = np.zeros(
            len(genotype.variants_info), dtype=int
        )

        self.expression = transcriptome.X
        self.genotypes = genotype.genotypes

        # fragments

        # create bounds from chromosome positions
        # the windows will be bounded by these positions
        self.bounds = np.hstack(
            [
                fragments.chromosomes["position_start"].values,
                fragments.chromosomes["position_end"].values[[-1]],
            ]
        )

        variant_positions = genotype.variants_info["position"].values
        assert variant_positions.max() < self.bounds.max()
        assert variant_positions.min() > self.bounds.min()

        self.variant_positions = variant_positions

        # cache upper and lower bounds
        self.variant_upper_bounds = self.bounds[
            np.searchsorted(self.bounds, self.variant_positions)
        ]
        self.variant_lower_bounds = self.bounds[
            np.searchsorted(self.bounds, self.variant_positions) - 1
        ]

        # store window size
        self.window_size = window_size

        # store fragment data
        self.chunkcoords = fragments.chunkcoords.numpy()
        self.relcoords = fragments.relcoords.numpy()
        self.chunk_size = fragments.chunk_size
        self.chunkcoords_indptr = fragments.chunkcoords_indptr.numpy()
        self.clusters = fragments.clusters.numpy()

        self.n_variants = len(genotype.variants_info)
        self.n_clusters = len(fragments.clusters_info)
        self.clusters_oi = np.arange(self.n_clusters)

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

        # get cut site coordinates
        relative_coordinates = []
        cluster_ixs = []
        variant_ixs = []
        local_variant_ixs = []

        for local_variant_ix, variant_ix in enumerate(variants_oi):
            position = self.variant_positions[variant_ix]
            upper_bound = self.variant_upper_bounds[variant_ix]
            lower_bound = self.variant_lower_bounds[variant_ix]

            window_start = max(lower_bound, position - self.window_size // 2)
            window_end = min(upper_bound, position + self.window_size // 2)

            window_chunks_start = window_start // self.chunk_size
            window_chunks_end = (window_end // self.chunk_size) + 1

            chunks_from, chunks_to = (
                self.chunkcoords_indptr[window_chunks_start],
                self.chunkcoords_indptr[window_chunks_end],
            )

            clusters_oi = self.clusters[chunks_from:chunks_to]

            # sorting is necessary here as the original data is not sorted
            # and because sorted data (acording to variantxcluster) will be expected downstream by torch_scatter
            # this is probably a major slow down
            # it might be faster to not sort here, and use torch_scatter scatter operations downstream
            order = np.argsort(clusters_oi)

            relative_coordinates.append(
                (
                    self.chunkcoords[chunks_from:chunks_to] * self.chunk_size
                    + self.relcoords[chunks_from:chunks_to]
                    - position
                )[order]
            )
            cluster_ixs.append(clusters_oi[order])
            variant_ixs.append(np.repeat(variant_ix, chunks_to - chunks_from))
            local_variant_ixs.append(
                np.repeat(local_variant_ix, chunks_to - chunks_from)
            )
        relative_coordinates = np.hstack(relative_coordinates)
        cluster_ixs = np.hstack(cluster_ixs)
        variant_ixs = np.hstack(variant_ixs)
        local_variant_ixs = np.hstack(local_variant_ixs)
        local_cluster_ixs = cluster_ixs

        # cluster-variant mapping
        n_variants = len(variants_oi)
        local_clusterxvariant_indptr = indices_to_indptr(
            local_cluster_ixs * n_variants + local_variant_ixs,
            n_variants * self.n_clusters,
        )

        # expression
        expression = self.expression[:, :, minibatch.genes_oi]

        # genotypes
        genotypes = self.genotypes[:, variants_oi]

        # tss distance
        variantxgene_tss_distances = self.variantxgene_tss_distances[variantxgene_ixs]

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
            relative_coordinates=torch.from_numpy(relative_coordinates),
            clusters_oi=self.clusters_oi,
            local_cluster_ixs=torch.from_numpy(local_cluster_ixs),
            local_clusterxvariant_indptr=torch.from_numpy(local_clusterxvariant_indptr),
            window_size=self.window_size,
            variantxgene_tss_distances=torch.from_numpy(variantxgene_tss_distances),
        )

    def copy(self):
        new_copy = copy.copy(self)
        new_copy.variant_ix_to_local_variant_ix = copy.copy(
            new_copy.variant_ix_to_local_variant_ix
        )
        return new_copy

import torch
import numpy as np
import dataclasses
from functools import cached_property

import chromatinhd


@dataclasses.dataclass
class Data:
    relative_coordinates: torch.Tensor
    local_cluster_ixs: torch.Tensor
    local_clusterxvariant_indptr: torch.Tensor
    cluster_cut_lib: torch.Tensor
    local_variant_to_local_variantxgene_reshaper: torch.Tensor
    variantxgene_to_gene: torch.Tensor
    variantxgene_to_local_gene: torch.Tensor
    variantxgene_ixs: torch.Tensor

    expression: torch.Tensor
    genotypes: torch.Tensor

    window_size: np.ndarray
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


class ChunkFragments:
    cellxgene_batch_size: int

    preloaded = False

    out_coordinates: torch.Tensor
    out_genemapping: torch.Tensor
    out_local_cellxgene_ix: torch.Tensor

    n_genes: int

    def __init__(
        self,
        fragments: chromatinhd.data.Fragments,
        variant_positions,
        window_size,
        clusters_oi=None,
        n_fragment_per_cellxgene: int = None,
    ):
        # store fragment data
        self.chunkcoords = fragments.chunkcoords.numpy().astype(np.int64)
        self.relcoords = fragments.relcoords.numpy()
        self.chunk_size = fragments.chunk_size
        self.chunkcoords_indptr = fragments.chunkcoords_indptr.numpy()
        self.clusters = fragments.clusters.numpy()

        # store the library size
        self.cluster_cut_lib = torch.bincount(
            fragments.clusters, minlength=len(fragments.clusters_info)
        )

        # create bounds from chromosome positions
        # the windows will be bounded by these positions
        self.bounds = np.hstack(
            [
                fragments.chromosomes["position_start"].values,
                fragments.chromosomes["position_end"].values[[-1]],
            ]
        )

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

    def load(self, minibatch):
        if not self.preloaded:
            self.preload()

        # optional filtering based on fragments_oi
        coordinates = self.coordinates
        genemapping = self.genemapping
        cellxgene_indptr = self.cellxgene_indptr

        minibatch.cellxgene_oi = cell_gene_to_cellxgene(
            minibatch.cells_oi, minibatch.genes_oi, self.n_genes
        )

        assert len(minibatch.cellxgene_oi) <= self.cellxgene_batch_size, (
            len(minibatch.cellxgene_oi),
            self.cellxgene_batch_size,
        )
        n_fragments = chromatinhd.loaders.extraction.fragments.extract_fragments(
            minibatch.cellxgene_oi,
            cellxgene_indptr,
            coordinates,
            genemapping,
            self.out_coordinates.numpy(),
            self.out_genemapping.numpy(),
            self.out_local_cellxgene_ix.numpy(),
        )
        if n_fragments > self.fragment_buffer_size:
            raise ValueError("n_fragments is too large for the current buffer size")

        if n_fragments == 0:
            n_fragments = 1
        self.out_coordinates.resize_((n_fragments, 2))
        self.out_genemapping.resize_((n_fragments))
        self.out_local_cellxgene_ix.resize_((n_fragments))

        return Data(
            local_cellxgene_ix=self.out_local_cellxgene_ix,
            coordinates=self.out_coordinates,
            n_fragments=n_fragments,
            genemapping=self.out_genemapping,
            window=self.window,
            n_total_genes=self.n_genes,
            **minibatch.items(),
        )

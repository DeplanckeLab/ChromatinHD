import chromatinhd.data.motifscan
import dataclasses
import torch
from chromatinhd.utils.numpy import indptr_to_indices
import numpy as np


@dataclasses.dataclass
class Result:
    indices: torch.Tensor
    positions: torch.Tensor
    scores: torch.Tensor
    local_genexmotif_ix: torch.Tensor
    local_gene_ix: torch.Tensor
    n_genes: int

    def to(self, device):
        for field_name, field in self.__dataclass_fields__.items():
            if field.type is torch.Tensor:
                self.__setattr__(field_name, self.__getattribute__(field_name).to(device))
        return self


class Motifs:
    """
    Provides motifscan data for a minibatch.
    """

    def __init__(
        self,
        motifscan: chromatinhd.data.motifscan.Motifscan,
    ):
        self.motifscan = motifscan
        self.region_width = motifscan.regions.window[1] - motifscan.regions.window[0]
        self.n_motifs = motifscan.motifs.shape[0]

    def load(self, minibatch):
        local_genexmotif_ix = []
        scores = []
        positions = []
        indices = []
        local_gene_ix = []
        for i, gene_ix in enumerate(minibatch.genes_oi):
            indptr_start = gene_ix * self.region_width
            indptr_end = (gene_ix + 1) * self.region_width
            indices_gene = self.motifscan.indices[
                self.motifscan.indptr[indptr_start] : self.motifscan.indptr[indptr_end]
            ]
            positions_gene = (
                indptr_to_indices(self.motifscan.indptr[indptr_start : indptr_end + 1])
                + self.motifscan.regions.window[0]
            )
            scores_gene = self.motifscan.scores[self.motifscan.indptr[indptr_start] : self.motifscan.indptr[indptr_end]]

            indices.append(indices_gene)
            positions.append(positions_gene)
            scores.append(scores_gene)

            local_genexmotif_ix_gene = np.ones_like(indices_gene, dtype=np.int32) * i * self.n_motifs + indices_gene
            local_genexmotif_ix.append(local_genexmotif_ix_gene)
            local_gene_ix_gene = np.ones_like(indices_gene, dtype=np.int32) * i
            local_gene_ix.append(local_gene_ix_gene)

        indices = torch.from_numpy(np.concatenate(indices)).contiguous()
        positions = torch.from_numpy(np.concatenate(positions)).contiguous()
        scores = torch.from_numpy(np.concatenate(scores)).contiguous()
        local_genexmotif_ix = torch.from_numpy(np.concatenate(local_genexmotif_ix)).contiguous()
        local_gene_ix = torch.from_numpy(np.concatenate(local_gene_ix)).contiguous()
        return Result(indices, positions, scores, local_genexmotif_ix, local_gene_ix, n_genes=minibatch.n_genes)

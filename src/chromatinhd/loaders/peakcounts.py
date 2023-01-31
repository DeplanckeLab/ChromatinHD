import torch
import numpy as np
import dataclasses

@dataclasses.dataclass
class Result():
    counts:np.ndarray
    cells_oi:np.ndarray
    genes_oi:np.ndarray
    @property
    def n_cells(self):
        return len(self.cells_oi)

    @property
    def n_genes(self):
        return len(self.genes_oi)

    def to(self, device):
        for field_name, field in self.__dataclass_fields__.items():
            if field.type is torch.Tensor:
                self.__setattr__(field_name, self.__getattribute__(field_name).to(device))
        return self

    @property
    def genes_oi_torch(self):
        return torch.from_numpy(self.genes_oi).to(self.coordinates.device)

    @property
    def cells_oi_torch(self):
        return torch.from_numpy(self.cells_oi).to(self.coordinates.device)

class PeakcountsResult(Result):
    pass

class Peakcounts():
    def __init__(self, fragments, peakcounts):
        self.peakcounts = peakcounts
        assert "gene_ix" in peakcounts.peaks.columns
        var = peakcounts.var
        var["ix"] = np.arange(peakcounts.var.shape[0])
        peakcounts.var = var
        assert "ix" in peakcounts.var.columns

        assert peakcounts.counts.shape[1] == peakcounts.var.shape[0]

        self.gene_peak_mapping = []
        i = 0
        cur_gene_ix = -1
        for peak_ix, gene_ix in zip(peakcounts.var["ix"][peakcounts.peaks.index].values, peakcounts.peaks["gene_ix"]):
            while gene_ix != cur_gene_ix:
                self.gene_peak_mapping.append([])
                cur_gene_ix += 1
            self.gene_peak_mapping[-1].append(peak_ix)

        
    def load(self, minibatch):
        peak_ixs = np.concatenate([self.gene_peak_mapping[gene_ix] for gene_ix in minibatch.genes_oi])
        counts = self.peakcounts.counts[minibatch.cells_oi, :][:, peak_ixs]
        
        return PeakcountsResult(
            counts = counts,
            **minibatch.items()
        )

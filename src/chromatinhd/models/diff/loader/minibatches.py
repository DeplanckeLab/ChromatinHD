import numpy as np
import dataclasses
import itertools
import math
import torch


@dataclasses.dataclass
class Minibatch:
    cells_oi: np.ndarray
    genes_oi: np.ndarray
    phase: str = "train"
    device: str = "cpu"

    def items(self):
        return {"cells_oi": self.cells_oi, "genes_oi": self.genes_oi}

    def filter_genes(self, genes):
        genes_oi = self.genes_oi[genes[self.genes_oi]]

        return Minibatch(
            cells_oi=self.cells_oi,
            genes_oi=genes_oi,
            phase=self.phase,
        )

    def to(self, device):
        self.device = device

    @property
    def genes_oi_torch(self):
        return torch.from_numpy(self.genes_oi).to(self.device)

    @property
    def cells_oi_torch(self):
        return torch.from_numpy(self.genes_oi).to(self.device)

    @property
    def n_cells(self):
        return len(self.cells_oi)

    @property
    def n_genes(self):
        return len(self.genes_oi)


class Minibatcher:
    def __init__(
        self,
        cells,
        genes,
        n_cells_step,
        n_genes_step,
        use_all_cells=False,
        use_all_genes=True,
        permute_cells=True,
        permute_genes=False,
    ):
        self.cells = cells
        if not isinstance(genes, np.ndarray):
            genes = np.array(genes)
        self.genes = genes
        self.n_genes = len(genes)
        self.n_cells_step = n_cells_step
        self.n_genes_step = n_genes_step

        self.permute_cells = permute_cells
        self.permute_genes = permute_genes

        self.use_all_cells = use_all_cells or len(cells) < n_cells_step
        self.use_all_genes = use_all_genes or len(genes) < n_genes_step

        self.cellxgene_batch_size = n_cells_step * n_genes_step

        # calculate length
        n_cells = len(cells)
        n_genes = len(genes)
        if self.use_all_cells:
            n_cell_bins = math.ceil(n_cells / n_cells_step)
        else:
            n_cell_bins = math.floor(n_cells / n_cells_step)
        if self.use_all_genes:
            n_gene_bins = math.ceil(n_genes / n_genes_step)
        else:
            n_gene_bins = math.floor(n_genes / n_genes_step)
        self.length = n_cell_bins * n_gene_bins

        self.i = 0

    def __len__(self):
        return self.length

    def __iter__(self):
        self.rg = np.random.RandomState(self.i)

        if self.permute_cells:
            cells = self.rg.permutation(self.cells)
        else:
            cells = self.cells
        if self.permute_genes:
            genes = self.rg.permutation(self.genes)
        else:
            genes = self.genes

        gene_cuts = [*np.arange(0, len(genes), step=self.n_genes_step)]
        if self.use_all_genes:
            gene_cuts.append(len(genes))
        gene_bins = [genes[a:b] for a, b in zip(gene_cuts[:-1], gene_cuts[1:])]

        cell_cuts = [*np.arange(0, len(cells), step=self.n_cells_step)]
        if self.use_all_cells:
            cell_cuts.append(len(cells))
        cell_bins = [cells[a:b] for a, b in zip(cell_cuts[:-1], cell_cuts[1:])]

        for cells_oi, genes_oi in itertools.product(cell_bins, gene_bins):
            yield Minibatch(cells_oi=cells_oi, genes_oi=genes_oi)

        self.i += 1

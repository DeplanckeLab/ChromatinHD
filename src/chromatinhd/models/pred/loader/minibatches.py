import numpy as np
import dataclasses
import itertools
import math
import torch
import random


@dataclasses.dataclass
class Minibatch:
    cells_oi: np.ndarray
    regions_oi: np.ndarray
    phase: str = "train"
    device: str = "cpu"

    def items(self):
        return {"cells_oi": self.cells_oi, "regions_oi": self.regions_oi}

    def filter_regions(self, regions):
        regions_oi = self.regions_oi[regions[self.regions_oi]]

        return Minibatch(
            cells_oi=self.cells_oi,
            regions_oi=regions_oi,
            phase=self.phase,
        )

    def to(self, device):
        self.device = device

    @property
    def regions_oi_torch(self):
        return torch.from_numpy(self.regions_oi).to(self.device)

    @property
    def cells_oi_torch(self):
        return torch.from_numpy(self.regions_oi).to(self.device)

    @property
    def n_cells(self):
        return len(self.cells_oi)

    @property
    def n_regions(self):
        return len(self.regions_oi)


class Minibatcher:
    def __init__(
        self,
        cells,
        regions,
        n_cells_step,
        n_regions_step,
        use_all_cells=False,
        use_all_regions=True,
        permute_cells=True,
        permute_regions=False,
    ):
        self.cells = cells
        if not isinstance(regions, np.ndarray):
            regions = np.array(regions)
        self.regions = regions
        self.n_regions = len(regions)
        self.n_cells_step = n_cells_step
        self.n_regions_step = n_regions_step

        self.permute_cells = permute_cells
        self.permute_regions = permute_regions

        self.use_all_cells = use_all_cells or len(cells) < n_cells_step
        self.use_all_regions = use_all_regions or len(regions) < n_regions_step

        self.cellxregion_batch_size = n_cells_step * n_regions_step

        # calculate length
        n_cells = len(cells)
        n_regions = len(regions)
        if self.use_all_cells:
            n_cell_bins = math.ceil(n_cells / n_cells_step)
        else:
            n_cell_bins = math.floor(n_cells / n_cells_step)
        if self.use_all_regions:
            n_region_bins = math.ceil(n_regions / n_regions_step)
        else:
            n_region_bins = math.floor(n_regions / n_regions_step)
        self.length = n_cell_bins * n_region_bins

        self.i = 0

        self.rg = None

    def __len__(self):
        return self.length

    def __iter__(self):
        self.rg = np.random.RandomState(self.i)

        if self.permute_cells:
            cells = self.rg.permutation(self.cells)
        else:
            cells = self.cells
        if self.permute_regions:
            regions = self.rg.permutation(self.regions)
        else:
            regions = self.regions

        region_cuts = [*np.arange(0, len(regions), step=self.n_regions_step)]
        if self.use_all_regions:
            region_cuts.append(len(regions))
        region_bins = [regions[a:b] for a, b in zip(region_cuts[:-1], region_cuts[1:])]

        cell_cuts = [*np.arange(0, len(cells), step=self.n_cells_step)]
        if self.use_all_cells:
            cell_cuts.append(len(cells))
        cell_bins = [cells[a:b] for a, b in zip(cell_cuts[:-1], cell_cuts[1:])]

        product = itertools.product(cell_bins, region_bins)

        if self.permute_cells and self.permute_regions:
            rng = random.Random(self.i)
            product = list(product)
            rng.shuffle(list(product))

        for cells_oi, regions_oi in product:
            yield Minibatch(cells_oi=cells_oi, regions_oi=regions_oi)

        self.i += 1

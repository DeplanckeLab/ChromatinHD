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

    @property
    def genes_oi(self):
        return self.regions_oi


class Minibatcher:
    """
    Provides minibatches of cells and regions to load.

    Examples:
        >>> cells = np.arange(100)
        >>> regions = np.arange(100)
        >>> minibatcher = Minibatcher(cells=cells, regions=regions, n_cells_step=10, n_regions_step=10)
        >>> for minibatch in minibatcher:
        >>>     pass
    """

    def __init__(
        self,
        cells: np.ndarray,
        regions: np.ndarray,
        n_cells_step: int,
        n_regions_step: int,
        use_all_cells: bool = False,
        use_all_regions: bool = True,
        permute_cells: bool = True,
        permute_regions: bool = False,
        max_length: int = None,
    ):
        """
        Parameters:
            cells:
                Cells to load.
            regions:
                Regions to load.
            n_cells_step:
                Number of cells to load per minibatch.
            n_regions_step:
                Number of regions to load per minibatch.
            use_all_cells:
                Whether to provide all in each epoch. This may lead to minibatches of different size.
            use_all_regions:
                Whether to provide all regions in each epoch. This may lead to minibatches of different size.
            permute_cells:
                Whether to permute cells in each epoch.
            permute_regions:
                Whether to permute regions in each epoch.
        """
        self.cells = cells
        if not isinstance(regions, np.ndarray):
            regions = np.array(regions)
        self.regions = regions
        self.n_regions = len(regions)

        if n_cells_step > len(cells):
            n_cells_step = len(cells)
        if n_regions_step > len(regions):
            n_regions_step = len(regions)

        self.n_cells_step = n_cells_step
        self.n_regions_step = n_regions_step

        self.permute_cells = permute_cells
        self.permute_regions = permute_regions

        self.use_all_cells = use_all_cells or (len(cells) <= n_cells_step)
        self.use_all_regions = use_all_regions or (len(regions) <= n_regions_step)

        self.cellxregion_batch_size = n_cells_step * n_regions_step

        self.i = 0

        self.rg = None

        self._setup_bins()

    def _setup_bins(self):
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

        self.length = len(cell_bins) * len(region_bins)

        product = itertools.product(cell_bins, region_bins)

        if self.permute_cells and self.permute_regions:
            rng = random.Random(self.i)
            product = list(product)
            rng.shuffle(list(product))
        return product

    def __len__(self):
        return self.length

    def __iter__(self):
        product = self._setup_bins()
        for cells_oi, regions_oi in product:
            yield Minibatch(cells_oi=cells_oi, regions_oi=regions_oi)

        self.i += 1

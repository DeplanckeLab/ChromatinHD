import numpy as np
import pandas as pd

from chromatinhd.flow import (
    Flow,
    Stored,
)

from chromatinhd.data.fragments import Fragments


class Folds(Flow):
    """
    Folds of multiple cell and reion combinations
    """

    folds: dict = Stored()
    """The folds"""

    def sample_cells(
        self,
        fragments: Fragments,
        n_folds: int,
        n_repeats: int = 1,
        overwrite: bool = False,
        seed: int = 1,
    ):
        """
        Sample cells and regions into folds

        Parameters:
            fragments:
                the fragments
            n_folds:
                the number of folds
            n_repeats:
                the number of repeats
            overwrite:
                whether to overwrite existing folds
        """
        if not overwrite and self.get("folds").exists(self):
            return self

        folds = []

        for repeat_ix in range(n_repeats):
            generator = np.random.RandomState(repeat_ix * seed)

            cells_all = generator.permutation(fragments.n_cells)

            cell_bins = np.floor((np.arange(len(cells_all)) / (len(cells_all) / n_folds)))

            for i in range(n_folds):
                cells_train = cells_all[cell_bins != i]
                cells_validation_test = cells_all[cell_bins == i]
                cells_validation = cells_validation_test[: (len(cells_validation_test) // 2)]
                cells_test = cells_validation_test[(len(cells_validation_test) // 2) :]

                folds.append(
                    {
                        "cells_train": cells_train,
                        "cells_validation": cells_validation,
                        "cells_test": cells_test,
                        "repeat": repeat_ix,
                    }
                )
        self.folds = folds

        return self

    def sample_cellxregion(
        self,
        fragments: Fragments,
        n_folds: int,
        n_repeats: int = 1,
        stratify_by_chromosome=True,
        overwrite: bool = False,
    ):
        """
        Sample cells and regions into folds

        Parameters:
            fragments:
                the fragments
            n_folds:
                the number of folds
            n_repeats:
                the number of repeats
            overwrite:
                whether to overwrite existing folds
        """
        if not overwrite and self.get("folds").exists(self):
            return self

        folds = []

        for repeat_ix in range(n_repeats):
            generator = np.random.RandomState(repeat_ix)

            cells_all = generator.permutation(fragments.n_cells)

            cell_bins = np.floor((np.arange(len(cells_all)) / (len(cells_all) / n_folds)))

            regions_all = np.arange(fragments.n_regions)

            if stratify_by_chromosome:
                chr_column = "chr" if "chr" in fragments.regions.coordinates.columns else "chrom"
                chr_order = generator.permutation(fragments.regions.coordinates[chr_column].unique())
                region_chrs = pd.Categorical(
                    fragments.regions.coordinates[chr_column].astype(str), categories=chr_order
                ).codes
                region_bins = np.floor((region_chrs / (len(chr_order) / n_folds))).astype(int)
            else:
                region_bins = np.floor((np.arange(len(regions_all)) / (len(regions_all) / n_folds)))

            for i in range(n_folds):
                cells_train = cells_all[cell_bins != i]
                cells_validation_test = cells_all[cell_bins == i]
                cells_validation = cells_validation_test[: (len(cells_validation_test) // 2)]
                cells_test = cells_validation_test[(len(cells_validation_test) // 2) :]

                regions_train = regions_all[region_bins != i]
                regions_validation_test = generator.permutation(regions_all[region_bins == i])
                regions_validation = regions_validation_test[: (len(regions_validation_test) // 2)]
                regions_test = regions_validation_test[(len(regions_validation_test) // 2) :]

                folds.append(
                    {
                        "cells_train": cells_train,
                        "cells_validation": cells_validation,
                        "cells_test": cells_test,
                        "regions_train": regions_train,
                        "regions_validation": regions_validation,
                        "regions_test": regions_test,
                        "repeat": repeat_ix,
                    }
                )
        self.folds = folds
        return self

    def __getitem__(self, ix):
        return self.folds[ix]

    def __len__(self):
        return len(self.folds)

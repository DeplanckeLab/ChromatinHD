import numpy as np
import pandas as pd
import pickle

from chromatinhd.flow import (
    Flow,
    StoredTorchInt32,
    Stored,
    StoredTorchInt64,
    TSV,
    Linked,
)

from chromatinhd.data.fragments import Fragments

import torch
import math
import pathlib
import typing
import tqdm


class Folds(Flow):
    """
    Folds of multiple cell and gene combinations
    """

    folds = Stored("folds")
    """the folds"""

    def sample_cells(self, fragments: Fragments, n_folds: int, n_repeats: int = 1):
        folds = []

        for repeat_ix in range(n_repeats):
            generator = np.random.RandomState(repeat_ix)

            cells_all = generator.permutation(fragments.n_cells)

            cell_bins = np.floor(
                (np.arange(len(cells_all)) / (len(cells_all) / n_folds))
            )

            for i in range(n_folds):
                cells_train = cells_all[cell_bins != i]
                cells_validation_test = cells_all[cell_bins == i]
                cells_validation = cells_validation_test[
                    : (len(cells_validation_test) // 2)
                ]
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

    def sample_cellxgene(self, fragments: Fragments, n_folds: int, n_repeats: int = 1):
        folds = []

        for repeat_ix in range(n_repeats):
            generator = np.random.RandomState(repeat_ix)

            cells_all = generator.permutation(fragments.n_cells)

            cell_bins = np.floor(
                (np.arange(len(cells_all)) / (len(cells_all) / n_folds))
            )

            genes_all = np.arange(fragments.n_genes)

            chr_order = generator.permutation(
                fragments.regions.coordinates["chr"].unique()
            )
            gene_chrs = pd.Categorical(
                fragments.regions.coordinates["chr"].astype(str), categories=chr_order
            ).codes
            gene_bins = np.floor((gene_chrs / (len(chr_order) / n_folds))).astype(int)

            for i in range(n_folds):
                cells_train = cells_all[cell_bins != i]
                cells_validation_test = cells_all[cell_bins == i]
                cells_validation = cells_validation_test[
                    : (len(cells_validation_test) // 2)
                ]
                cells_test = cells_validation_test[(len(cells_validation_test) // 2) :]

                genes_train = genes_all[gene_bins != i]
                genes_validation_test = genes_all[gene_bins == i]
                genes_validation = genes_validation_test[
                    : (len(genes_validation_test) // 2)
                ]
                genes_test = genes_validation_test[(len(genes_validation_test) // 2) :]

                folds.append(
                    {
                        "cells_train": cells_train,
                        "cells_validation": cells_validation,
                        "cells_test": cells_test,
                        "genes_train": genes_train,
                        "genes_validation": genes_validation,
                        "genes_test": genes_test,
                        "repeat": repeat_ix,
                    }
                )
        self.folds = folds

    def __getitem__(self, ix):
        return self.folds[ix]

    def __len__(self):
        return len(self.folds)

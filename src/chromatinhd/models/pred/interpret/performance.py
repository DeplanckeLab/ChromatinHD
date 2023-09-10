import chromatinhd as chd
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import scipy.stats
import tqdm.auto as tqdm
from chromatinhd import get_default_device

from chromatinhd.flow.objects import Stored


class Performance(chd.flow.Flow):
    """
    The train/validation/test performance of a (set of) models.
    """

    scores = Stored()
    genescores = Stored()

    def score(
        self,
        fragments,
        transcriptome,
        models,
        folds,
        regions=None,
        force=False,
        device=None,
    ):
        if regions is None:
            regions = fragments.regions.var.index

        pbar = tqdm.tqdm(enumerate(zip(folds, models)), leave=False)

        scores = []
        genescores = []

        phases = ["train", "validation", "test"]

        genescores = xr.Dataset(
            {
                "cor": (["model", "phase", "gene"], np.zeros((len(models), len(phases), len(fragments.var.index)))),
                "zmse": (["model", "phase", "gene"], np.zeros((len(models), len(phases), len(fragments.var.index)))),
            },
            coords={"model": range(len(models)), "phase": phases, "gene": fragments.var.index},
        )
        scores = xr.Dataset(
            {
                "cor": (["model", "phase"], np.zeros((len(models), len(phases)))),
                "zmse": (["model", "phase"], np.zeros((len(models), len(phases)))),
            },
            coords={"model": range(len(models)), "phase": phases},
        )

        for fold_ix, (fold, model) in pbar:
            for phase in phases:
                cells_oi = fold[f"cells_{phase}"]
                prediction = model.get_prediction(fragments, transcriptome, cell_ixs=cells_oi)
                cor_genes = chd.utils.paircor(prediction["predicted"].values, transcriptome.X[cells_oi, :])
                cor_overall = cor_genes.mean()
                zmse_genes = chd.utils.pairzmse(prediction["predicted"].values, transcriptome.X[cells_oi, :])
                zmse_overall = zmse_genes.mean()

                scores["cor"][fold_ix, phases.index(phase)] = cor_overall
                scores["zmse"][fold_ix, phases.index(phase)] = zmse_overall
                genescores["cor"][fold_ix, phases.index(phase)] = cor_genes
                genescores["zmse"][fold_ix, phases.index(phase)] = zmse_genes

        self.scores = scores
        self.genescores = genescores

        return self

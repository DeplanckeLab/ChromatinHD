from __future__ import annotations

import chromatinhd as chd
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import scipy.stats
import tqdm.auto as tqdm
from chromatinhd import get_default_device

from chromatinhd.flow.objects import StoredDict, Dataset, Stored


def fdr(p_vals):
    from scipy.stats import rankdata

    ranked_p_values = rankdata(p_vals)
    qval = p_vals * len(p_vals) / ranked_p_values
    qval[qval > 1] = 1

    return qval


def fdr_nan(p_vals):
    qvals = np.zeros_like(p_vals)
    qvals[~np.isnan(p_vals)] = fdr(p_vals[~np.isnan(p_vals)])
    qvals[np.isnan(p_vals)] = np.nan
    return qvals


class Size(chd.flow.Flow):
    """
    Interpret a *pred* model positionally by censoring windows of across multiple window sizes.
    """

    design = chd.flow.Stored()
    """
    The design of the censoring windows.
    """

    regions = chd.flow.Stored(default=set)
    """
    The regions that have been scored.
    """

    scores = chd.flow.SparseDataset()
    interpolation = chd.flow.SparseDataset()
    censorer = Stored()

    @classmethod
    def create(cls, folds, transcriptome, fragments, censorer, path=None, phases=None, overwrite=False) -> Size:
        self = super().create(path, reset=overwrite)

        if phases is None:
            phases = ["test"]

        regions = fragments.regions.var.index

        coords_pointed = {
            regions.name: regions,
            "fold": pd.Index(range(len(folds)), name="fold"),
            "phase": pd.Index(phases, name="phase"),
        }
        coords_fixed = {
            censorer.design.index.name: censorer.design.index[1:],
        }

        self.censorer = censorer

        design_censorer = censorer.design

        if not self.o.scores.exists(self):
            self.scores = chd.sparse.SparseDataset.create(
                self.path / "scores",
                variables={
                    "deltacor": {
                        "dimensions": (regions.name, "fold", "phase", design_censorer.index.name),
                        "dtype": np.float32,
                    },
                    "reldeltacor": {
                        "dimensions": (regions.name, "fold", "phase", design_censorer.index.name),
                        "dtype": np.float32,
                    },
                    "lost": {
                        "dimensions": (regions.name, "fold", "phase", design_censorer.index.name),
                        "dtype": np.float32,
                    },
                    "censored": {
                        "dimensions": (regions.name, "fold", "phase", design_censorer.index.name),
                        "dtype": np.float32,
                    },
                    "effect": {
                        "dimensions": (regions.name, "fold", "phase", design_censorer.index.name),
                        "dtype": np.float32,
                    },
                    "scored": {
                        "dimensions": (regions.name, "fold"),
                        "dtype": bool,
                        "sparse": False,
                    },
                },
                coords_pointed=coords_pointed,
                coords_fixed=coords_fixed,
            )

        return self

    def score(
        self,
        models,
        folds=None,
        fragments=None,
        transcriptome=None,
        regions=None,
        force=False,
        device=None,
    ):
        force_ = force

        if regions is None:
            # get regions from models
            if models.regions_oi is not None:
                regions = models.regions_oi
            else:
                regions = self.scores.coords_pointed[list(self.scores.coords_pointed)[0]]

        if folds is None:
            folds = models.folds

        pbar = tqdm.tqdm(regions, leave=False)

        for region in pbar:
            pbar.set_description(region)

            for fold_ix, fold in enumerate(folds):
                force = force_
                if not self.scores["scored"][region, fold_ix]:
                    force = True

                if force:
                    model_name = f"{region}_{fold_ix}"
                    if model_name not in models:
                        continue

                    pbar.set_description(region + " " + str(fold_ix))

                    model = models[model_name]
                    predicted, expected, n_fragments = model.get_prediction_censored(
                        fragments=fragments,
                        transcriptome=transcriptome,
                        censorer=self.censorer,
                        cell_ixs=np.concatenate([fold["cells_validation"], fold["cells_test"]]),
                        regions=[region],
                        device=device,
                        min_fragments=0,
                    )

                    # select 1st region, given that we're working with one region anyway
                    predicted = predicted[..., 0]
                    expected = expected[..., 0]
                    n_fragments = n_fragments[..., 0]

                    cor = chd.utils.paircor(predicted, expected, dim=-1)
                    deltacor = cor[1:] - cor[0]

                    reldeltacor = (cor[1:] - cor[0]) / (cor[0] + 1e-6)

                    lost = (n_fragments[1:] - n_fragments[0]).mean(-1)

                    censored = lost / (n_fragments[0].mean(-1) + 1e-6)

                    effect = (predicted[1:] - predicted[0]).mean(-1)

                    self.scores["deltacor"][region, fold_ix, "test"] = deltacor
                    self.scores["reldeltacor"][region, fold_ix, "test"] = reldeltacor
                    self.scores["lost"][region, fold_ix, "test"] = lost
                    self.scores["effect"][region, fold_ix, "test"] = effect
                    self.scores["censored"][region, fold_ix, "test"] = censored
                    self.scores["scored"][region, fold_ix] = True

        return self

    @property
    def design(self):
        return self.censorer.design.iloc[1:]

    def get_scoring_path(self, region):
        path = self.path / f"{region}"
        path.mkdir(parents=True, exist_ok=True)
        return path

import chromatinhd as chd
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import scipy.stats
import tqdm.auto as tqdm
from chromatinhd import get_default_device

from chromatinhd.flow.objects import Stored, Dataset, StoredDict

from itertools import product


class Performance(chd.flow.Flow):
    """
    The train/validation/test performance of a (set of) models.
    """

    scores = chd.flow.SparseDataset()

    folds = None

    @classmethod
    def create(cls, folds, fragments, phases=None, overwrite=False, path=None):
        self = cls(path=path, reset=overwrite)

        self.folds = folds
        self.fragments = fragments

        if self.o.scores.exists(self) and not overwrite:
            assert self.scores.coords_pointed["region"].equals(fragments.var.index)

            return self

        if phases is None:
            phases = ["train", "validation", "test"]

        coords_pointed = {
            "region": fragments.regions.var.index,
            "fold": pd.Index(range(len(folds)), name="fold"),
            "phase": pd.Index(phases, name="phase"),
        }
        coords_fixed = {}
        self.scores = chd.sparse.SparseDataset.create(
            self.path / "scores",
            variables={
                "likelihood": {
                    "dimensions": ("region", "fold", "phase"),
                    "dtype": np.float32,
                    "sparse": False,
                },
                "scored": {
                    "dimensions": ("region", "fold"),
                    "dtype": np.bool,
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
        force=False,
        device=None,
        regions=None,
        pbar=True,
    ):
        fragments = self.fragments
        folds = self.folds

        phases = self.scores.coords["phase"].values

        region_name = fragments.var.index.name

        design = self.scores["scored"].sel_xr().to_dataframe(name="scored")
        design = design.loc[~design["scored"]]
        design = design.reset_index()[[region_name, "fold"]]

        if regions is not None:
            design = design.loc[design[region_name].isin(regions)]

        progress = design.groupby("fold")
        if pbar is True:
            progress = tqdm.tqdm(progress, total=len(progress), leave=False)

        for fold_ix, subdesign in progress:
            if models.fitted(fold_ix):
                for phase in phases:
                    fold = folds[fold_ix]
                    cells_oi = fold[f"cells_{phase}"]

                    likelihood = models.get_prediction(
                        fold_ix=fold_ix, cell_ixs=cells_oi, regions=subdesign[region_name], return_raw=True
                    )

                    for region_ix, region_oi in enumerate(subdesign[region_name].values):
                        self.scores["likelihood"][
                            region_oi,
                            fold_ix,
                            phase,
                        ] = likelihood[:, region_ix].mean()

                for region_oi in subdesign[region_name]:
                    self.scores["scored"][region_oi, fold_ix] = True

        return self

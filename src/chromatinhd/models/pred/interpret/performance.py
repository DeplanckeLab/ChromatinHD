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
    def create(cls, folds, transcriptome, fragments, phases=None, overwrite=False, path=None):
        self = cls(path=path, reset=overwrite)

        self.folds = folds
        self.transcriptome = transcriptome
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
                "cor": {
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

        if regions is None:
            regions_oi = fragments.var.index.tolist() if models.regions_oi is None else models.regions_oi
            if isinstance(regions_oi, pd.Series):
                regions_oi = regions_oi.tolist()
        else:
            regions_oi = regions

        design = (
            self.scores["scored"]
            .sel_xr()
            .sel({fragments.var.index.name: regions_oi, "fold": range(len(folds))})
            .to_dataframe(name="scored")
        )
        design["force"] = (~design["scored"]) | force

        design = design.groupby("gene").any()

        regions_oi = design.index[design["force"]]

        if len(regions_oi) == 0:
            return self

        for fold_ix in range(len(folds)):
            for phase in phases:
                fold = folds[fold_ix]
                cells_oi = fold[f"cells_{phase}"]

                for region_ix, region_oi in enumerate(regions_oi):
                    predicted, expected, n_fragments = models.get_prediction(
                        region=region_oi,
                        fold_ix=fold_ix,
                        cell_ixs=cells_oi,
                        return_raw=True,
                        fragments=self.fragments,
                        transcriptome=self.transcriptome,
                    )

                    cor = chd.utils.paircor(predicted, expected)

                    self.scores["cor"][
                        region_oi,
                        fold_ix,
                        phase,
                    ] = cor[0]
                    self.scores["scored"][region_oi, fold_ix] = True

        return self

    @property
    def scored(self):
        return self.o.scores.exists(self)

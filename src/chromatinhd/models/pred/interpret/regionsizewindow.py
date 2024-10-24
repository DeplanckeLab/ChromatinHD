import itertools
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd
import tqdm.auto as tqdm
import xarray as xr

import chromatinhd as chd
from chromatinhd import get_default_device
from chromatinhd.data.folds import Folds
from chromatinhd.data.fragments import Fragments
from chromatinhd.data.transcriptome import Transcriptome
from chromatinhd.models.pred.model.multiscale import Models

from chromatinhd.flow.objects import StoredDict, Dataset, DataArray


class RegionSizeWindow(chd.flow.Flow):
    """
    Interpret a *pred* model positionally by censoring windows and comparing
    the decrease in predictivity per cell between pairs of windows
    """

    design = chd.flow.Stored()

    scores = StoredDict(Dataset)
    interaction = StoredDict(DataArray)

    def score(
        self,
        models: Models,
        censorer,
        regions: Optional[List] = None,
        folds=None,
        force=False,
        device=None,
    ):
        """
        Score the models

        Parameters:
            fragments:
                the fragments
            transcriptome:
                the transcriptome
            models:
                the models
            folds:
                the folds
            regions:
                which regions to score, defaults to all

        """
        force_ = force
        design = censorer.design.iloc[1:].copy()
        self.design = design

        if device is None:
            device = get_default_device()

        if folds is None:
            folds = models.folds

        pbar = tqdm.tqdm(regions, leave=False)
        for region in pbar:
            pbar.set_description(region)

            force = force_

            if region not in self.scores:
                force = True

            deltacor_folds = []
            effect_folds = []
            lost_folds = []

            if force:
                for fold_ix, fold in enumerate(folds):
                    model_name = f"{region}_{fold_ix}"
                    if model_name not in models:
                        raise ValueError(f"Model {model_name} not found")

                    pbar.set_description(region + " " + str(fold_ix))

                    model = models[model_name]
                    predicted, expected, n_fragments = model.get_prediction_censored(
                        # fragments=fragments,
                        # transcriptome=transcriptome,
                        censorer=censorer,
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
                    deltacor_folds.append(cor[1:] - cor[0])

                    lost_folds.append((n_fragments[0] - n_fragments[1:]).mean(-1))

                    effect_folds.append((predicted[0] - predicted[1:]).mean(-1))

                result = xr.Dataset(
                    {
                        "deltacor": xr.DataArray(
                            deltacor_folds,
                            coords=[
                                ("fold", np.arange(len(folds))),
                                ("window", design.index),
                            ],
                        ),
                        "lost": xr.DataArray(
                            lost_folds,
                            coords=[
                                ("fold", np.arange(len(folds))),
                                ("window", design.index),
                            ],
                        ),
                        "effect": xr.DataArray(
                            effect_folds,
                            coords=[
                                ("fold", np.arange(len(folds))),
                                ("window", design.index),
                            ],
                        ),
                    }
                )

                self.scores[region] = result

        return self

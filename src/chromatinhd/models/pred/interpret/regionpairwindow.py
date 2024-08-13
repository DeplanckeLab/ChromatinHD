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

import chromatinhd.utils


def zscore(x, dim=0):
    return (x - x.mean(axis=dim, keepdims=True)) / x.std(axis=dim, keepdims=True)


def zscore_relative(x, y, dim=0):
    return (x - y.mean(axis=dim, keepdims=True)) / y.std(axis=dim, keepdims=True)


def fdr(p_vals):
    from scipy.stats import rankdata

    ranked_p_values = rankdata(p_vals)
    qval = p_vals * len(p_vals) / ranked_p_values
    qval[qval > 1] = 1

    return qval


class RegionPairWindow(chd.flow.Flow):
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
        transcriptome=None,
        fragments=None,
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

        if regions is None:
            regions = fragments.var.index

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
            copredictivity_folds = []
            lost_folds = []

            if force:
                for fold_ix, fold in enumerate(folds):
                    model_name = f"{region}_{fold_ix}"
                    if model_name not in models:
                        continue
                        raise ValueError(f"Model {model_name} not found")

                    pbar.set_description(region + " " + str(fold_ix))

                    model = models[model_name]
                    predicted, expected, n_fragments = model.get_prediction_censored(
                        fragments=fragments,
                        transcriptome=transcriptome,
                        censorer=censorer,
                        cell_ixs=np.concatenate([fold["cells_validation"], fold["cells_test"]]),
                        regions=[region],
                        device=device,
                    )

                    # select 1st region, given that we're working with one region anyway
                    predicted = predicted[..., 0]
                    expected = expected[..., 0]
                    n_fragments = n_fragments[..., 0]

                    # calculate delta cor per cell
                    predicted_censored = predicted[1:]
                    predicted_full = predicted[0][None, ...]
                    predicted_full_norm = zscore(predicted_full, 1)
                    predicted_censored_norm = zscore_relative(predicted_censored, predicted_full, 1)

                    expected_norm = zscore(expected[None, ...], 1)

                    celldeltacor = -np.abs(predicted_censored_norm - expected_norm) - -np.abs(
                        predicted_full_norm - expected_norm
                    )
                    with np.errstate(divide="ignore", invalid="ignore"):
                        copredictivity = np.corrcoef(celldeltacor)
                    copredictivity[np.isnan(copredictivity)] = 0.0

                    copredictivity_folds.append(copredictivity)

                    cor = chd.utils.paircor(predicted, expected, dim=-1)
                    deltacor = cor[1:] - cor[0]

                    lost = (n_fragments[0] - n_fragments[1:]).mean(-1)

                    deltacor_folds.append(deltacor)
                    lost_folds.append(lost)

                if len(lost_folds) == 0:
                    continue

                lost_folds = np.stack(lost_folds, 0)
                deltacor_folds = np.stack(deltacor_folds, 0)
                copredictivity_folds = np.stack(copredictivity_folds, 0)

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
                    }
                )

                windows_oi = lost_folds.mean(0) > 1e-3
                windows_oi = np.ones(len(design), dtype=bool)

                interaction = xr.DataArray(
                    copredictivity_folds[:, windows_oi][:, :, windows_oi],
                    coords=[
                        ("fold", np.arange(len(folds))),
                        ("window1", design.index[windows_oi]),
                        ("window2", design.index[windows_oi]),
                    ],
                )

                self.scores[region] = result
                self.interaction[region] = interaction

        return self

    def get_plotdata(self, region, windows=None):
        """
        Get plotdata for a region
        """

        if windows is None:
            windows = self.design
        else:
            x = self.design[["window_start", "window_end"]].values
            y = windows[["start", "end"]].values

            windows = self.design.loc[chromatinhd.utils.intervals.interval_contains_inclusive(x, y)]

        plotdata_windows = self.scores[region].mean("fold").to_dataframe()
        plotdata_interaction = self.interaction[region].mean("fold").to_pandas().unstack().to_frame("cor")

        plotdata_interaction = (
            plotdata_interaction.copy()
            .join(plotdata_windows.rename(columns=lambda x: x + "1"), on="window1")
            .join(plotdata_windows.rename(columns=lambda x: x + "2"), on="window2")
        )

        # make plotdata, making sure we have all window combinations, otherwise nan
        plotdata = (
            pd.DataFrame(itertools.combinations(windows.index, 2), columns=["window1", "window2"])
            .set_index(["window1", "window2"])
            .join(plotdata_interaction)
        )
        plotdata.loc[np.isnan(plotdata["cor"]), "cor"] = 0.0
        plotdata["dist"] = (
            windows.loc[plotdata.index.get_level_values("window2"), "window_mid"].values
            - windows.loc[plotdata.index.get_level_values("window1"), "window_mid"].values
        )

        plotdata.loc[plotdata["dist"] < 1000, "cor"] = 0.0

        plotdata = plotdata.query("dist > 0")

        return plotdata

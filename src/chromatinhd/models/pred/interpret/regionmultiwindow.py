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


class RegionMultiWindow(chd.flow.Flow):
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
    def create(
        cls, folds, transcriptome, fragments, censorer, path=None, phases=None, overwrite=False
    ) -> RegionMultiWindow:
        self = super().create(path, reset=overwrite)

        if phases is None:
            # phases = ["train", "validation", "test"]
            phases = ["validation", "test"]

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
                    "lost": {
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

        if not self.o.interpolation.exists(self):
            positions_oi = np.arange(
                self.design["window_start"].min(),
                self.design["window_end"].max() + 1,
                10,
            )

            self.design_interpolation = pd.DataFrame(
                {
                    "position": positions_oi,
                }
            ).set_index("position")

            self.interpolation = chd.sparse.SparseDataset.create(
                self.path / "interpolation",
                variables={
                    "deltacor": {
                        "dimensions": (regions.name, self.design_interpolation.index.name),
                        "dtype": np.float32,
                    },
                    "lost": {"dimensions": (regions.name, self.design_interpolation.index.name), "dtype": np.float32},
                    "effect": {"dimensions": (regions.name, self.design_interpolation.index.name), "dtype": np.float32},
                    "interpolated": {"dimensions": (regions.name,), "dtype": bool, "sparse": False},
                },
                coords_pointed={regions.name: regions},
                coords_fixed={"position": self.design_interpolation.index},
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
        min_fragments=3,
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
                        min_fragments=min_fragments,
                    )

                    # select 1st region, given that we're working with one region anyway
                    predicted = predicted[..., 0]
                    expected = expected[..., 0]
                    n_fragments = n_fragments[..., 0]

                    cor = chd.utils.paircor(predicted, expected, dim=-1)
                    deltacor = cor[1:] - cor[0]

                    lost = (n_fragments[0] - n_fragments[1:]).mean(-1)

                    effect = (predicted[0] - predicted[1:]).mean(-1)

                    self.scores["deltacor"][region, fold_ix, "test"] = deltacor
                    self.scores["lost"][region, fold_ix, "test"] = lost
                    self.scores["effect"][region, fold_ix, "test"] = effect
                    self.scores["scored"][region, fold_ix] = True

        return self

    def score2(
        self,
        models,
        folds=None,
        fragments=None,
        transcriptome=None,
        regions=None,
        force=False,
        device=None,
        min_fragments=3,
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
                    deltacor, lost, effect = model.get_performance_censored(
                        fragments=fragments,
                        transcriptome=transcriptome,
                        censorer=self.censorer,
                        cell_ixs=np.concatenate([fold["cells_validation"], fold["cells_test"]]),
                        regions=[region],
                        device=device,
                        min_fragments=min_fragments,
                    )

                    self.scores["deltacor"][region, fold_ix, "test"] = deltacor
                    self.scores["lost"][region, fold_ix, "test"] = lost
                    self.scores["effect"][region, fold_ix, "test"] = effect
                    self.scores["scored"][region, fold_ix] = True

        return self

    @property
    def design(self):
        return self.censorer.design.iloc[1:]

    def interpolate(self, regions=None, force=False, pbar=True):
        force_ = force

        if regions is None:
            regions = self.scores.coords_pointed[list(self.scores.coords_pointed)[0]]

        progress = regions
        if pbar:
            progress = tqdm.tqdm(progress, leave=False)

        for region in progress:
            if pbar:
                progress.set_description(region)

            if not all([self.scores["scored"][region, fold_ix] for fold_ix in self.scores.coords_pointed["fold"]]):
                continue

            force = force_
            if not self.interpolation["interpolated"][region]:
                force = True

            if force:
                deltacor, lost, effect = self._interpolate(region)
                self.interpolation["deltacor"][region] = deltacor
                self.interpolation["effect"][region] = effect
                self.interpolation["lost"][region] = lost
                self.interpolation["interpolated"][region] = True

        return self

    def _interpolate(self, region):
        deltacors = []
        effects = []
        losts = []
        for fold_ix in self.scores.coords_pointed["fold"]:
            deltacors.append(self.scores["deltacor"][region, fold_ix, "test"])
            effects.append(self.scores["effect"][region, fold_ix, "test"])
            losts.append(self.scores["lost"][region, fold_ix, "test"])
        deltacors = np.stack(deltacors)
        effects = np.stack(effects)
        losts = np.stack(losts)

        scores_statistical = []
        for i in range(deltacors.shape[1]):
            if deltacors.shape[0] > 1:
                scores_statistical.append(scipy.stats.ttest_1samp(deltacors[:, i], 0, alternative="less").pvalue)
            else:
                scores_statistical.append(0.0)
        scores_statistical = pd.DataFrame({"pvalue": scores_statistical})
        scores_statistical["qval"] = fdr_nan(scores_statistical["pvalue"])

        plotdata = pd.DataFrame(
            {
                "deltacor": deltacors.mean(0),
                "effect": effects.mean(0),
                "lost": losts.mean(0),
            },
            index=self.design.index,
        )
        plotdata = self.design.join(plotdata)

        plotdata["qval"] = scores_statistical["qval"].values

        window_sizes_info = pd.DataFrame({"window_size": self.design["window_size"].unique()}).set_index("window_size")
        window_sizes_info["ix"] = np.arange(len(window_sizes_info))

        # interpolate
        positions_oi = np.arange(
            self.design["window_start"].min(),
            self.design["window_end"].max() + 1,
            10,
        )

        deltacor_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
        lost_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
        effect_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
        for window_size, window_size_info in window_sizes_info.iterrows():
            plotdata_oi = plotdata.query("window_size == @window_size")
            x = plotdata_oi["window_mid"].values.copy()
            y = plotdata_oi["deltacor"].values.copy()
            # y[(plotdata_oi["qval"] > 0.2) | pd.isnull(plotdata_oi["qval"])] = 0.0
            deltacor_interpolated_ = np.clip(
                np.interp(positions_oi, x, y) / window_size * 1000,
                -np.inf,
                0,
            )
            deltacor_interpolated[window_size_info["ix"], :] = deltacor_interpolated_

            lost_interpolated_ = (
                np.interp(positions_oi, plotdata_oi["window_mid"], plotdata_oi["lost"]) / window_size * 1000
            )
            lost_interpolated[window_size_info["ix"], :] = lost_interpolated_

            effect_interpolated_ = (
                np.interp(
                    positions_oi,
                    plotdata_oi["window_mid"],
                    plotdata_oi["effect"],
                )
                / window_size
                * 1000
            )
            effect_interpolated[window_size_info["ix"], :] = effect_interpolated_
        return deltacor_interpolated.mean(0), lost_interpolated.mean(0), effect_interpolated.mean(0)

    def get_plotdata(self, region):
        if not self.interpolation["interpolated"][region]:
            raise ValueError(f"Region {region} not interpolated. Run .interpolate() first.")

        plotdata = self.interpolation.sel_xr(region, variables=["deltacor", "lost", "effect"]).to_pandas()

        return plotdata

    def get_scoring_path(self, region):
        path = self.path / f"{region}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def select_windows(self, region_id, max_merge_distance=500, min_length=50, padding=500, lost_cutoff=0.5):
        from scipy.ndimage import convolve

        def spread_true(arr, width=5):
            kernel = np.ones(width, dtype=bool)
            result = convolve(arr, kernel, mode="constant", cval=False)
            result = result != 0
            return result

        plotdata = self.get_plotdata(region_id)
        selection = pd.DataFrame({"chosen": (plotdata["lost"] > lost_cutoff)})

        # add padding
        step = plotdata.index.get_level_values("position")[1] - plotdata.index.get_level_values("position")[0]
        k_padding = int(padding // step)
        selection["chosen"] = spread_true(selection["chosen"], width=k_padding)

        # select all contiguous regions where chosen is true
        selection["selection"] = selection["chosen"].cumsum()

        regions = pd.DataFrame(
            {
                "start": selection.index[
                    (np.diff(np.pad(selection["chosen"], (1, 1), constant_values=False).astype(int)) == 1)[:-1]
                ],
                "end": selection.index[
                    (np.diff(np.pad(selection["chosen"], (1, 1), constant_values=False).astype(int)) == -1)[1:]
                ],
            }
        )

        # merge regions that are close to each other
        regions["distance_to_next"] = regions["start"].shift(-1) - regions["end"]

        regions["merge"] = (regions["distance_to_next"] < max_merge_distance).fillna(False)
        regions["group"] = (~regions["merge"]).cumsum().shift(1).fillna(0).astype(int)
        regions = (
            regions.groupby("group")
            .agg({"start": "min", "end": "max", "distance_to_next": "last"})
            .reset_index(drop=True)
        )

        # filter on length
        regions["length"] = regions["end"] - regions["start"]
        regions = regions[regions["length"] > min_length]

        return regions

    def extract_predictive_windows(self, region_id=None, deltacor_cutoff=-0.001):
        """
        Extract predictive windows for one (or more) regions
        """

        feature_name = list(self.scores.coords_pointed.keys())[0]

        if region_id is None:
            region_id = self.scores.coords_pointed[feature_name]

        if isinstance(region_id, str):
            region_id = [region_id]

        extracted = []

        for region_id in region_id:
            if self.interpolation["interpolated"][region_id]:
                plotdata = self.get_plotdata(region_id)
                plotdata["chosen"] = (plotdata["deltacor"] < deltacor_cutoff) & (plotdata["effect"] > 0)

                extracted_region_positive = pd.DataFrame(
                    {
                        "start": plotdata.index[
                            (np.diff(np.pad(plotdata["chosen"], (1, 1), constant_values=False).astype(int)) == 1)[:-1]
                        ].astype(int),
                        "end": plotdata.index[
                            (np.diff(np.pad(plotdata["chosen"], (1, 1), constant_values=False).astype(int)) == -1)[1:]
                        ].astype(int),
                        feature_name: region_id,
                        "effect_direction": +1,
                    }
                )
                extracted.append(extracted_region_positive)

                plotdata["chosen"] = (plotdata["deltacor"] < deltacor_cutoff) & (plotdata["effect"] < 0)
                extracted_region_negative = pd.DataFrame(
                    {
                        "start": plotdata.index[
                            (np.diff(np.pad(plotdata["chosen"], (1, 1), constant_values=False).astype(int)) == 1)[:-1]
                        ],
                        "end": plotdata.index[
                            (np.diff(np.pad(plotdata["chosen"], (1, 1), constant_values=False).astype(int)) == -1)[1:]
                        ],
                        feature_name: region_id,
                        "effect_direction": -1,
                    }
                )
                extracted.append(extracted_region_negative)

        return pd.concat(extracted)

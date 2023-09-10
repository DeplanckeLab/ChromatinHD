import chromatinhd as chd
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import scipy.stats
import tqdm.auto as tqdm
from chromatinhd import get_default_device

from chromatinhd.flow.objects import StoredDict, Dataset


def fdr(p_vals):
    from scipy.stats import rankdata

    ranked_p_values = rankdata(p_vals)
    qval = p_vals * len(p_vals) / ranked_p_values
    qval[qval > 1] = 1

    return qval


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

    scores = StoredDict(Dataset)
    interpolated = StoredDict(Dataset)

    def score(
        self,
        fragments,
        transcriptome,
        models,
        folds,
        censorer,
        regions=None,
        force=False,
        device=None,
    ):
        force_ = force
        design = censorer.design.iloc[1:].copy()
        self.design = design

        if regions is None:
            regions = fragments.regions.var.index

        pbar = tqdm.tqdm(regions, leave=False)

        for region in pbar:
            pbar.set_description(region)

            force = force_
            if region not in self.scores:
                force = True

            if force:
                deltacor_folds = []
                lost_folds = []
                effect_folds = []
                for fold, model in zip(folds, models):
                    predicted, expected, n_fragments = model.get_prediction_censored(
                        fragments,
                        transcriptome,
                        censorer,
                        cell_ixs=np.concatenate([fold["cells_validation"], fold["cells_test"]]),
                        regions=[region],
                        device=device,
                    )

                    # select 1st region, given that we're working with one region anyway
                    predicted = predicted[..., 0]
                    expected = expected[..., 0]
                    n_fragments = n_fragments[..., 0]

                    cor = chd.utils.paircor(predicted, expected, dim=-1)
                    deltacor = cor[1:] - cor[0]

                    lost = (n_fragments[0] - n_fragments[1:]).mean(-1)

                    effect = (predicted[0] - predicted[1:]).mean(-1)

                    deltacor_folds.append(deltacor)
                    lost_folds.append(lost)
                    effect_folds.append(effect)

                deltacor_folds = np.stack(deltacor_folds, 0)
                lost_folds = np.stack(lost_folds, 0)
                effect_folds = np.stack(effect_folds, 0)

                result = xr.Dataset(
                    {
                        "deltacor": xr.DataArray(
                            deltacor_folds,
                            coords=[
                                ("model", np.arange(len(models))),
                                ("window", design.index),
                            ],
                        ),
                        "lost": xr.DataArray(
                            lost_folds,
                            coords=[
                                ("model", np.arange(len(models))),
                                ("window", design.index),
                            ],
                        ),
                        "effect": xr.DataArray(
                            effect_folds,
                            coords=[
                                ("model", np.arange(len(models))),
                                ("window", design.index),
                            ],
                        ),
                    }
                )

                self.scores[region] = result

        return self

    def interpolate(self, regions=None, force=False):
        force_ = force

        if regions is None:
            regions = self.scores.keys()

        pbar = tqdm.tqdm(regions, leave=False)

        for region in pbar:
            pbar.set_description(region)

            if region not in self.scores:
                continue

            force = force_
            if region not in self.interpolated:
                force = True

            if force:
                scores = self.scores[region]

                x = scores["deltacor"].values
                scores_statistical = []
                for i in range(x.shape[1]):
                    if x.shape[0] > 1:
                        scores_statistical.append(scipy.stats.ttest_1samp(x[:, i], 0, alternative="less").pvalue)
                    else:
                        scores_statistical.append(1.0)
                scores_statistical = pd.DataFrame({"pvalue": scores_statistical})
                scores_statistical["qval"] = fdr(scores_statistical["pvalue"])

                plotdata = scores.mean("model").stack().to_dataframe()
                plotdata = self.design.join(plotdata)

                plotdata["qval"] = scores_statistical["qval"].values

                window_sizes_info = pd.DataFrame({"window_size": self.design["window_size"].unique()}).set_index(
                    "window_size"
                )
                window_sizes_info["ix"] = np.arange(len(window_sizes_info))

                # interpolate
                positions_oi = np.arange(
                    self.design["window_start"].min(),
                    self.design["window_end"].max() + 1,
                )

                deltacor_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
                lost_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
                effect_interpolated = np.zeros((len(window_sizes_info), len(positions_oi)))
                for window_size, window_size_info in window_sizes_info.iterrows():
                    plotdata_oi = plotdata.query("window_size == @window_size")
                    x = plotdata_oi["window_mid"].values.copy()
                    y = plotdata_oi["deltacor"].values.copy()
                    y[plotdata_oi["qval"] > 0.1] = 0.0
                    deltacor_interpolated_ = np.clip(
                        np.interp(positions_oi, x, y) / window_size * 1000,
                        -np.inf,
                        0,
                        # np.inf,
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

                deltacor = xr.DataArray(
                    deltacor_interpolated.mean(0),
                    coords=[
                        ("position", positions_oi),
                    ],
                )
                lost = xr.DataArray(
                    lost_interpolated.mean(0),
                    coords=[
                        ("position", positions_oi),
                    ],
                )

                effect = xr.DataArray(
                    effect_interpolated.mean(0),
                    coords=[
                        ("position", positions_oi),
                    ],
                )

                # save
                interpolated = xr.Dataset({"deltacor": deltacor, "lost": lost, "effect": effect})

                self.interpolated[region] = interpolated

        return self

    def get_plotdata(self, region):
        if region not in self.interpolated:
            raise ValueError(f"Region {region} not in interpolated. Run .interpolate() first.")
        plotdata = self.interpolated[region].to_dataframe()

        return plotdata

    def get_scoring_path(self, region):
        path = self.path / f"{region}"
        path.mkdir(parents=True, exist_ok=True)
        return path

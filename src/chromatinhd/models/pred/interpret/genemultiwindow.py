import chromatinhd as chd
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import scipy.stats
import tqdm.auto as tqdm


def fdr(p_vals):
    from scipy.stats import rankdata

    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr


class GeneMultiWindow(chd.flow.Flow):
    design = chd.flow.Stored("design")

    genes = chd.flow.Stored("genes", default=set)

    def score(
        self,
        fragments,
        transcriptome,
        models,
        folds,
        genes,
        censorer,
        force=False,
        device="cuda",
    ):
        force_ = force
        design = censorer.design.iloc[1:].copy()
        self.design = design

        pbar = tqdm.tqdm(genes, leave=False)

        for gene in pbar:
            pbar.set_description(gene)
            scores_file = self.get_scoring_path(gene) / "scores.pkl"

            force = force_
            if not scores_file.exists():
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
                        cell_ixs=np.concatenate(
                            [fold["cells_validation"], fold["cells_test"]]
                        ),
                        genes=[gene],
                        device=device,
                    )

                    # select 1st gene, given that we're working with one gene anyway
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

                pickle.dump(result, scores_file.open("wb"))

                self.genes = self.genes | {gene}

    def interpolate(self, genes=None, force=False):
        force_ = force

        if genes is None:
            genes = self.genes

        pbar = tqdm.tqdm(genes, leave=False)

        for gene in pbar:
            pbar.set_description(gene)
            scores_file = self.get_scoring_path(gene) / "scores.pkl"

            if not scores_file.exists():
                continue

            interpolate_file = self.get_scoring_path(gene) / "interpolated.pkl"

            force = force_
            if not interpolate_file.exists():
                force = True

            if force:
                scores = pickle.load(scores_file.open("rb"))
                x = scores["deltacor"].values
                scores_statistical = []
                for i in range(x.shape[1]):
                    scores_statistical.append(
                        scipy.stats.ttest_1samp(x[:, i], 0, alternative="less").pvalue
                    )
                scores_statistical = pd.DataFrame({"pvalue": scores_statistical})
                scores_statistical["qval"] = fdr(scores_statistical["pvalue"])

                plotdata = scores.mean("model").stack().to_dataframe()
                plotdata = self.design.join(plotdata)

                plotdata["qval"] = scores_statistical["qval"].values

                window_sizes_info = pd.DataFrame(
                    {"window_size": self.design["window_size"].unique()}
                ).set_index("window_size")
                window_sizes_info["ix"] = np.arange(len(window_sizes_info))

                # interpolate
                positions_oi = np.arange(
                    self.design["window_start"].min(),
                    self.design["window_end"].max() + 1,
                )

                deltacor_interpolated = np.zeros(
                    (len(window_sizes_info), len(positions_oi))
                )
                lost_interpolated = np.zeros(
                    (len(window_sizes_info), len(positions_oi))
                )
                effect_interpolated = np.zeros(
                    (len(window_sizes_info), len(positions_oi))
                )
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
                    deltacor_interpolated[
                        window_size_info["ix"], :
                    ] = deltacor_interpolated_

                    lost_interpolated_ = (
                        np.interp(
                            positions_oi, plotdata_oi["window_mid"], plotdata_oi["lost"]
                        )
                        / window_size
                        * 1000
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
                    effect_interpolated[
                        window_size_info["ix"], :
                    ] = effect_interpolated_

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
                interpolated = xr.Dataset(
                    {"deltacor": deltacor, "lost": lost, "effect": effect}
                )
                pickle.dump(
                    interpolated,
                    interpolate_file.open("wb"),
                )

    def get_plotdata(self, gene):
        interpolated_file = self.get_scoring_path(gene) / "interpolated.pkl"
        if not interpolated_file.exists():
            raise FileNotFoundError(f"File {interpolated_file} does not exist")

        interpolated = pickle.load(interpolated_file.open("rb"))

        plotdata = interpolated.to_dataframe()

        return plotdata

    def get_scoring_path(self, gene):
        path = self.path / f"{gene}"
        path.mkdir(parents=True, exist_ok=True)
        return path

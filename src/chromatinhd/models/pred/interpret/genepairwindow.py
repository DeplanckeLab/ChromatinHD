import chromatinhd as chd
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import scipy.stats
import tqdm.auto as tqdm
import numpy as np
import itertools


def zscore(x, dim=0):
    return (x - x.mean(axis=dim, keepdims=True)) / x.std(axis=dim, keepdims=True)


def zscore_relative(x, y, dim=0):
    return (x - y.mean(axis=dim, keepdims=True)) / y.std(axis=dim, keepdims=True)


def fdr(p_vals):
    from scipy.stats import rankdata

    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr


class GenePairWindow(chd.flow.Flow):
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
            interaction_file = self.get_scoring_path(gene) / "interaction.pkl"

            force = force_
            if not all([file.exists() for file in [scores_file, interaction_file]]):
                force = True

            if force:
                deltacor_folds = []
                copredictivity_folds = []
                lost_folds = []
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

                    # calculate delta cor per cell
                    # calculate effect per cellxgene combination
                    predicted_censored = predicted[1:]
                    predicted_full = predicted[0][None, ...]
                    predicted_full_norm = zscore(predicted_full, 1)
                    predicted_censored_norm = zscore_relative(
                        predicted_censored, predicted_full, 1
                    )

                    expected_norm = zscore(expected[None, ...], 1)

                    celldeltacor = -np.abs(
                        predicted_censored_norm - expected_norm
                    ) - -np.abs(predicted_full_norm - expected_norm)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        copredictivity = np.corrcoef(celldeltacor)
                    copredictivity[np.isnan(copredictivity)] = 0.0

                    copredictivity_folds.append(copredictivity)

                    cor = chd.utils.paircor(predicted, expected, dim=-1)
                    deltacor = cor[1:] - cor[0]

                    lost = (n_fragments[0] - n_fragments[1:]).mean(-1)

                    deltacor_folds.append(deltacor)
                    lost_folds.append(lost)

                lost_folds = np.stack(lost_folds, 0)
                deltacor_folds = np.stack(deltacor_folds, 0)
                copredictivity_folds = np.stack(copredictivity_folds, 0)
                print(copredictivity_folds.shape)

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
                    }
                )

                windows_oi = lost_folds.mean(0) > 1e-3

                interaction = xr.DataArray(
                    copredictivity_folds[:, windows_oi][:, :, windows_oi],
                    coords=[
                        ("model", np.arange(len(models))),
                        ("window1", design.index[windows_oi]),
                        ("window2", design.index[windows_oi]),
                    ],
                )

                pickle.dump(result, scores_file.open("wb"))
                pickle.dump(interaction, interaction_file.open("wb"))

                self.genes = self.genes | {gene}

    def get_plotdata(self, gene):
        """
        Get plotdata for a gene
        """
        interaction = pickle.load(
            open(
                self.get_scoring_path(gene) / "interaction.pkl",
                "rb",
            )
        )

        plotdata = interaction.mean("model").to_dataframe("cor").reset_index()
        plotdata["window1"] = plotdata["window1"].astype("category")
        plotdata["window2"] = plotdata["window2"].astype("category")

        plotdata = (
            pd.DataFrame(
                itertools.combinations(self.design.index, 2),
                columns=["window1", "window2"],
            )
            .set_index(["window1", "window2"])
            .join(plotdata.set_index(["window1", "window2"]))
        )
        plotdata = plotdata.reset_index().fillna({"cor": 0.0})
        plotdata["window_mid1"] = self.design.loc[plotdata["window1"]][
            "window_mid"
        ].values
        plotdata["window_mid2"] = self.design.loc[plotdata["window2"]][
            "window_mid"
        ].values
        plotdata["dist"] = np.abs(plotdata["window_mid1"] - plotdata["window_mid2"])
        plotdata = plotdata.query("(window_mid1 < window_mid2)")
        # plotdata = plotdata.query("dist > 1000")

        # x = interaction.stack({"window1_window2": ["window1", "window2"]}).values
        # print(x.shape)
        # scores_statistical = []
        # for i in range(x.shape[1]):
        #     scores_statistical.append(scipy.stats.ttest_1samp(x[:, i], 0).pvalue)
        # scores_statistical = pd.DataFrame({"pval": scores_statistical})
        # scores_statistical["pval"] = scores_statistical["pval"].fillna(1.0)
        # scores_statistical["qval"] = fdr(scores_statistical["pval"])

        # plotdata["pval"] = scores_statistical["pval"].values
        # plotdata["qval"] = scores_statistical["qval"].values

        plotdata.loc[plotdata["dist"] < 1000, "cor"] = 0.0

        return plotdata

    def get_scoring_path(self, gene):
        path = self.path / f"{gene}"
        path.mkdir(parents=True, exist_ok=True)
        return path

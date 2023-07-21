import chromatinhd as chd
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import scipy.stats
import tqdm.auto as tqdm
import numpy as np


def zscore(x, dim=0):
    return (x - x.mean(axis=dim, keepdims=True)) / x.std(axis=dim, keepdims=True)


def zscore_relative(x, y, dim=0):
    return (x - y.mean(axis=dim, keepdims=True)) / y.std(axis=dim, keepdims=True)


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

                    deltacor_folds.append(deltacor)

                deltacor_folds = np.stack(deltacor_folds, 0)
                copredictivity_folds = np.stack(copredictivity_folds, 0)

                result = xr.Dataset(
                    {
                        "deltacor": xr.DataArray(
                            deltacor_folds,
                            coords=[
                                ("model", np.arange(len(models))),
                                ("window", design.index),
                            ],
                        ),
                    }
                )

                interaction = xr.DataArray(
                    copredictivity_folds,
                    coords=[
                        ("model", np.arange(len(models))),
                        ("window1", design.index),
                        ("window2", design.index),
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

        plotdata_copredictivity = (
            interaction.mean("model").to_dataframe("cor").reset_index()
        )

        plotdata_copredictivity["window_mid1"] = self.design.loc[
            plotdata_copredictivity["window1"]
        ]["window_mid"].values
        plotdata_copredictivity["window_mid2"] = self.design.loc[
            plotdata_copredictivity["window2"]
        ]["window_mid"].values
        plotdata_copredictivity["dist"] = np.abs(
            plotdata_copredictivity["window_mid1"]
            - plotdata_copredictivity["window_mid2"]
        )
        plotdata_copredictivity = plotdata_copredictivity.query(
            "(window_mid1 < window_mid2)"
        )
        plotdata_copredictivity.loc[
            plotdata_copredictivity["dist"] <= 1000, "cor"
        ] = 0.0

        return plotdata_copredictivity

    def get_scoring_path(self, gene):
        path = self.path / f"{gene}"
        path.mkdir(parents=True, exist_ok=True)
        return path

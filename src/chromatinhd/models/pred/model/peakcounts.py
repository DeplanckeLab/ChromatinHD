import numpy as np
import pandas as pd
import pickle
import time

import xgboost as xgb

import tqdm.auto as tqdm

import typing
import scipy.sparse

from chromatinhd.flow import Flow, Stored
import chromatinhd as chd

import sklearn.linear_model
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
import sklearn
import xarray as xr


def calculate_r2(y, y_predicted, y_train):
    return 1 - ((y_predicted - y) ** 2).sum() / ((y - y_train.mean()) ** 2).sum()


def rf_cv(x_train, y_train, x_validation, y_validation):
    param_grid = {"max_depth": [5, 10, 20], "max_features": [5, 10, 20], "n_estimators": [20, 50]}

    cv = [((np.arange(len(x_train)), np.arange(len(x_train), len(x_train) + len(x_validation))))]
    x_trainvalidation = np.concatenate([x_train, x_validation], axis=0)
    y_trainvalidation = np.concatenate([y_train, y_validation], axis=0)
    grid_search = GridSearchCV(
        estimator=sklearn.ensemble.RandomForestRegressor(random_state=0),
        param_grid=param_grid,
        cv=cv,
    )

    grid_search.fit(x_trainvalidation, y_trainvalidation)
    return grid_search.best_estimator_


def lasso_cv(x_train, y_train, x_validation, y_validation):
    cv = [((np.arange(len(x_train)), np.arange(len(x_train), len(x_train) + len(x_validation))))]
    x_trainvalidation = np.concatenate([x_train, x_validation], axis=0)
    y_trainvalidation = np.concatenate([y_train, y_validation], axis=0)
    lm = sklearn.linear_model.LassoCV(n_alphas=10, cv=cv)
    lm.fit(x_trainvalidation, y_trainvalidation)
    return lm


def xgboost_cv(x_train, y_train, x_validation, y_validation):
    import xgboost

    lm = xgboost.XGBRegressor(n_estimators=100, early_stopping_rounds=50)
    eval_set = [(x_train, y_train)]
    lm.fit(x_train, y_train, eval_set=eval_set, verbose=False)
    return lm


class Prediction(Flow):
    scores = chd.flow.SparseDataset()

    def initialize(self, peakcounts, transcriptome, folds, subset_cells=False):
        self.peakcounts = peakcounts
        self.transcriptome = transcriptome
        self.folds = folds

        self.phases = ["train", "validation", "test"]

        coords_pointed = {
            "region": self.transcriptome.var.index,
            "fold": pd.Index(range(len(self.folds)), name="fold"),
            "phase": pd.Index(self.phases, name="phase"),
        }
        coords_fixed = {}

        if not self.o.scores.exists(self):
            self.scores = chd.sparse.SparseDataset.create(
                self.path / "scores",
                variables={
                    "cor": {
                        "dimensions": ("region", "fold", "phase"),
                        "dtype": np.float32,
                        "sparse": False,
                    },
                    "time": {
                        "dimensions": ("region",),
                        "dtype": np.float32,
                        "sparse": False,
                        "fill_value": np.nan,
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

        self.subset_cells = subset_cells

    def score(
        self,
        predictor="linear",
        layer=None,
    ):
        if "dispersions_norm" in self.transcriptome.var.columns:
            gene_ids = self.transcriptome.var.sort_values("dispersions_norm", ascending=False).index
        else:
            gene_ids = self.transcriptome.var.index
        for gene_oi in tqdm.tqdm(gene_ids):
            if not self.scores["scored"][gene_oi].all():
                self._score_gene(gene_oi, predictor, layer)

    def _score_gene(self, gene_oi, predictor, layer):
        peaks, x = self.peakcounts.get_peak_counts(gene_oi)

        if layer is None:
            layer = list(self.transcriptome.layers.keys())[0]
        y = self.transcriptome.layers[layer][:, self.transcriptome.var.index == gene_oi][:, 0]

        start = time.time()

        for fold_ix, fold in enumerate(self.folds):
            if x.shape[1] > 0:
                cells_train = np.hstack([fold["cells_train"]])

                x_train = x[cells_train]
                x_validation = x[fold["cells_validation"]]
                x_test = x[fold["cells_test"]]

                y_train = y[cells_train]
                y_validation = y[fold["cells_validation"]]
                y_test = y[fold["cells_test"]]

                if predictor == "linear":
                    lm = sklearn.linear_model.LinearRegression()
                    try:
                        lm.fit(x_train, y_train)
                    except np.linalg.LinAlgError:
                        continue
                else:
                    if predictor == "lasso":
                        lm = lasso_cv(x_train, y_train, x_validation, y_validation)
                    elif predictor == "rf":
                        lm = rf_cv(x_train, y_train, x_validation, y_validation)
                    elif predictor == "ridge":
                        lm = sklearn.linear_model.RidgeCV(alphas=10)
                    elif predictor == "xgboost":
                        import xgboost

                        lm = xgboost_cv(x_train, y_train, x_validation, y_validation)
                    else:
                        raise ValueError(f"predictor {predictor} not recognized")

                cors = []

                y_predicted = lm.predict(x_train)
                cors.append(np.corrcoef(y_train, y_predicted)[0, 1])

                y_predicted = lm.predict(x_validation)
                cors.append(np.corrcoef(y_validation, y_predicted)[0, 1])

                y_predicted = lm.predict(x_test)
                cors.append(np.corrcoef(y_test, y_predicted)[0, 1])
            else:
                cors = [0, 0, 0]

            self.scores["cor"][gene_oi, fold_ix, "train"] = cors[0]
            self.scores["cor"][gene_oi, fold_ix, "validation"] = cors[1]
            self.scores["cor"][gene_oi, fold_ix, "test"] = cors[2]

            self.scores["scored"][gene_oi, fold_ix] = True

        end = time.time()
        self.scores["time"][gene_oi] = end - start

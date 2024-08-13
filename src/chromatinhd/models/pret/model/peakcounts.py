import numpy as np
import pandas as pd

import xgboost as xgb

import tqdm.auto as tqdm

import typing
import scipy.sparse

from chromatinhd.flow import Flow, Stored


class PeaksGene(Flow):
    default_name = "geneprediction"

    transcriptome: "typing.Any"
    peaks: "typing.Any"
    layer = Stored()

    def __init__(self, path, transcriptome, peaks, layer=None):
        super().__init__(path)
        self.transcriptome = transcriptome
        self.peaks = peaks
        self.layer = layer

    def _create_regressor(self, n, train_ix, validation_ix):
        regressor = xgb.XGBRegressor(n_estimators=100)
        return regressor

    def _preprocess_features(self, X):
        return X

    def score(self, peak_gene_links, folds):
        if self.layer is None:
            X_transcriptome = self.transcriptome.X
        else:
            X_transcriptome = self.transcriptome.layers[self.layer]

        if scipy.sparse.issparse(X_transcriptome):
            X_transcriptome = self.transcriptome.adata.X.tocsc()
        else:
            X_transcriptome = scipy.sparse.csc_matrix(X_transcriptome)

        X_peaks = self.peaks.counts.tocsc()

        var_transcriptome = self.transcriptome.var
        var_transcriptome["ix"] = np.arange(var_transcriptome.shape[0])

        var_peaks = self.peaks.var
        var_peaks["ix"] = np.arange(var_peaks.shape[0])

        def extract_data(gene_oi, peaks_oi):
            x = np.array(X_peaks[:, peaks_oi["ix"]].todense())
            y = np.array(X_transcriptome[:, gene_oi["ix"]].todense())[:, 0]
            return x, y

        obs = self.transcriptome.obs
        obs["ix"] = np.arange(obs.shape[0])

        scores = []

        for fold_ix, fold in enumerate(folds):
            train_ix, validation_ix, test_ix = (
                fold["cells_train"],
                fold["cells_validation"],
                fold["cells_test"],
            )

            for gene, peak_gene_links_oi in tqdm.tqdm(peak_gene_links.groupby("gene")):
                peaks_oi = var_peaks.loc[peak_gene_links_oi["peak"]]
                gene_oi = var_transcriptome.loc[gene]

                x, y = extract_data(gene_oi, peaks_oi)

                self.regressor = self._create_regressor(len(obs), train_ix, validation_ix)

                x = self._preprocess_features(x)
                x[np.isnan(x)] = 0.0

                task = [x, y, train_ix, validation_ix, test_ix]

                result = self._score(*task)
                score = pd.DataFrame(
                    {
                        "cor": result,
                        "phase": ["train", "validation", "test"],
                    }
                )
                score["gene"] = gene_oi.name
                score["fold"] = fold_ix
                scores.append(score)

        scores = pd.concat(scores, ignore_index=True).groupby(["phase", "gene"]).mean()

        self.scores = scores

    def _fit(self, x, y, train_ix, validation_ix):
        if x[train_ix].std() < 1e-5:
            x_ = x[train_ix] + np.random.normal(0, 1e-5, x[train_ix].shape)
        else:
            x_ = x[train_ix]
        if y[train_ix].std() < 1e-5:
            y_ = y[train_ix] + np.random.normal(0, 1e-5, y[train_ix].shape)
        else:
            y_ = y[train_ix]

        try:
            self.regressor.fit(x_, y_)
        except np.linalg.LinAlgError:
            self.regressor.fit(
                np.random.normal(0, 1e-5, y[train_ix].shape), np.random.normal(0, 1e-5, y[train_ix].shape)
            )

    def _score(
        self,
        x,
        y,
        train_ix,
        validation_ix,
        test_ix,
    ):
        print(x.shape[1])
        # if no peaks detected, simply predict mean
        if x.shape[1] == 0:
            predicted = np.repeat(
                y[train_ix].mean(),
                (len(train_ix) + len(validation_ix) + len(test_ix)),
            )
        else:
            self._fit(x, y, train_ix, validation_ix)
            predicted = self.regressor.predict(x)
            # except BaseException as e:
            #     print(e)
            #     predicted = np.repeat(
            #         y[train_ix].mean(),
            #         (len(train_ix) + len(validation_ix) + len(test_ix)),
            #     )

        # correlation
        # train
        if (y[train_ix].std() < 1e-5) or (predicted[train_ix].std() < 1e-5):
            cor_train = 0.0
        else:
            cor_train = np.corrcoef(y[train_ix], predicted[train_ix])[0, 1]

        # validation
        if (y[validation_ix].std() < 1e-5) or (predicted[validation_ix].std() < 1e-5):
            cor_validation = 0.0
        else:
            cor_validation = np.corrcoef(y[validation_ix], predicted[validation_ix])[0, 1]

        # test
        if (y[test_ix].std() < 1e-5) or (predicted[test_ix].std() < 1e-5):
            cor_test = 0.0
        else:
            cor_test = np.corrcoef(y[test_ix], predicted[test_ix])[0, 1]

        return [cor_train, cor_validation, cor_test]

    def get_scoring_folder(self):
        scores_folder = self.path / "scoring" / "performance"
        scores_folder.mkdir(exist_ok=True, parents=True)
        return scores_folder

    _scores = None

    @property
    def scores(self):
        if self._scores is None:
            scores_folder = self.get_scoring_folder()
            self._scores = pd.read_pickle(scores_folder / "genescores.pkl")
        return self._scores

    @scores.setter
    def scores(self, value):
        scores_folder = self.get_scoring_folder()
        value.to_pickle(scores_folder / "genescores.pkl")
        value.groupby("gene").mean().to_pickle(scores_folder / "scores.pkl")
        self._scores = value


class PeaksGeneLinear(PeaksGene):
    default_name = "geneprediction_linear"

    def _create_regressor(self, n, train_ix, validation_ix):
        import sklearn.linear_model

        regressor = sklearn.linear_model.LinearRegression()
        return regressor


class PeaksGeneLasso(PeaksGene):
    default_name = "geneprediction_lasso"

    def _create_regressor(self, n, train_ix, validation_ix):
        import sklearn.linear_model
        import sklearn.model_selection

        regressor = sklearn.linear_model.LassoCV(
            n_alphas=10,
            n_jobs=12,
            # cv=[(train_ix, validation_ix)],
        )
        return regressor

    def _fit(self, x, y, train_ix, validation_ix):
        if x[train_ix].std() < 1e-5:
            x_ = x[train_ix] + np.random.normal(0, 1e-5, x[train_ix].shape)
        else:
            x_ = x[train_ix]
        if y[train_ix].std() < 1e-5:
            y_ = y[train_ix] + np.random.normal(0, 1e-5, y[train_ix].shape)
        else:
            y_ = y[train_ix]
        self.regressor.fit(x_, y_)


class PeaksGenePolynomial(PeaksGeneLinear):
    default_name = "geneprediction_poly"

    def _create_regressor(self, n, train_ix, validation_ix):
        import sklearn.linear_model

        regressor = sklearn.linear_model.Ridge()
        return regressor

    def _preprocess_features(self, X):
        import sklearn.preprocessing

        poly = sklearn.preprocessing.PolynomialFeatures(interaction_only=True, include_bias=False)
        return poly.fit_transform(X)


class PeaksGeneXGBoost(PeaksGene):
    default_name = "geneprediction_xgboost"

    def _create_regressor(self, n, train_ix, validation_ix):
        import xgboost as xgb

        regressor = xgb.XGBRegressor(n_estimators=100, early_stopping_rounds=50)
        return regressor

    def _score(
        self,
        x,
        y,
        train_ix,
        validation_ix,
        test_ix,
    ):
        # if no peaks detected, simply predict mean
        if x.shape[1] == 0:
            predicted = np.repeat(
                y[train_ix].mean(),
                (len(train_ix) + len(validation_ix) + len(test_ix)),
            )
        else:
            try:
                eval_set = [(x[validation_ix], y[validation_ix])]
                self.regressor.fit(x[train_ix], y[train_ix], eval_set=eval_set, verbose=False)
                predicted = self.regressor.predict(x)
            except BaseException as e:
                print(e)
                predicted = np.repeat(
                    y[train_ix].mean(),
                    (len(train_ix) + len(validation_ix) + len(test_ix)),
                )

        # correlation
        # train
        if (y[train_ix].std() < 1e-5) or (predicted[train_ix].std() < 1e-5):
            cor_train = 0.0
        else:
            cor_train = np.corrcoef(y[train_ix], predicted[train_ix])[0, 1]

        # validation
        if (y[validation_ix].std() < 1e-5) or (predicted[validation_ix].std() < 1e-5):
            cor_validation = 0.0
        else:
            cor_validation = np.corrcoef(y[validation_ix], predicted[validation_ix])[0, 1]

        # test
        # print(predicted[test_ix].std())
        if (y[test_ix].std() < 1e-5) or (predicted[test_ix].std() < 1e-5):
            cor_test = 0.0
        else:
            cor_test = np.corrcoef(y[test_ix], predicted[test_ix])[0, 1]

        return [cor_train, cor_validation, cor_test]

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None

import tqdm.auto as tqdm
import typing

from chromatinhd.flow import Flow


def scoreit(regressor, x, y, train_ix, x_test, y_test):
    # if no peaks detected, simply predict mean
    if x.shape[1] == 0:
        predicted = np.repeat(
            y[train_ix].mean(),
            (len(y_test)),
        )
    else:
        try:
            regressor.fit(x[train_ix], y[train_ix])
            predicted = regressor.predict(x_test)
        except:
            predicted = np.repeat(
                y[train_ix].mean(),
                (len(y_test)),
            )

    # test
    if (y_test.std() < 1e-5) or (predicted.std() < 1e-5):
        cor_test = 0.0
    else:
        cor_test = np.corrcoef(y_test, predicted)[0, 1]

    return [cor_test]


class PeaksGene(Flow):
    default_name = "geneprediction"

    traintranscriptome: "typing.Any"
    trainpeaks: "typing.Any"
    testtranscriptome: "typing.Any"
    testpeaks: "typing.Any"

    def __init__(
        self,
        path,
        traintranscriptome,
        trainpeakcounts,
        testtranscriptome,
        testpeakcounts,
    ):
        super().__init__(path)
        self.traintranscriptome = traintranscriptome
        self.trainpeakcounts = trainpeakcounts
        self.testtranscriptome = testtranscriptome
        self.testpeakcounts = testpeakcounts

    def _create_regressor(self):
        regressor = xgb.XGBRegressor(n_estimators=100)
        return regressor

    def _preprocess_features(self, X):
        return X

    def score(self, peak_gene_links, folds):
        X_traintranscriptome = self.traintranscriptome.adata.X.tocsc()
        X_trainpeaks = self.trainpeakcounts.counts.tocsc()

        X_testtranscriptome = self.testtranscriptome.adata.X.tocsc()
        X_testpeaks = self.testpeakcounts.counts.tocsc()

        assert X_trainpeaks.shape[1] == X_testpeaks.shape[1]

        var_transcriptome = self.traintranscriptome.var
        var_transcriptome["ix"] = np.arange(var_transcriptome.shape[0])

        trainvar_peaks = self.trainpeakcounts.var
        trainvar_peaks["ix"] = np.arange(trainvar_peaks.shape[0])

        testvar_peaks = self.testpeakcounts.var
        testvar_peaks["ix"] = np.arange(testvar_peaks.shape[0])

        def extract_data(gene_oi, trainpeaks_oi, testpeaks_oi):
            x = np.array(X_trainpeaks[:, trainpeaks_oi["ix"]].todense())
            y = np.array(X_traintranscriptome[:, gene_oi["ix"]].todense())[:, 0]
            x_test = np.array(X_testpeaks[:, testpeaks_oi["ix"]].todense())
            y_test = np.array(X_testtranscriptome[:, gene_oi["ix"]].todense())[:, 0]
            return x, y, x_test, y_test

        regressor = self._create_regressor()

        scores = []

        for fold_ix, fold in enumerate(folds):
            train_ix = fold["cells_train"]

            for gene, peak_gene_links_oi in tqdm.tqdm(peak_gene_links.groupby("gene")):
                trainpeaks_oi = trainvar_peaks.loc[peak_gene_links_oi["peak"]]
                testpeaks_oi = testvar_peaks.loc[peak_gene_links_oi["peak"]]
                gene_oi = var_transcriptome.loc[gene]

                x, y, x_test, y_test = extract_data(gene_oi, trainpeaks_oi, testpeaks_oi)

                regressor = self._create_regressor()

                x = self._preprocess_features(x)
                x[np.isnan(x)] = 0.0
                x_test = self._preprocess_features(x_test)
                x_test[np.isnan(x_test)] = 0.0

                result = scoreit(regressor, x, y, train_ix, x_test, y_test)
                score = pd.DataFrame(
                    {
                        "cor": result,
                        "phase": ["test"],
                    }
                )
                score["gene"] = gene_oi.name
                score["fold"] = fold_ix
                scores.append(score)

        scores = pd.concat(scores, ignore_index=True).groupby(["phase", "gene"]).mean()

        self.scores = scores

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

    def _create_regressor(self):
        import sklearn.linear_model

        regressor = sklearn.linear_model.LinearRegression()
        return regressor


class PeaksGeneLasso(PeaksGene):
    default_name = "geneprediction_lasso"

    def _create_regressor(self):
        import sklearn.linear_model
        import sklearn.model_selection

        regressor = sklearn.linear_model.LassoCV(n_alphas=10, n_jobs=12)
        return regressor


class PeaksGeneXGBoost(PeaksGene):
    default_name = "geneprediction_xgboost"

    def _create_regressor(self):
        import xgboost as xgb

        regressor = xgb.XGBRegressor(n_estimators=100, early_stopping_rounds=50)
        return regressor

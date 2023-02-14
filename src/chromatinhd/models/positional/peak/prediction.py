import numpy as np
import pandas as pd

import xgboost as xgb

import tqdm.auto as tqdm

import typing
import scipy.sparse

from chromatinhd.flow import Flow


def scoreit(
    regressor,
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
            regressor.fit(x[train_ix], y[train_ix])
            predicted = regressor.predict(x)
        except:
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
    if (y[test_ix].std() < 1e-5) or (predicted[test_ix].std() < 1e-5):
        cor_test = 0.0
    else:
        cor_test = np.corrcoef(y[test_ix], predicted[test_ix])[0, 1]

    return [cor_train, cor_validation, cor_test]


class PeaksGene(Flow):
    default_name = "geneprediction"

    transcriptome: "typing.Any"
    peaks: "typing.Any"

    def __init__(self, path, transcriptome, peaks):
        super().__init__(path)
        self.transcriptome = transcriptome
        self.peaks = peaks

    def _create_regressor(self):
        regressor = xgb.XGBRegressor(n_estimators=100)
        return regressor

    def _preprocess_features(self, X):
        return X

    def score(self, peak_gene_links, folds):
        X_transcriptome = self.transcriptome.adata.X.tocsc()
        X_peaks = self.peaks.counts.tocsc()

        var_transcriptome = self.transcriptome.var
        var_transcriptome["ix"] = np.arange(var_transcriptome.shape[0])

        var_peaks = self.peaks.var
        var_peaks["ix"] = np.arange(var_peaks.shape[0])

        def extract_data(gene_oi, peaks_oi):
            x = np.array(X_peaks[:, peaks_oi["ix"]].todense())
            y = np.array(X_transcriptome[:, gene_oi["ix"]].todense())[:, 0]
            return x, y

        regressor = self._create_regressor()

        obs = self.transcriptome.obs
        obs["ix"] = np.arange(obs.shape[0])

        scores = []

        import multiprocess

        pool = multiprocess.Pool(5)

        for fold_ix, fold in enumerate(folds):
            train_ix, validation_ix, test_ix = (
                fold["cells_train"],
                fold["cells_validation"],
                fold["cells_test"],
            )

            genes = []
            futures = []

            for gene, peak_gene_links_oi in tqdm.tqdm(peak_gene_links.groupby("gene")):
                peaks_oi = var_peaks.loc[peak_gene_links_oi["peak"]]
                gene_oi = var_transcriptome.loc[gene]

                x, y = extract_data(gene_oi, peaks_oi)

                regressor = self._create_regressor()

                x = self._preprocess_features(x)
                x[np.isnan(x)] = 0.0

                task = [regressor, x, y, train_ix, validation_ix, test_ix]
                genes.append(gene_oi.name)

                futures.append(pool.apply_async(scoreit, args=task))

            for future, gene in zip(futures, genes):
                result = future.get()
                score = pd.DataFrame(
                    {
                        "cor": result,
                        "phase": ["train", "validation", "test"],
                    }
                )
                score["gene"] = gene
                score["fold"] = fold_ix
                scores.append(score)

        pool.close()

        scores = pd.concat(scores, ignore_index=True).groupby(["phase", "gene"]).mean()

        self.scores = scores

    def score(self, peak_gene_links, folds):
        if scipy.sparse.issparse(self.transcriptome.adata.X):
            X_transcriptome = self.transcriptome.adata.X.tocsc()
        else:
            X_transcriptome = scipy.sparse.csc_matrix(self.transcriptome.adata.X)
        X_peaks = self.peaks.counts.tocsc()

        var_transcriptome = self.transcriptome.var
        var_transcriptome["ix"] = np.arange(var_transcriptome.shape[0])

        var_peaks = self.peaks.var
        var_peaks["ix"] = np.arange(var_peaks.shape[0])

        def extract_data(gene_oi, peaks_oi):
            x = np.array(X_peaks[:, peaks_oi["ix"]].todense())
            y = np.array(X_transcriptome[:, gene_oi["ix"]].todense())[:, 0]
            return x, y

        regressor = self._create_regressor()

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

                regressor = self._create_regressor()

                x = self._preprocess_features(x)
                x[np.isnan(x)] = 0.0

                task = [regressor, x, y, train_ix, validation_ix, test_ix]

                result = scoreit(*task)
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

    def get_scoring_folder(self):
        scores_folder = self.path / "scoring" / "overall"
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


class PeaksGenePolynomial(PeaksGeneLinear):
    default_name = "geneprediction_poly"

    def _create_regressor(self):
        import sklearn.linear_model

        regressor = sklearn.linear_model.Ridge()
        return regressor

    def _preprocess_features(self, X):
        import sklearn.preprocessing

        poly = sklearn.preprocessing.PolynomialFeatures(
            interaction_only=True, include_bias=False
        )
        return poly.fit_transform(X)


class PeaksGeneLasso(PeaksGene):
    default_name = "geneprediction_lasso"

    def _create_regressor(self):
        import sklearn.linear_model

        regressor = sklearn.linear_model.Lasso()
        return regressor

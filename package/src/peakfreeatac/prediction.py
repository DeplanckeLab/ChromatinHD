import numpy as np
import pandas as pd

import scanpy as sc

import xgboost as xgb

import tqdm.auto as tqdm

import dataclasses
import pathlib
import typing

from peakfreeatac.flow import Flow


def split(n, seed=1, train_ratio=0.8):
    generator = np.random.RandomState(seed)
    train_ix = generator.random(n) < train_ratio
    validation_ix = ~train_ix

    return train_ix, validation_ix

def cal_mse(y, predicted):
    mse = np.sqrt(((predicted - y) ** 2).mean())
    return mse

class PeaksGene(Flow):
    default_name = "geneprediction"

    transcriptome:'typing.Any'
    peaks :'typing.Any'

    def __init__(self, path, transcriptome, peaks):
        super().__init__(path)
        self.transcriptome = transcriptome
        self.peaks = peaks

    def _create_regressor(self):
            regressor = xgb.XGBRegressor()
            return regressor

    def score(self, peak_gene_links, cells_train, cells_validation):
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

        train_ix = obs["ix"][cells_train]
        validation_ix = obs["ix"][cells_validation]

        scores = []
        for gene, peak_gene_links_oi in tqdm.tqdm(peak_gene_links.groupby("gene")):
            peaks_oi = var_peaks.loc[peak_gene_links_oi["peak"]]
            gene_oi = var_transcriptome.loc[gene]

            x, y = extract_data(gene_oi, peaks_oi)

            # if no peaks detected, simply predict mean
            if len(peaks_oi) == 0:
                predicted = np.repeat(y[train_ix].mean(), len(train_ix) + len(validation_ix))
            else:
                regressor.fit(x[train_ix], y[train_ix])
                predicted = regressor.predict(x)

            mse_train = cal_mse(y[train_ix], predicted[train_ix])
            mse_validation = cal_mse(y[validation_ix], predicted[validation_ix])
            mse_validation_dummy = cal_mse(y[validation_ix], y[validation_ix].mean())
            mse_train_dummy = cal_mse(y[train_ix], y[train_ix].mean())

            # correlation
            if (y[train_ix].std() < 1e-5) or (predicted[train_ix].std() < 1e-5):
                cor_train = 0.
            else:
                cor_train = np.corrcoef(y[train_ix], predicted[train_ix])[0, 1]
            if (y[validation_ix].std() < 1e-5) or (predicted[validation_ix].std() < 1e-5):
                cor_validation = 0.
            else:
                cor_validation = np.corrcoef(y[validation_ix], predicted[validation_ix])[0, 1]

            scores.append(
                pd.DataFrame({
                    "gene": gene,
                    "split_ix": 1,
                    "mse": [mse_train, mse_validation],
                    "cor":[cor_train, cor_validation],
                    "mse_dummy": [mse_train_dummy, mse_validation_dummy],
                    "phase":["train", "validation"],
                })
            )

        scores = pd.concat(scores, ignore_index= True).set_index(["phase", "gene"])

        self.scores = scores

    _scores = None
    @property
    def scores(self):
        if self._scores is None:
            self._scores = pd.read_table(self.path / "scores.tsv", index_col = [0, 1])
        return self._scores
    @scores.setter
    def scores(self, value):
        value.to_csv(self.path / "scores.tsv", sep = "\t")
        self._scores = value


class PeaksGeneLinear(PeaksGene):
    default_name = "geneprediction_linear"

    def _create_regressor(self):
            import sklearn.linear_model
            regressor = sklearn.linear_model.LinearRegression()
            return regressor

class OriginalPeakPrediction():
    default_name = "originalpeakprediction"

    def score(self):
        X_transcriptome = self.transcriptome.adata.X.tocsc().todense()
        X_peaks = self.peaks.X.tocsc().todense()

        var_transcriptome = self.transcriptome.adata.var
        var_transcriptome["ix"] = np.arange(var_transcriptome.shape[0])

        var_peaks = self.peaks.var
        var_peaks["ix"] = np.arange(var_peaks.shape[0])

        gene_peak_links = self.transcriptome.gene_peak_links

        def extract_data(gene_oi, peaks_oi):
            x = np.array(X_peaks[:, peaks_oi["ix"]])
            y = np.array(X_transcriptome[:, gene_oi["ix"]])[:, 0]
            return x, y

        def cal_mse(y, predicted):
            mse = np.sqrt(((predicted - y) ** 2).mean())
            return mse

        genes = var_transcriptome.index

        def create_regressor():
            import sklearn.linear_model
            regressor = sklearn.linear_model.Ridge()
            # regressor = xgb.XGBRegressor()
            return regressor

        regressor = create_regressor()

        scores = []
        for gene in tqdm.tqdm(genes):
            gene_oi = var_transcriptome.loc[gene]

            original_peak_ids = gene_peak_links.query("gene == @gene")["peak"]
            for original_peak, peaks_oi in var_peaks.loc[var_peaks["original_peak"].isin(original_peak_ids)].groupby("original_peak"):
                x, y = extract_data(gene_oi, peaks_oi)

                for i in range(1):
                    train_ix, validation_ix = split(x.shape[0], seed=i)
                    regressor.fit(x[train_ix], y[train_ix])

                    predicted = regressor.predict(x)
                    mse_train = cal_mse(y[train_ix], predicted[train_ix])
                    mse_validation = cal_mse(y[validation_ix], predicted[validation_ix])
                    mse_dummy = cal_mse(y[validation_ix], y[validation_ix].mean())
                    mse_train_mean = cal_mse(y[train_ix], y[train_ix].mean())

                    scores.append(
                        {
                            "gene": gene,
                            "original_peak":original_peak,
                            "split_ix": i,
                            "mse_train": mse_train,
                            "mse_validation": mse_validation,
                            "mse_dummy": mse_dummy,
                            "mse_train_mean": mse_train_mean,
                        }
                    )

        scores = pd.DataFrame(scores)
        scores["mse_validation_ratio"] = scores["mse_validation"] / scores["mse_dummy"]
        scores["mse_train_ratio"] = scores["mse_train"] / scores["mse_train_mean"]

        self.store("scores", scores)
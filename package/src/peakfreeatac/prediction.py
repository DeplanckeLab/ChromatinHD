import laflow as laf

import numpy as np
import pandas as pd

import scanpy as sc

import xgboost as xgb

import tqdm.auto as tqdm


def split(n, seed=1, train_ratio=0.8):
    generator = np.random.RandomState(seed)
    train_ix = generator.random(n) < train_ratio
    test_ix = ~train_ix

    return train_ix, test_ix

def cal_mse(y, predicted):
    mse = np.sqrt(((predicted - y) ** 2).mean())
    return mse


class GenePrediction(laf.Flow):
    default_name = "geneprediction"

    transcriptome = laf.FlowObj()
    accessibility = laf.FlowObj()

    def score(self):
        X_transcriptome = self.transcriptome.adata.X.tocsc().todense()
        X_accessibility = self.accessibility.adata.X.tocsc().todense()

        var_transcriptome = self.transcriptome.adata.var
        var_transcriptome["ix"] = np.arange(var_transcriptome.shape[0])

        var_accessibility = self.accessibility.var
        var_accessibility["ix"] = np.arange(var_accessibility.shape[0])

        def extract_data(gene_oi, peaks_oi):
            x = np.array(X_accessibility[:, peaks_oi["ix"]])
            y = np.array(X_transcriptome[:, gene_oi["ix"]])[:, 0]
            return x, y

        genes = var_accessibility["gene"].unique()

        def create_regressor():
            regressor = xgb.XGBRegressor()
            # regressor = lgb.LGBMRegressor()
            # import sklearn.linear_model
            # regressor = sklearn.linear_model.LinearRegression()
            # regressor = sklearn.ensemble.RandomForestRegressor()
            return regressor

        regressor = create_regressor()

        scores = []
        for gene in tqdm.tqdm(genes):
            peaks_oi = var_accessibility.query("gene == @gene")
            gene_oi = var_transcriptome.loc[gene]

            x, y = extract_data(gene_oi, peaks_oi)

            for i in range(1):
                train_ix, test_ix = split(x.shape[0], seed=i)
                regressor.fit(x[train_ix], y[train_ix])

                predicted = regressor.predict(x)
                mse_train = cal_mse(y[train_ix], predicted[train_ix])
                mse_test = cal_mse(y[test_ix], predicted[test_ix])
                mse_test_mean = cal_mse(y[test_ix], y[test_ix].mean())
                mse_train_mean = cal_mse(y[train_ix], y[train_ix].mean())

                scores.append(
                    {
                        "gene": gene,
                        "split_ix": i,
                        "mse_train": mse_train,
                        "mse_test": mse_test,
                        "mse_test_mean": mse_test_mean,
                        "mse_train_mean": mse_train_mean,
                    }
                )

        scores = pd.DataFrame(scores)
        scores["mse_test_ratio"] = scores["mse_test"] / scores["mse_test_mean"]
        scores["mse_train_ratio"] = scores["mse_train"] / scores["mse_train_mean"]

        self.store("scores", scores)




class OriginalPeakPrediction(laf.Flow):
    default_name = "originalpeakprediction"

    transcriptome = laf.FlowObj()
    accessibility = laf.FlowObj()

    def score(self):
        X_transcriptome = self.transcriptome.adata.X.tocsc().todense()
        X_accessibility = self.accessibility.adata.X.tocsc().todense()

        var_transcriptome = self.transcriptome.adata.var
        var_transcriptome["ix"] = np.arange(var_transcriptome.shape[0])

        var_accessibility = self.accessibility.var
        var_accessibility["ix"] = np.arange(var_accessibility.shape[0])

        gene_peak_links = self.transcriptome.gene_peak_links

        def extract_data(gene_oi, peaks_oi):
            x = np.array(X_accessibility[:, peaks_oi["ix"]])
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
            for original_peak, peaks_oi in var_accessibility.loc[var_accessibility["original_peak"].isin(original_peak_ids)].groupby("original_peak"):
                x, y = extract_data(gene_oi, peaks_oi)

                for i in range(1):
                    train_ix, test_ix = split(x.shape[0], seed=i)
                    regressor.fit(x[train_ix], y[train_ix])

                    predicted = regressor.predict(x)
                    mse_train = cal_mse(y[train_ix], predicted[train_ix])
                    mse_test = cal_mse(y[test_ix], predicted[test_ix])
                    mse_test_mean = cal_mse(y[test_ix], y[test_ix].mean())
                    mse_train_mean = cal_mse(y[train_ix], y[train_ix].mean())

                    scores.append(
                        {
                            "gene": gene,
                            "original_peak":original_peak,
                            "split_ix": i,
                            "mse_train": mse_train,
                            "mse_test": mse_test,
                            "mse_test_mean": mse_test_mean,
                            "mse_train_mean": mse_train_mean,
                        }
                    )

        scores = pd.DataFrame(scores)
        scores["mse_test_ratio"] = scores["mse_test"] / scores["mse_test_mean"]
        scores["mse_train_ratio"] = scores["mse_train"] / scores["mse_train_mean"]

        self.store("scores", scores)
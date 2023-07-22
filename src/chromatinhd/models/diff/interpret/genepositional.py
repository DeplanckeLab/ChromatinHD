import chromatinhd as chd
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import scipy.stats
import tqdm.auto as tqdm
import torch


def fdr(p_vals):
    from scipy.stats import rankdata

    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr


class GenePositional(chd.flow.Flow):
    genes = chd.flow.Stored("genes", default=set)

    def score(
        self,
        fragments,
        clustering,
        models,
        folds,
        genes=None,
        force=False,
        device="cuda",
    ):
        force_ = force

        if genes is None:
            genes = fragments.var.index

        pbar = tqdm.tqdm(genes, leave=False)

        window = fragments.regions.window

        for gene in pbar:
            pbar.set_description(gene)
            probs_file = self.get_scoring_path(gene) / "probs.pkl"

            force = force_
            if not probs_file.exists():
                force = True

            if force:
                design_gene = pd.DataFrame(
                    {"gene_ix": [fragments.var.index.get_loc(gene)]}
                ).astype("category")
                design_gene.index = pd.Series([gene], name="gene")
                design_clustering = pd.DataFrame(
                    {"active_cluster": np.arange(clustering.n_clusters)}
                ).astype("category")
                design_clustering.index = clustering.cluster_info.index
                design_coord = pd.DataFrame(
                    {"coord": np.arange(window[0], window[1] + 1, step=25)}
                ).astype("category")
                design_coord.index = design_coord["coord"]
                design = chd.utils.crossing(
                    design_gene, design_clustering, design_coord
                )

                batch_size = 5000
                design["batch"] = np.floor(
                    np.arange(design.shape[0]) / batch_size
                ).astype(int)

                probs = []
                for model in models:
                    probs_model = []
                    for _, design_subset in design.groupby("batch"):
                        pseudocoordinates = torch.from_numpy(
                            design_subset["coord"].values.astype(int)
                        )
                        pseudocoordinates = (pseudocoordinates - window[0]) / (
                            window[1] - window[0]
                        )
                        pseudocluster = torch.nn.functional.one_hot(
                            torch.from_numpy(
                                design_subset["active_cluster"].values.astype(int)
                            ),
                            clustering.n_clusters,
                        ).to(torch.float)
                        gene_ix = torch.from_numpy(
                            design_subset["gene_ix"].values.astype(int)
                        )

                        prob = model.evaluate_pseudo(
                            pseudocoordinates,
                            clustering=pseudocluster,
                            gene_ix=gene_ix,
                        )

                        probs_model.append(prob.numpy())
                    probs_model = np.hstack(probs_model)
                    probs.append(probs_model)

                probs = np.vstack(probs)
                probs = probs.mean(axis=0)

                probs = xr.DataArray(
                    probs.reshape(  # we have only one gene anyway
                        (
                            design_clustering.shape[0],
                            design_coord.shape[0],
                        )
                    ),
                    coords=[
                        design_clustering.index,
                        design_coord.index,
                    ],
                )

                pickle.dump(probs, probs_file.open("wb"))

                self.genes = self.genes | {gene}

    def get_plotdata(self, gene):
        """
        Returns the plotdata for a given gene

        Parameters:
            gene:
                the gene

        Returns:
            Two dataframes, one with the probabilities per cluster, one with the mean
        """
        probs_file = self.get_scoring_path(gene) / "probs.pkl"
        if not probs_file.exists():
            raise FileNotFoundError(f"File {probs_file} does not exist")

        probs = pickle.load(probs_file.open("rb"))
        plotdata = probs.to_dataframe("prob")

        window = probs.coords["coord"].values[[0, -1]]

        plotdata["prob"] = (
            plotdata["prob"]
            - np.log(
                plotdata.reset_index()
                .groupby(["cluster"])
                .apply(
                    lambda x: np.trapz(
                        np.exp(x["prob"]),
                        x["coord"].astype(float) / (window[1] - window[0]),
                    )
                )
            ).mean()
        )
        plotdata_mean = plotdata[["prob"]].groupby("coord").mean()

        return plotdata, plotdata_mean

    def get_scoring_path(self, gene):
        path = self.path / f"{gene}"
        path.mkdir(parents=True, exist_ok=True)
        return path

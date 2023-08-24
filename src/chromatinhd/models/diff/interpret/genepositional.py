import chromatinhd as chd
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import tqdm.auto as tqdm
import torch
from chromatinhd import get_default_device
from chromatinhd.data.clustering import Clustering
from chromatinhd.data.fragments import Fragments
from chromatinhd.models.diff.model.cutnf import Models

from chromatinhd.flow import Flow
from chromatinhd.flow.objects import Stored, DataArray, StoredDict


def fdr(p_vals):
    from scipy.stats import rankdata

    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr


class GenePositional(chd.flow.Flow):
    """
    Positional interpretation of *diff* models
    """

    probs = StoredDict(DataArray)

    def score(
        self,
        fragments: Fragments,
        clustering: Clustering,
        models: Models,
        genes: list = None,
        force: bool = False,
        device: str = None,
        step: int = 25,
        batch_size: int = 5000,
    ):
        """
        Main scoring function

        Parameters:
            fragments:
                the fragments
            clustering:
                the clustering
            models:
                the models
            genes:
                the genes to score, if None, all genes are scored
            force:
                whether to force rescoring even if the scores already exist
            device:
                the device to use
        """
        force_ = force

        if genes is None:
            genes = fragments.var.index

        pbar = tqdm.tqdm(genes, leave=False)

        window = fragments.regions.window

        if device is None:
            device = get_default_device()

        for gene in pbar:
            pbar.set_description(gene)

            force = force_
            if gene in self.probs:
                force = True

            if force:
                design_region = pd.DataFrame({"region_ix": [fragments.var.index.get_loc(gene)]}).astype("category")
                design_region.index = pd.Series([gene], name="gene")
                design_clustering = pd.DataFrame({"active_cluster": np.arange(clustering.n_clusters)}).astype(
                    "category"
                )
                design_clustering.index = clustering.cluster_info.index
                design_coord = pd.DataFrame({"coord": np.arange(window[0], window[1] + 1, step=step)}).astype(
                    "category"
                )
                design_coord.index = design_coord["coord"]
                design = chd.utils.crossing(design_region, design_clustering, design_coord)

                design["batch"] = np.floor(np.arange(design.shape[0]) / batch_size).astype(int)

                probs = []
                for model in models:
                    probs_model = []
                    for _, design_subset in design.groupby("batch"):
                        pseudocoordinates = torch.from_numpy(design_subset["coord"].values.astype(int))
                        pseudocoordinates = (pseudocoordinates - window[0]) / (window[1] - window[0])
                        pseudocluster = torch.nn.functional.one_hot(
                            torch.from_numpy(design_subset["active_cluster"].values.astype(int)),
                            clustering.n_clusters,
                        ).to(torch.float)
                        region_ix = torch.from_numpy(design_subset["region_ix"].values.astype(int))

                        prob = model.evaluate_pseudo(
                            pseudocoordinates,
                            clustering=pseudocluster,
                            region_ix=region_ix,
                            device=device,
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

                self.probs[gene] = probs

        return self

    def get_plotdata(self, gene: str) -> (pd.DataFrame, pd.DataFrame):
        """
        Returns average and differential probabilities for a particular gene.

        Parameters:
            gene:
                the gene

        Returns:
            Two dataframes, one with the probabilities per cluster, one with the mean
        """
        probs = self.probs[gene]

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

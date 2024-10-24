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
from chromatinhd.flow.objects import Stored, DataArray, StoredDict, Linked


def fdr(p_vals):
    from scipy.stats import rankdata

    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr


class Slices:
    """
    Stores data of slices within regions

    Parameters
    ----------
    region_ixs : np.ndarray
        Region indices
    start_position_ixs : np.ndarray
        Start position indices
    end_position_ixs : np.ndarray
        End position indices
    data : np.ndarray
        Data of slices
    n_regions : int
        Number of regions
    step: int
        Step size
    window: tuple
        Window of the region
    """

    step = 1

    def __init__(self, region_ixs, start_position_ixs, end_position_ixs, data, n_regions, step, window):
        self.region_ixs = region_ixs
        self.start_position_ixs = start_position_ixs
        self.end_position_ixs = end_position_ixs
        self.n_regions = n_regions
        self.data = data
        self.window = window
        indptr = np.concatenate(
            [
                [0],
                np.cumsum(end_position_ixs - start_position_ixs),
            ],
            axis=0,
        )
        self.indptr = indptr
        self.step = step

    def get_slice_scores(self, regions=None, min_length=10):
        slicescores = pd.DataFrame(
            {
                "start": self.start_position_ixs * self.step + self.window[0],
                "end": self.end_position_ixs * self.step + self.window[0],
                "region_ix": self.region_ixs,
            }
        )
        if self.data.shape[0] == len(self.start_position_ixs):
            slicescores["score"] = self.data
        else:
            pass
        if regions is not None:
            slicescores["region"] = pd.Categorical(
                regions.coordinates.index[self.region_ixs], regions.coordinates.index
            )
        slicescores["length"] = slicescores["end"] - slicescores["start"]
        slicescores = slicescores.loc[slicescores["length"] >= min_length]

        slicescores.index = (
            slicescores["region_ix"].astype(str)
            + ":"
            + slicescores["start"].astype(str)
            + "-"
            + slicescores["end"].astype(str)
        )
        return slicescores


class DifferentialSlices:
    """
    Stores data of slices within regions linked to a specific cluster

    Parameters
    ----------
    region_ixs : np.ndarray
        Region indices
    cluster_ixs : np.ndarray
        Cluster indices
    start_position_ixs : np.ndarray
        Start position indices
    end_position_ixs : np.ndarray
        End position indices
    data : np.ndarray
        Data of slices
    n_regions : int
        Number of regions
    step : int
        Step size
    window: tuple
        Window of the region
    """

    step = 1

    def __init__(self, region_ixs, cluster_ixs, start_position_ixs, end_position_ixs, data, n_regions, step, window):
        self.region_ixs = region_ixs
        self.start_position_ixs = start_position_ixs
        self.end_position_ixs = end_position_ixs
        self.n_regions = n_regions
        self.cluster_ixs = cluster_ixs
        self.data = data
        self.step = step
        self.window = window
        indptr = np.concatenate(
            [
                [0],
                np.cumsum(end_position_ixs - start_position_ixs),
            ],
            axis=0,
        )
        self.indptr = indptr

    def get_slice_scores(self, regions=None, clustering=None, cluster_info=None, min_length=10):
        slicescores = pd.DataFrame(
            {
                "start": self.start_position_ixs * self.step + self.window[0],
                "end": self.end_position_ixs * self.step + self.window[0],
                "region_ix": self.region_ixs,
                "cluster_ix": self.cluster_ixs,
            }
        )
        if self.data.shape[0] == len(self.start_position_ixs):
            slicescores["score"] = self.data
        else:
            pass

        if clustering is not None:
            if cluster_info is None:
                cluster_info = clustering.cluster_info
        if cluster_info is not None:
            slicescores["cluster"] = pd.Categorical(cluster_info.index[self.cluster_ixs], cluster_info.index)
        if regions is not None:
            slicescores["region"] = pd.Categorical(
                regions.coordinates.index[self.region_ixs], regions.coordinates.index
            )
        slicescores["length"] = slicescores["end"] - slicescores["start"]
        slicescores = slicescores.loc[slicescores["length"] >= min_length]
        slicescores.index = (
            slicescores["region_ix"].astype(str)
            + ":"
            + slicescores["start"].astype(str)
            + "-"
            + slicescores["end"].astype(str)
        )
        slicescores.index.name = "slice"
        return slicescores


class DifferentialPeaks:
    """
    Stores data of slices within regions linked to a specific cluster

    Parameters
    ----------
    region_ixs : np.ndarray
        Region indices
    cluster_ixs : np.ndarray
        Cluster indices
    start_position_ixs : np.ndarray
        Start position indices
    end_position_ixs : np.ndarray
        End position indices
    data : np.ndarray
        Data of slices
    n_regions : int
        Number of regions
    """

    step = 1

    def __init__(self, region_ixs, cluster_ixs, start_position_ixs, end_position_ixs, data, n_regions):
        self.region_ixs = region_ixs
        self.start_position_ixs = start_position_ixs
        self.end_position_ixs = end_position_ixs
        self.n_regions = n_regions
        self.cluster_ixs = cluster_ixs
        self.data = data

    def get_slice_scores(self, regions=None, clustering=None, cluster_info=None, min_length=10):
        slicescores = pd.DataFrame(
            {
                "start": self.start_position_ixs * self.step + self.window[0],
                "end": self.end_position_ixs * self.step + self.window[0],
                "region_ix": self.region_ixs,
                "cluster_ix": self.cluster_ixs,
                "score": self.data,
            }
        )

        if clustering is not None:
            cluster_info = clustering.cluster_info
        if cluster_info is not None:
            slicescores["cluster"] = pd.Categorical(cluster_info.index[self.cluster_ixs], cluster_info.index)
        if regions is not None:
            slicescores["region"] = pd.Categorical(
                regions.coordinates.index[self.region_ixs], regions.coordinates.index
            )
        slicescores["length"] = slicescores["end"] - slicescores["start"]
        slicescores = slicescores.loc[slicescores["length"] >= min_length]
        slicescores.index = (
            slicescores["region_ix"].astype(str)
            + ":"
            + slicescores["start"].astype(str)
            + "-"
            + slicescores["end"].astype(str)
        )
        return slicescores


class RegionPositional(chd.flow.Flow):
    """
    Positional interpretation of *diff* models
    """

    regions = Linked()
    clustering = Linked()

    probs = StoredDict(DataArray)

    def score(
        self,
        models: Models,
        fragments: Fragments = None,
        clustering: Clustering = None,
        regions_oi: list = None,
        force: bool = False,
        device: str = "cpu",
        step: int = 50,
        batch_size: int = 5000,
        normalize_per_cell: int = 100,
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
            regions_oi:
                the regions to score, if None, all regions are scored
            force:
                whether to force rescoring even if the scores already exist
            device:
                the device to use
        """
        force_ = force

        if fragments is None:
            fragments = models.fragments
        if clustering is None:
            clustering = models.clustering

        if regions_oi is None:
            regions_oi = fragments.var.index

        self.regions = fragments.regions

        pbar = tqdm.tqdm(regions_oi, leave=False)

        window = fragments.regions.window

        if device is None:
            device = get_default_device()

        for region in pbar:
            pbar.set_description(region)

            force = force_
            if region not in self.probs:
                force = True

            if force:
                design_region = pd.DataFrame({"region_ix": [fragments.var.index.get_loc(region)]}).astype("category")
                design_region.index = pd.Series([region], name="region")
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

                if len(models) == 0:
                    raise ValueError("No models to score")
                for model in models:
                    probs_model = []
                    for _, design_subset in design.groupby("batch"):
                        pseudocoordinates = torch.from_numpy(design_subset["coord"].values.astype(int))
                        # pseudocoordinates = (pseudocoordinates - window[0]) / (window[1] - window[0])
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
                            normalize_per_cell=normalize_per_cell,
                        )

                        probs_model.append(prob.numpy())
                    probs_model = np.hstack(probs_model)
                    probs.append(probs_model)

                probs = np.vstack(probs)
                probs = probs.mean(axis=0)

                probs = xr.DataArray(
                    probs.reshape(  # we have only one region anyway
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

                self.probs[region] = probs

        return self

    def get_plotdata(self, region: str, clusters=None, relative_to=None, scale = 1.) -> (pd.DataFrame, pd.DataFrame):
        """
        Returns average and differential probabilities for a particular region.

        Parameters:
            region:
                the region

        Returns:
            Two dataframes, one with the probabilities per cluster, one with the mean
        """
        probs = self.probs[region]

        if clusters is not None:
            probs = probs.sel(cluster=clusters)

        plotdata = probs.to_dataframe("prob")
        # plotdata["prob"] = plotdata["prob"] * scale - plotdata["prob"].mean() * scale

        if relative_to is not None:
            plotdata_mean = plotdata[["prob"]].query("cluster in @relative_to").groupby("coord", observed=False).mean()
        else:
            plotdata_mean = plotdata[["prob"]].groupby("coord", observed=True).mean()


        return plotdata, plotdata_mean

    @property
    def scored(self):
        return len(self.probs) > 0

    def calculate_slices(self, prob_cutoff=1.5, clusters_oi=None, cluster_grouping=None, step=1):
        start_position_ixs = []
        end_position_ixs = []
        data = []
        region_ixs = []

        if clusters_oi is not None:
            if isinstance(clusters_oi, (pd.Series, pd.Index)):
                clusters_oi = clusters_oi.tolist()

        for region, probs in tqdm.tqdm(self.probs.items(), leave=False, total=len(self.probs)):
            if clusters_oi is not None:
                probs = probs.sel(cluster=clusters_oi)

            if cluster_grouping is not None:
                probs = probs.groupby(cluster_grouping).mean()

            region_ix = self.regions.var.index.get_loc(region)
            desired_x = np.arange(*self.regions.window, step=step) - self.regions.window[0]
            x = probs.coords["coord"].values - self.regions.window[0]
            y = probs.values

            y_interpolated = chd.utils.interpolate_1d(
                torch.from_numpy(desired_x), torch.from_numpy(x), torch.from_numpy(y)
            ).numpy()

            # from y_interpolated, determine start and end positions of the relevant slices
            start_position_ixs_region, end_position_ixs_region, data_region = extract_slices(
                y_interpolated, prob_cutoff
            )
            start_position_ixs.append(start_position_ixs_region)
            end_position_ixs.append(end_position_ixs_region)
            data.append(data_region)
            region_ixs.append(np.ones(len(start_position_ixs_region), dtype=int) * region_ix)
        data = np.concatenate(data, axis=0)
        start_position_ixs = np.concatenate(start_position_ixs, axis=0)
        end_position_ixs = np.concatenate(end_position_ixs, axis=0)
        region_ixs = np.concatenate(region_ixs, axis=0)

        slices = Slices(
            region_ixs,
            start_position_ixs,
            end_position_ixs,
            data,
            self.regions.n_regions,
            step=step,
            window=self.regions.window,
        )
        return slices

    def calculate_differential_slices(self, slices, fc_cutoff=2.0, score="diff", a=None, b=None, n=None, expand = 0):
        if score == "diff":
            data_diff = slices.data - slices.data.mean(1, keepdims=True)
            data_selected = data_diff > np.log(fc_cutoff)
        elif score == "diff2":
            data_diff = slices.data - slices.data.mean(1, keepdims=True)

            data_selected = (
                (data_diff > np.log(4.0))
                | ((data_diff > np.log(3.0)) & (slices.data > 0.0))
                | ((data_diff > np.log(2.0)) & (slices.data > 1.0))
            )
            # data_diff = data_diff * np.exp(slices.data.mean(1, keepdims=True))
        elif score == "diff3":
            probs_mean = slices.data.mean(1, keepdims=True)
            actual = slices.data
            diff = slices.data - probs_mean

            x1, y1 = a
            x2, y2 = b

            X = diff
            Y = actual

            data_diff = -((x2 - x1) * (Y - y1) - (y2 - y1) * (X - x1))
        else:
            raise ValueError(f"Unknown score {score}")

        if n is None:
            data_selected = data_diff > np.log(fc_cutoff)
        else:
            cutoff = np.quantile(data_diff, 1 - n / data_diff.shape[0], axis=0, keepdims=True)
            data_selected = data_diff > cutoff

        region_indices = np.repeat(slices.region_ixs, slices.end_position_ixs - slices.start_position_ixs)
        position_indices = np.concatenate(
            [np.arange(start, end) for start, end in zip(slices.start_position_ixs, slices.end_position_ixs)]
        )

        positions = []
        region_ixs = []
        cluster_ixs = []
        for ct_ix in range(data_diff.shape[1]):
            # select which data is relevant
            oi = data_selected[:, ct_ix]
            if oi.sum() == 0:
                continue
            positions_oi = position_indices[oi]
            regions_oi = region_indices[oi]

            start = np.where(
                np.pad(np.diff(positions_oi) != 1, (1, 0), constant_values=True)
                | np.pad(np.diff(regions_oi) != 0, (1, 0), constant_values=True)
            )[0]
            end = np.pad(start[1:], (0, 1), constant_values=len(positions_oi)) - 1

            positions.append(np.stack([positions_oi[start], positions_oi[end]], axis=1))
            region_ixs.append(regions_oi[start])
            cluster_ixs.append(np.ones(len(start), dtype=int) * ct_ix)
        start_position_ixs, end_position_ixs = np.concatenate(positions, axis=0).T
        region_ixs = np.concatenate(region_ixs, axis=0)
        cluster_ixs = np.concatenate(cluster_ixs, axis=0)

        if expand > 0:
            start_position_ixs = start_position_ixs - expand
            end_position_ixs = end_position_ixs + expand

        differential_slices = DifferentialSlices(
            region_ixs,
            cluster_ixs,
            start_position_ixs,
            end_position_ixs,
            data_diff,
            self.regions.n_regions,
            step=slices.step,
            window=slices.window,
        )
        return differential_slices

    def calculate_top_slices(self, slices, fc_cutoff=2.0):
        data_diff = slices.data - slices.data.mean(1, keepdims=True)

        region_indices = np.repeat(slices.region_ixs, slices.end_position_ixs - slices.start_position_ixs)
        position_indices = np.concatenate(
            [np.arange(start, end) for start, end in zip(slices.start_position_ixs, slices.end_position_ixs)]
        )

        # select which data is relevant
        oi = data_diff[:,].max(1) > np.log(fc_cutoff)
        positions_oi = position_indices[oi]
        regions_oi = region_indices[oi]

        start = np.where(
            np.pad(np.diff(positions_oi) != 1, (1, 0), constant_values=True)
            | np.pad(np.diff(regions_oi) != 0, (1, 0), constant_values=True)
        )[0]
        end = np.pad(start[1:], (0, 1), constant_values=len(positions_oi)) - 1

        region_ixs = regions_oi[start]
        data = data_diff[oi].max(1)

        start_position_ixs, end_position_ixs = positions_oi[start], positions_oi[end]

        differential_slices = Slices(
            region_ixs,
            start_position_ixs,
            end_position_ixs,
            data,
            self.regions.n_regions,
            step=slices.step,
            window=slices.window,
        )
        return differential_slices

    def select_windows(self, region_id, max_merge_distance=500, min_length=50, padding=500, prob_cutoff=1.5, differential_prob_cutoff=None, keep_tss = False):
        """
        Select windows based on the number of fragments

        Parameters:
            region_id:
                the identifier of the region of interest
            max_merge_distance:
                the maximum distance between windows before merging
            min_length:
                the minimum length of a window
            padding:
                the padding to add to each window
            prob_cutoff:
                the probability cutoff
            differential_prob_cutoff:
                the differential probability cutoff
        """

        from scipy.ndimage import convolve

        def spread_true(arr, width=5):
            kernel = np.ones(width, dtype=bool)
            result = convolve(arr, kernel, mode="constant", cval=False)
            result = result != 0
            return result

        plotdata, plotdata_mean = self.get_plotdata(region_id)
        selection = pd.DataFrame({"chosen": (plotdata["prob"].unstack() > prob_cutoff).any()})

        if differential_prob_cutoff is not None:
            plotdata_diff = (plotdata - plotdata_mean)["prob"].unstack()
            selection["chosen_differential"] = selection["chosen"] & (np.exp(plotdata_diff.values) > differential_prob_cutoff).any(0)

        # add padding
        step = plotdata.index.get_level_values("coord")[1] - plotdata.index.get_level_values("coord")[0]
        k_padding = padding // step
        selection["chosen"] = spread_true(selection["chosen"], width=k_padding)

        # select all contiguous regions where chosen is true
        selection["selection"] = selection["chosen"].cumsum()

        windows = pd.DataFrame(
            {
                "start": selection.index[
                    (np.diff(np.pad(selection["chosen"], (1, 1), constant_values=False).astype(int)) == 1)[:-1]
                ].astype(int),
                "end": selection.index[
                    (np.diff(np.pad(selection["chosen"], (1, 1), constant_values=False).astype(int)) == -1)[1:]
                ].astype(int),
            }
        )

        # merge windows that are close to each other
        windows["distance_to_next"] = windows["start"].shift(-1) - windows["end"]

        windows["merge"] = (windows["distance_to_next"] < max_merge_distance).fillna(False)
        windows["group"] = (~windows["merge"]).cumsum().shift(1).fillna(0).astype(int)
        windows = (
            windows.groupby("group")
            .agg({"start": "min", "end": "max", "distance_to_next": "last"})
            .reset_index(drop=True)
        )

        if differential_prob_cutoff is not None:
            windows["extra_selection"] = windows.apply(
                lambda x: (plotdata_diff.iloc[:, plotdata_diff.columns.get_loc(x["start"]) : plotdata_diff.columns.get_loc(x["end"])] > np.log(differential_prob_cutoff)).any().any(), axis=1
            )
            if keep_tss:
                windows.loc[
                    (windows["start"] < 0) & (windows["end"] > 0), "extra_selection"
                ] = True
            windows = windows.loc[windows["extra_selection"]]

        # filter on length
        windows["length"] = windows["end"] - windows["start"]
        windows = windows[windows["length"] > min_length]
        return windows

    def get_interpolated(self, region_id, clusters=None, desired_x=None, step=1):
        probs = self.probs[region_id]

        x_raw = probs.coords["coord"].values
        y_raw = probs.values

        if desired_x is None:
            assert step is not None
            desired_x = np.arange(*self.regions.window, step=step) - self.regions.window[0]

        y = chd.utils.interpolate_1d(
            torch.from_numpy(desired_x), torch.from_numpy(x_raw), torch.from_numpy(y_raw)
        ).numpy()

        return y


def extract_slices(x, cutoff=0.0):
    """
    Given a matrix, extract the indices and values of the contiguous slices where at least one column is above a certain cutoff
    """
    selected = (x > cutoff).any(0).astype(int)
    selected_padded = np.pad(selected, ((1, 1)))
    (start_position_indices,) = np.where(np.diff(selected_padded, axis=-1) == 1)
    (end_position_indices,) = np.where(np.diff(selected_padded, axis=-1) == -1)
    start_position_indices = start_position_indices + 1
    end_position_indices = end_position_indices + 1 - 1

    data = []
    for start_ix, end_ix in zip(start_position_indices, end_position_indices):
        data.append(x[:, start_ix:end_ix].transpose(1, 0))
    if len(data) == 0:
        data = np.zeros((0, x.shape[0]))
    else:
        data = np.concatenate(data, axis=0)

    return start_position_indices, end_position_indices, data

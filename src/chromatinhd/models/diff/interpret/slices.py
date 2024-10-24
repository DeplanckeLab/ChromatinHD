import pandas as pd
import numpy as np
import torch
import xarray as xr
import tqdm


def filter_slices_probs(prob_cutoff=0.0):
    prob_cutoff = 0.0
    # prob_cutoff = -1.
    # prob_cutoff = -4.

    start_position_ixs = []
    end_position_ixs = []
    data = []
    region_ixs = []
    for region, probs in tqdm.tqdm(regionpositional.probs.items()):
        region_ix = fragments.var.index.get_loc(region)
        desired_x = np.arange(*fragments.regions.window) - fragments.regions.window[0]
        x = probs.coords["coord"].values - fragments.regions.window[0]
        y = probs.values

        y_interpolated = chd.utils.interpolate_1d(
            torch.from_numpy(desired_x), torch.from_numpy(x), torch.from_numpy(y)
        ).numpy()

        # from y_interpolated, determine start and end positions of the relevant slices
        start_position_ixs_region, end_position_ixs_region, data_region = extract_slices(y_interpolated, prob_cutoff)
        start_position_ixs.append(start_position_ixs_region + fragments.regions.window[0])
        end_position_ixs.append(end_position_ixs_region + fragments.regions.window[0])
        data.append(data_region)
        region_ixs.append(np.ones(len(start_position_ixs_region), dtype=int) * region_ix)
    data = np.concatenate(data, axis=0)
    start_position_ixs = np.concatenate(start_position_ixs, axis=0)
    end_position_ixs = np.concatenate(end_position_ixs, axis=0)
    region_ixs = np.concatenate(region_ixs, axis=0)

    slices = Slices(region_ixs, start_position_ixs, end_position_ixs, data, fragments.n_regions)


def extract_slices(x, cutoff=0.0):
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

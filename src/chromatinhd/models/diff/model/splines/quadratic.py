import math

import torch
from torch.nn import functional as F

DEFAULT_MIN_BIN_WIDTH = 1e-5
DEFAULT_MIN_BIN_HEIGHT = 1e-5


def unconstrained_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    inverse=False,
    tails="linear",
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
):
    inside_interval_mask = (inputs <= 1) & (inputs >= 0)

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths,
        # unnormalized_widths=unnormalized_widths[inside_interval_mask, ...],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, ...],
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
    )

    return outputs, logabsdet


def calculate_widths(unnormalized_widths, min_bin_width=DEFAULT_MIN_BIN_WIDTH):
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = (
        min_bin_width + (1 - min_bin_width * unnormalized_widths.shape[-1]) * widths
    )
    return widths


def calculate_heights(
    unnormalized_heights, widths, min_bin_height=DEFAULT_MIN_BIN_HEIGHT, local=True
):
    unnorm_heights_exp = torch.exp(unnormalized_heights)

    min_bin_height = 1e-3

    if local:
        # per feature normalization
        unnormalized_area = torch.sum(
            ((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2) * widths,
            dim=-1,
            keepdim=True,
        )
        heights = unnorm_heights_exp / unnormalized_area
        heights = min_bin_height + (1 - min_bin_height) * heights
    else:
        # global normalization
        unnormalized_area = torch.sum(
            ((unnorm_heights_exp[..., :-1] + unnorm_heights_exp[..., 1:]) / 2) * widths,
        )
        heights = unnorm_heights_exp * unnorm_heights_exp.shape[-2] / unnormalized_area
        heights = min_bin_height + (1 - min_bin_height) * heights

    # to check
    # normalized_area = torch.sum(
    #     ((heights[..., :-1] + heights[..., 1:]) / 2) * widths,
    #     dim=-1,
    #     keepdim=True,
    # )
    # print(normalized_area.sum())

    return heights


def calculate_bin_left_cdf(heights, widths):
    bin_left_cdf = torch.cumsum(
        ((heights[..., :-1] + heights[..., 1:]) / 2) * widths, dim=-1
    )
    bin_left_cdf[..., -1] = 1.0
    bin_left_cdf = F.pad(bin_left_cdf, pad=(1, 0), mode="constant", value=0.0)
    return bin_left_cdf


def calculate_bin_locations(widths):
    bin_locations = torch.cumsum(widths, dim=-1)
    bin_locations[..., -1] = 1.0
    bin_locations = F.pad(bin_locations, pad=(1, 0), mode="constant", value=0.0)
    return bin_locations


def quadratic_spline(
    inputs,
    unnormalized_widths=None,
    unnormalized_heights=None,
    widths=None,
    heights=None,
    bin_left_cdf=None,
    bin_locations=None,
    inverse=False,
):
    # calculate widths
    if widths is None:
        widths = calculate_widths(unnormalized_widths)

    num_bins = widths.shape[-1]
    if widths.ndim == inputs.ndim:
        widths = widths.expand(inputs.shape[0], -1)

    # calculate heights
    if heights is None:
        heights = calculate_heights(unnormalized_heights, widths)

    if heights.ndim == inputs.ndim:
        heights = heights.expand(inputs.shape[0], -1)

    # calculate bin indices
    if bin_left_cdf is None:
        bin_left_cdf = calculate_bin_left_cdf(heights, widths)

    if bin_locations is None:
        bin_locations = calculate_bin_locations(widths)

    if inverse:
        bin_idx = torch.searchsorted(bin_left_cdf, inputs.unsqueeze(-1)).squeeze(-1) - 1
    else:
        bin_idx = (
            torch.searchsorted(bin_locations, inputs.unsqueeze(-1)).squeeze(-1) - 1
        )

    bin_idx = torch.clamp(bin_idx, 0, num_bins - 1)

    if bin_idx.ndim < inputs.ndim:
        bin_idx = bin_idx.unsqueeze(-1)

    # get bin locations/widths for input values
    input_bin_locations = bin_locations.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
    input_bin_widths = widths.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)

    if bin_left_cdf.ndim > bin_idx.ndim:
        input_left_cdf = bin_left_cdf.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_left_heights = heights.gather(-1, bin_idx.unsqueeze(-1)).squeeze(-1)
        input_right_heights = heights.gather(-1, bin_idx.unsqueeze(-1) + 1).squeeze(-1)
    else:
        input_left_cdf = bin_left_cdf.gather(-1, bin_idx)
        input_left_heights = heights.gather(-1, bin_idx)
        input_right_heights = heights.gather(-1, bin_idx + 1)

    # calculate log abs det and output
    a = 0.5 * (input_right_heights - input_left_heights) * input_bin_widths
    b = input_left_heights * input_bin_widths
    c = input_left_cdf

    if inverse:
        c_ = c - inputs
        alpha = torch.clamp((-b + torch.sqrt(b.pow(2) - 4 * a * c_)) / (2 * a), 0, 1)
        # avoids alpha = -inf due to perfectly horizontal curve (i.e. a = 0)
        alpha[alpha.isnan()] = 0.0

        outputs = alpha * input_bin_widths + input_bin_locations
        logabsdet = -torch.log(
            (alpha * (input_right_heights - input_left_heights) + input_left_heights)
        )
    else:
        # due to numerical imprecision, alpha can sometimes fall outside of 0 and 1
        alpha = torch.clamp((inputs - input_bin_locations) / input_bin_widths, 0, 1)

        outputs = a * alpha.pow(2) + b * alpha + c

        logabsdet = torch.log(
            (alpha * (input_right_heights - input_left_heights) + input_left_heights)
        )

    outputs = torch.clamp(outputs, 0, 1)

    return outputs, logabsdet

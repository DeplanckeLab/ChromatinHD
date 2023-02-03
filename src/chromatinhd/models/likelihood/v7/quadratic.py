import math

import torch
from torch.nn import functional as F

DEFAULT_MIN_BIN_WIDTH = 1e-4
DEFAULT_MIN_BIN_HEIGHT = 1e-5


def calculate_logarea(heights, widths, dim=-1):
    return torch.logsumexp(
        (
            torch.logaddexp(
                torch.log(heights[..., :-1]),
                torch.log(heights[..., 1:]),
            )
            - math.log(2)
        )
        + torch.log(widths),
        dim=dim,
        keepdim=True,
    )


def calculate_area(heights, widths, dim=-1):
    return torch.exp(calculate_logarea(heights, widths, dim=dim))


def calculate_widths(
    unnormalized_widths, min_bin_width=DEFAULT_MIN_BIN_WIDTH, set_min_bin_width=True
):
    widths = F.softmax(unnormalized_widths, dim=-1)

    if set_min_bin_width:
        widths = (
            min_bin_width + (1 - min_bin_width * unnormalized_widths.shape[-1]) * widths
        )
    return widths


def calculate_heights(
    unnormalized_heights, widths, min_bin_height=DEFAULT_MIN_BIN_HEIGHT
):
    # unnorm_heights_exp = torch.nn.functional.softplus(unnormalized_heights)
    unnorm_heights_exp = torch.exp(unnormalized_heights)

    unnormalized_area = calculate_area(
        unnorm_heights_exp, widths, dim=tuple(range(1, unnorm_heights_exp.ndim))
    )
    heights = unnorm_heights_exp / unnormalized_area

    # min_bin_height = 1e-5
    # heights = min_bin_height + (1 - min_bin_height) * heights

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
    widths=None,
    heights=None,
    bin_left_cdf=None,
    bin_locations=None,
    inverse=False,
):
    num_bins = widths.shape[-1]
    if widths.ndim == inputs.ndim:
        widths = widths.expand(inputs.shape[0], -1)

    if heights.ndim == inputs.ndim:
        heights = heights.expand(inputs.shape[0], -1)

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

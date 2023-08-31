import torch
import math


def transform_linear_spline(positions, n, width, unnormalized_heights):
    binsize = torch.div(width, n, rounding_mode="floor")

    normalized_heights = torch.nn.functional.log_softmax(unnormalized_heights, -1)
    if normalized_heights.ndim == positions.ndim:
        normalized_heights = normalized_heights.unsqueeze(0)

    binixs = torch.div(positions, binsize, rounding_mode="trunc")

    logprob = torch.gather(normalized_heights, 1, binixs.unsqueeze(1)).squeeze(1)

    positions = positions - binixs * binsize
    width = binsize

    return logprob, positions, width


def calculate_logprob(positions, nbins, width, unnormalized_heights_zooms):
    """
    Calculate the zoomed log probability per position given a set of unnormalized_heights_zooms
    """
    assert len(nbins) == len(unnormalized_heights_zooms)

    curpositions = positions
    curwidth = width
    logprob = torch.zeros_like(positions, dtype=torch.float)
    for i, (n, unnormalized_heights_zoom) in enumerate(zip(nbins, unnormalized_heights_zooms)):
        assert (curwidth % n) == 0
        logprob_layer, curpositions, curwidth = transform_linear_spline(
            curpositions,
            n,
            curwidth,
            unnormalized_heights_zoom,
        )
        logprob += logprob_layer
    logprob = logprob - math.log(
        curwidth
    )  # if any width is left, we need to divide by the remaining number of possibilities to get a properly normalized probability
    return logprob


def extract_unnormalized_heights(positions, totalbinwidths, unnormalized_heights_all):
    """
    Extracts the unnormalized heights per zoom level from the global unnormalized heights tensor with size (totaln, n)
    You typically do not want to use this function directly, as the benifits of a zoomed likelihood are lost in this way.
    This function is mainly useful for debugging, inference or testing purposes
    """
    totalbinixs = torch.div(positions[:, None], totalbinwidths, rounding_mode="floor")
    totalbinsectors = torch.nn.functional.pad(totalbinixs[..., :-1], (1, 0))
    # totalbinsectors = torch.div(totalbinixs, self.nbins[None, :], rounding_mode="floor")
    unnormalized_heights_zooms = [
        torch.index_select(
            unnormalized_heights_all[i], 0, totalbinsector
        )  # index_select is much faster than direct indexin
        for i, totalbinsector in enumerate(totalbinsectors.T)
    ]

    return unnormalized_heights_zooms

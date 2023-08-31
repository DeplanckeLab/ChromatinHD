import torch
import numpy as np
import math
from typing import Union
from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.data.fragments import Fragments, FragmentsView
from chromatinhd.models.diff.model.splines import quadratic


class FragmentPositionDistribution(torch.nn.Module):
    pass


def calculate_binixs(coordinates, window, binsize):
    return torch.div(coordinates - window[0], binsize, rounding_mode="floor")


class Baseline(FragmentPositionDistribution):
    def __init__(self, fragments, clustering, binsize=200, baseline=None):
        super().__init__()

        baseline_pretrained = baseline if baseline is not None else None

        self.binsize = binsize

        if baseline_pretrained is not None:
            baseline = baseline_pretrained
        else:
            baseline = torch.nn.Embedding(
                fragments.n_regions,
                (fragments.regions.width // binsize),
                sparse=True,
            )

            # initialize by counting
            # binwidth = fragments.regions.region_width / fragmentprob_size
            # coordinates = fragments.coordinates[:, 0].numpy() - fragments.regions.window[0]
            # selected = (coordinates >= 0) & (coordinates < fragments.regions.region_width)
            # coordinates = coordinates[selected]
            # binixs = (coordinates // binwidth).astype(int)
            # region_ix = fragments.mapping[selected, 1]
            # count = torch.bincount(
            #     (region_ix * fragmentprob_size + binixs),
            #     minlength=fragmentprob_size * (fragments.n_regions),
            # ).reshape((fragments.n_regions), fragmentprob_size)
            # init_baseline = torch.log(count.float() + 1e-5)

            # baseline.weight.data[:] = init_baseline

        self.baseline = baseline

    def log_prob(self, data):
        unnormalized_height = self.baseline(data.minibatch.regions_oi_torch)

        logprob = torch.zeros((data.fragments.n_fragments, 2), device=data.fragments.coordinates.device)

        heights = torch.nn.functional.log_softmax(unnormalized_height, -1)

        bin_ixs_left = calculate_binixs(data.fragments.coordinates[:, 0], data.fragments.window, self.binsize)
        logprob[:, 0] = heights[data.fragments.local_region_ix, bin_ixs_left]
        bin_ixs_right = calculate_binixs(data.fragments.coordinates[:, 1], data.fragments.window, self.binsize)
        logprob[:, 1] = heights[data.fragments.local_region_ix, bin_ixs_right]
        return logprob

    def parameters_sparse(self):
        yield self.baseline.weight


class FragmentPositionDistribution1(FragmentPositionDistribution):
    def __init__(self, fragments: Union[Fragments, FragmentsView], clustering, binsize=200, baseline=None):
        super().__init__()

        baseline_pretrained = baseline if baseline is not None else None

        self.binsize = binsize

        if baseline_pretrained is not None:
            baseline = baseline_pretrained
        else:
            baseline = torch.nn.Embedding(
                fragments.n_regions,
                (fragments.regions.width // binsize),
                sparse=True,
            )
        self.baseline = baseline

        self.binsize = binsize
        self.binwidth = fragments.regions.width // binsize

        self.delta_logit = EmbeddingTensor(fragments.n_regions, (clustering.n_clusters, self.binwidth), sparse=True)
        self.delta_logit.weight.data[:] = 0

    def log_prob(self, data):
        unnormalized_height = self.baseline(data.minibatch.regions_oi_torch).unsqueeze(1) + self.delta_logit(
            data.minibatch.regions_oi_torch
        )

        logprob = torch.zeros((data.fragments.n_fragments, 2), device=data.fragments.coordinates.device)

        heights = torch.nn.functional.log_softmax(unnormalized_height, -1) - math.log(self.binsize)

        bin_ixs_left = calculate_binixs(data.fragments.coordinates[:, 0], data.fragments.window, self.binsize)
        logprob[:, 0] = heights[
            data.fragments.local_region_ix, data.clustering.labels[data.fragments.local_cell_ix], bin_ixs_left
        ]
        bin_ixs_right = calculate_binixs(data.fragments.coordinates[:, 1], data.fragments.window, self.binsize)
        logprob[:, 1] = heights[
            data.fragments.local_region_ix, data.clustering.labels[data.fragments.local_cell_ix], bin_ixs_right
        ]
        return logprob

    def parameters_sparse(self):
        yield self.baseline.weight
        yield self.delta_logit.weight


class FragmentPositionDistribution2(FragmentPositionDistribution):
    def __init__(
        self, fragments: Union[Fragments, FragmentsView], clustering, binsize=200, baseline=None, delta_logit=None
    ):
        super().__init__()

        baseline_pretrained = baseline if baseline is not None else None

        self.binsize = binsize

        if baseline_pretrained is not None:
            baseline = baseline_pretrained
        else:
            baseline = torch.nn.Embedding(
                fragments.n_regions,
                (fragments.regions.width // binsize),
                sparse=True,
            )
        self.baseline = baseline

        self.binsize = binsize
        self.binwidth = fragments.regions.width // binsize

        if delta_logit is None:
            self.delta_logit = EmbeddingTensor(fragments.n_regions, (clustering.n_clusters, self.binwidth), sparse=True)
            self.delta_logit.weight.data[:] = 0
        else:
            self.delta_logit = delta_logit

        self.register_buffer("bin_positions", torch.arange(*fragments.regions.window, binsize) + binsize // 2)

        self.width = fragments.regions.width
        self.inside = torch.nn.Parameter(torch.ones(1) * 10)

    def log_prob(self, data):
        unnormalized_heights = self.baseline(data.minibatch.regions_oi_torch).unsqueeze(1) + self.delta_logit(
            data.minibatch.regions_oi_torch
        )

        logprob = torch.zeros((data.fragments.n_fragments, 2), device=data.fragments.coordinates.device)

        heights_left = torch.nn.functional.log_softmax(unnormalized_heights, -1) - math.log(self.binsize)
        bin_ixs_left = calculate_binixs(data.fragments.coordinates[:, 0], data.fragments.window, self.binsize)
        logprob[:, 0] = heights_left[
            data.fragments.local_region_ix, data.clustering.labels[data.fragments.local_cell_ix], bin_ixs_left
        ]

        # calculate weight
        bin_ixs_right = calculate_binixs(data.fragments.coordinates[:, 1], data.fragments.window, self.binsize)
        bin_ixs_match = bin_ixs_left == bin_ixs_right

        logprob_inside = torch.log(torch.sigmoid(self.inside)) - math.log(self.binwidth)
        logprob_outside = torch.log(1 - torch.sigmoid(self.inside)) - math.log(self.width - self.binwidth)

        logprob[:, 1] = (logprob_inside * bin_ixs_match.float()) + (logprob_outside * (1 - bin_ixs_match.float()))

        return logprob

    def parameters_sparse(self):
        yield self.baseline.weight
        yield self.delta_logit.weight


class FragmentsizeDistribution(torch.nn.Module):
    def __init__(self, fragments):
        super().__init__()

        # create spline
        self.width = 1024
        self.bin_width = 1024 // 4

        # make sure self.width is divisible by bin_width
        assert self.width % self.bin_width == 0
        n_bins = self.width // self.bin_width

        unnormalized_widths = torch.ones((n_bins - 1,))
        self.register_buffer("widths", quadratic.calculate_widths(unnormalized_widths))
        self.register_buffer("bin_locations", quadratic.calculate_bin_locations(self.widths))

        # initialize heights
        counts = torch.bincount(
            torch.searchsorted(
                self.bin_locations,
                torch.from_numpy(fragments.coordinates[:1000000, 1] - fragments.coordinates[:1000000, 0]) / self.width,
                right=True,
            )
        )

        outside_count = counts[-1] / counts.sum()
        counts = counts[:-1] / counts.sum()

        self.unnormalized_heights = torch.nn.Parameter(torch.log(counts.float() + 1e-10))
        self.logprob_inside = torch.nn.Parameter(torch.log(outside_count))

    def log_prob(self, data):
        fragment_size = data.fragments.coordinates[:, 1] - data.fragments.coordinates[:, 0]
        _, logabsdet = quadratic.quadratic_spline(
            fragment_size / self.width,
            widths=self.widths,
            unnormalized_heights=self.unnormalized_heights,
            bin_locations=self.bin_locations,
        )

        logabsdet[fragment_size > self.width] = torch.log(1 - torch.exp(self.logprob_inside))
        logabsdet = logabsdet + self.logprob_inside

        return logabsdet


class FragmentPositionDistribution3(FragmentPositionDistribution):
    def __init__(
        self,
        fragments: Union[Fragments, FragmentsView],
        clustering,
        binsize=200,
        baseline=None,
        delta_logit=None,
        fragmentsize_distribution=None,
    ):
        super().__init__()

        baseline_pretrained = baseline if baseline is not None else None

        self.binsize = binsize

        if baseline_pretrained is not None:
            baseline = baseline_pretrained
        else:
            baseline = torch.nn.Embedding(
                fragments.n_regions,
                (fragments.regions.width // binsize),
                sparse=True,
            )
        self.baseline = baseline

        self.binsize = binsize
        self.binwidth = fragments.regions.width // binsize

        if delta_logit is None:
            self.delta_logit = EmbeddingTensor(fragments.n_regions, (clustering.n_clusters, self.binwidth), sparse=True)
            self.delta_logit.weight.data[:] = 0
        else:
            self.delta_logit = delta_logit

        self.register_buffer("bin_positions", torch.arange(*fragments.regions.window, binsize) + binsize // 2)

        if fragmentsize_distribution is None:
            fragmentsize_distribution = FragmentsizeDistribution(fragments)

        self.fragmentsize_distribution = fragmentsize_distribution

    def log_prob(self, data):
        unnormalized_heights = self.baseline(data.minibatch.regions_oi_torch).unsqueeze(1) + self.delta_logit(
            data.minibatch.regions_oi_torch
        )

        logprob = torch.zeros((data.fragments.n_fragments, 2), device=data.fragments.coordinates.device)

        heights_left = torch.nn.functional.log_softmax(unnormalized_heights, -1)
        bin_ixs_left = calculate_binixs(data.fragments.coordinates[:, 0], data.fragments.window, self.binsize)
        logprob[:, 0] = heights_left[
            data.fragments.local_region_ix, data.clustering.labels[data.fragments.local_cell_ix], bin_ixs_left
        ]

        logprob[:, 1] = self.fragmentsize_distribution.log_prob(data)
        return logprob

    def parameters_sparse(self):
        yield self.baseline.weight
        yield self.delta_logit.weight


##

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


def calculate_logprob(positions, nbins, width, unnormalized_heights_bins):
    assert len(nbins) == len(unnormalized_heights_bins)

    curpositions = positions
    curwidth = width
    logprob = torch.zeros_like(positions, dtype=torch.float)
    for i, n in enumerate(nbins):
        assert (curwidth % n) == 0
        logprob_layer, curpositions, curwidth = transform_linear_spline(
            curpositions, n, curwidth, unnormalized_heights_bins[i]
        )
        logprob += logprob_layer
    logprob = logprob - math.log(
        curwidth
    )  # if any width is left, we need to divide by the remaining number of possibilities to get a properly normalized probability
    return logprob


class FragmentsizeDistribution2(torch.nn.Module):
    def __init__(self, fragments, nbins=(8, 8, 8, 2), width=1024):
        super().__init__()

        self.register_buffer("nbins", torch.from_numpy(np.array(nbins)))
        self.total_width = fragments.regions.width
        self.width = width

        self.register_buffer("totalnbins", torch.cumprod(self.nbins, 0))
        self.register_buffer("totalbinwidths", torch.div(torch.tensor(width), self.totalnbins, rounding_mode="floor"))

        unnormalized_heights_all = []
        for i, (n, totaln) in enumerate(zip(self.nbins.numpy(), self.totalnbins.numpy())):
            if (width // totaln) % 1 > 0:
                raise ValueError("cumulative number of bins should be a multiple of width")
            setattr(self, f"unnormalized_heights_all_{i}", torch.nn.Parameter(torch.zeros(totaln).reshape(-1, n)))
        self.unnormalized_heights_all = unnormalized_heights_all

        self.logprob_inside = torch.nn.Parameter(torch.logit(torch.tensor(0.9)))

    def log_prob(self, data):
        fragmentsizes = torch.abs(data.fragments.coordinates[:, 1] - data.fragments.coordinates[:, 0])
        inside = fragmentsizes < self.width

        prob_inside = torch.sigmoid(self.logprob_inside)
        logprob_outside = torch.log(1 - prob_inside) - (math.log(self.total_width - self.width))
        logprob_inside = torch.log(prob_inside)
        log_prob = torch.where(inside, logprob_inside, logprob_outside)
        fragmentsizes_inside = fragmentsizes[inside]

        totalbinixs = torch.div(fragmentsizes_inside[:, None], self.totalbinwidths, rounding_mode="floor")
        totalbinsectors = torch.pad(totalbinix[..., 1:], (1, 0))
        # totalbinsectors = torch.div(totalbinixs, self.nbins[None, :], rounding_mode="floor")
        unnormalized_heights_bins = [
            torch.index_select(getattr(self, f"unnormalized_heights_all_{i}"), 0, totalbinsector)
            for i, totalbinsector in enumerate(totalbinsectors.T)
        ]

        log_prob[inside] += calculate_logprob(fragmentsizes_inside, self.nbins, self.width, unnormalized_heights_bins)

        return log_prob

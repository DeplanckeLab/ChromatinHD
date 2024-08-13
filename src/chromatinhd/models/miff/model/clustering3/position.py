import torch
import numpy as np
import math
from typing import Union
import itertools
from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.data.fragments import Fragments, FragmentsView
from chromatinhd.models.diff.model.splines import quadratic
import copy


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

        heights = torch.nn.functional.log_softmax(unnormalized_height, -1) - math.log(self.binsize)

        bin_ixs_left = calculate_binixs(data.fragments.coordinates[:, 0], data.fragments.window, self.binsize)
        logprob[:, 0] = heights[data.fragments.local_region_ix, bin_ixs_left]
        # bin_ixs_right = calculate_binixs(data.fragments.coordinates[:, 1], data.fragments.window, self.binsize)
        # logprob[:, 1] = heights[data.fragments.local_region_ix, bin_ixs_right]
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
        # bin_ixs_right = calculate_binixs(data.fragments.coordinates[:, 1], data.fragments.window, self.binsize)
        # logprob[:, 1] = heights[
        #     data.fragments.local_region_ix, data.clustering.labels[data.fragments.local_cell_ix], bin_ixs_right
        # ]
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

    @classmethod
    def from_baseline(cls, fragments, clustering, reference, **kwargs):
        return cls(
            fragments=fragments,
            clustering=clustering,
            baseline=torch.nn.Embedding.from_pretrained(reference.baseline.weight.data),
            delta_logit=EmbeddingTensor.from_pretrained(reference.delta_logit),
            **kwargs,
        )


##

from chromatinhd.models.miff.model.zoom import calculate_logprob, extract_unnormalized_heights


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
            unnormalized_heights_zoom = torch.nn.Parameter(torch.zeros(totaln).reshape(-1, n))
            unnormalized_heights_all.append(unnormalized_heights_zoom)
            setattr(self, f"unnormalized_heights_all_{i}", unnormalized_heights_zoom)
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

        unnormalized_heights_zooms = extract_unnormalized_heights(
            fragmentsizes_inside, self.totalbinwidths, self.unnormalized_heights_all
        )

        log_prob[inside] += calculate_logprob(fragmentsizes_inside, self.nbins, self.width, unnormalized_heights_zooms)

        return log_prob


class SineEncoder(torch.nn.Module):
    def __init__(self, n_frequencies):
        super().__init__()

        self.register_buffer(
            "frequencies",
            torch.tensor([[1 / 1000 ** (2 * i / n_frequencies)] * 2 for i in range(1, n_frequencies + 1)]).flatten(-2),
        )
        self.register_buffer(
            "shifts",
            torch.tensor([[0, torch.pi / 2] for _ in range(1, n_frequencies + 1)]).flatten(-2),
        )

        self.n_embedding_dimensions = n_frequencies * 2

    def forward(self, coordinates):
        embedding = torch.sin((coordinates[..., None] * self.frequencies + self.shifts).flatten(-2))
        return embedding


class FragmentsizeDistribution3(torch.nn.Module):
    predict_layers = 2

    def __init__(
        self,
        fragments,
        unnormalized_heights_all,
        nbins=(8, 8, 8, 2),
        width=1024,
        predict_layers=2,
        n_hidden_dimensions=10,
    ):
        super().__init__()

        self.register_buffer("nbins", torch.from_numpy(np.array(nbins)))
        self.total_width = fragments.regions.width
        self.width = width

        self.register_buffer("totalnbins", torch.cumprod(self.nbins, 0))
        self.register_buffer("totalbinwidths", torch.div(torch.tensor(width), self.totalnbins, rounding_mode="floor"))

        # set the baseline distribution
        self.unnormalized_heights_all_baseline = unnormalized_heights_all
        for i, unnormalized_height in enumerate(unnormalized_heights_all):
            self.register_buffer(f"unnormalized_height_all_baseline_{i}", unnormalized_height)

        self.logprob_inside = torch.nn.Parameter(torch.logit(torch.tensor(0.9)))

        self.position_embedder = SineEncoder(5)
        self.n_hidden_dimensions = n_hidden_dimensions
        self.predictor_0 = torch.nn.Sequential(
            torch.nn.Linear(self.position_embedder.n_embedding_dimensions, self.n_hidden_dimensions),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.n_hidden_dimensions, nbins[0], bias=False),
        )

        self.predict_layers = predict_layers
        self.predictor_1 = torch.nn.Sequential(
            torch.nn.Linear(self.position_embedder.n_embedding_dimensions * 2, self.n_hidden_dimensions),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.n_hidden_dimensions, nbins[1], bias=False),
        )

    def log_prob(self, data):
        fragmentsizes = torch.abs(data.fragments.coordinates[:, 1] - data.fragments.coordinates[:, 0])
        inside = fragmentsizes < self.width

        prob_inside = torch.sigmoid(self.logprob_inside)
        logprob_outside = torch.log(1 - prob_inside) - (math.log(self.total_width - self.width))
        logprob_inside = torch.log(prob_inside)
        log_prob = torch.where(inside, logprob_inside, logprob_outside)
        fragmentsizes_inside = fragmentsizes[inside]

        unnormalized_heights_zooms = extract_unnormalized_heights(
            fragmentsizes_inside,
            self.totalbinwidths,
            [getattr(self, f"unnormalized_height_all_baseline_{i}") for i in range(2)],
        )

        # input : cat(left cut site position embedding + current zoom fragment size embedding + current probability (and/or any extra state variables coming from previous neural network))
        # output: unnormalized heights diff at this zoom for each fragment
        unnormalized_heights_zooms_diff_0 = self.predictor_0(
            self.position_embedder(data.fragments.coordinates[inside, 0].unsqueeze(-1))
        )
        unnormalized_heights_zooms[0] = unnormalized_heights_zooms[0] + unnormalized_heights_zooms_diff_0

        if self.predict_layers > 1:
            bincenters_0 = (
                torch.div(fragmentsizes_inside, self.totalbinwidths[0], rounding_mode="floor") * self.totalbinwidths[0]
            )
            position_embedding = torch.cat(
                [
                    self.position_embedder(data.fragments.coordinates[inside, 0].unsqueeze(-1)),
                    self.position_embedder(bincenters_0.unsqueeze(-1)),
                ],
                dim=-1,
            )
            unnormalized_heights_zooms_diff_1 = self.predictor_1(position_embedding)
            unnormalized_heights_zooms[1] = unnormalized_heights_zooms[1] + unnormalized_heights_zooms_diff_1

        log_prob[inside] += calculate_logprob(fragmentsizes_inside, self.nbins, self.width, unnormalized_heights_zooms)

        return log_prob


class LinearMultihead(torch.nn.Module):
    bias = None

    def __init__(self, in_features: int, out_features: int, n_heads: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        if bias:
            self.bias = EmbeddingTensor(
                n_heads,
                (out_features,),
                sparse=True,
            )
            self.bias.data.zero_()

        self.weight = EmbeddingTensor(
            n_heads,
            (
                in_features,
                out_features,
            ),
            sparse=True,
        )
        stdv = 1.0 / math.sqrt(self.weight.shape[-1])
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.Tensor, head_ix: torch.Tensor):
        if self.bias is not None:
            return torch.einsum("ab,abc->ac", input, self.weight(head_ix)) + self.bias(head_ix)
        return torch.einsum("ab,abc->ac", input, self.weight(head_ix))

    def parameters_sparse(self):
        yield self.weight.weight
        if self.bias is not None:
            yield self.bias.weight


class FragmentsizeDistribution4(torch.nn.Module):
    def __init__(
        self,
        fragments,
        unnormalized_heights_all,
        nbins=(8, 8, 8, 2),
        width=1024,
        predict_layers=2,
        n_hidden_dimensions=10,
    ):
        super().__init__()

        self.register_buffer("nbins", torch.from_numpy(np.array(nbins)))
        self.total_width = fragments.regions.width
        self.width = width

        self.register_buffer("totalnbins", torch.cumprod(self.nbins, 0))
        self.register_buffer("totalbinwidths", torch.div(torch.tensor(width), self.totalnbins, rounding_mode="floor"))

        # set the baseline distribution
        self.unnormalized_heights_all_baseline = unnormalized_heights_all
        for i, unnormalized_height in enumerate(unnormalized_heights_all):
            self.register_buffer(f"unnormalized_height_all_baseline_{i}", unnormalized_height)

        self.logprob_inside = torch.nn.Parameter(torch.logit(torch.tensor(0.9)))

        self.position_embedder = SineEncoder(5)
        self.n_hidden_dimensions = n_hidden_dimensions
        self.predictor_0 = torch.nn.Sequential(
            torch.nn.Linear(self.position_embedder.n_embedding_dimensions, self.n_hidden_dimensions),
            torch.nn.Sigmoid(),
            # torch.nn.Linear(self.n_hidden_dimensions, nbins[0], bias=False),
        )
        self.predictor_0_final = LinearMultihead(self.n_hidden_dimensions, nbins[0], fragments.n_regions, bias=False)

        self.predict_layers = predict_layers
        self.predictor_1 = torch.nn.Sequential(
            torch.nn.Linear(self.position_embedder.n_embedding_dimensions * 2, self.n_hidden_dimensions),
            torch.nn.Sigmoid(),
            # torch.nn.Linear(self.n_hidden_dimensions, nbins[1], bias=False),
        )
        self.predictor_1_final = LinearMultihead(self.n_hidden_dimensions, nbins[1], fragments.n_regions, bias=False)

    def log_prob(self, data):
        fragmentsizes = torch.abs(data.fragments.coordinates[:, 1] - data.fragments.coordinates[:, 0])
        inside = fragmentsizes < self.width

        prob_inside = torch.sigmoid(self.logprob_inside)
        logprob_outside = torch.log(1 - prob_inside) - (math.log(self.total_width - self.width))
        logprob_inside = torch.log(prob_inside)
        log_prob = torch.where(inside, logprob_inside, logprob_outside)
        fragmentsizes_inside = fragmentsizes[inside]

        unnormalized_heights_zooms = extract_unnormalized_heights(
            fragmentsizes_inside,
            self.totalbinwidths,
            [getattr(self, f"unnormalized_height_all_baseline_{i}") for i in range(2)],
        )

        # input : cat(left cut site position embedding + current zoom fragment size embedding + current probability (and/or any extra state variables coming from previous neural network))
        # output: unnormalized heights diff at this zoom for each fragment
        unnormalized_heights_zooms_diff_0 = self.predictor_0_final(
            self.predictor_0(self.position_embedder(data.fragments.coordinates[inside, 0].unsqueeze(-1))),
            data.fragments.regionmapping[inside],
        )
        unnormalized_heights_zooms[0] = unnormalized_heights_zooms[0] + unnormalized_heights_zooms_diff_0

        if self.predict_layers > 1:
            bincenters_0 = (
                torch.div(fragmentsizes_inside, self.totalbinwidths[0], rounding_mode="floor") * self.totalbinwidths[0]
            )
            position_embedding = torch.cat(
                [
                    self.position_embedder(data.fragments.coordinates[inside, 0].unsqueeze(-1)),
                    self.position_embedder(bincenters_0.unsqueeze(-1)),
                ],
                dim=-1,
            )
            unnormalized_heights_zooms_diff_1 = self.predictor_1_final(
                self.predictor_1(position_embedding), data.fragments.regionmapping[inside]
            )
            unnormalized_heights_zooms[1] = unnormalized_heights_zooms[1] + unnormalized_heights_zooms_diff_1

        log_prob[inside] += calculate_logprob(fragmentsizes_inside, self.nbins, self.width, unnormalized_heights_zooms)

        return log_prob

    def parameters_sparse(self):
        return itertools.chain(self.predictor_0_final.parameters_sparse(), self.predictor_1_final.parameters_sparse())

    @classmethod
    def from_baseline(cls, fragments, baseline, **kwargs):
        return cls(
            fragments, unnormalized_heights_all=baseline.unnormalized_heights_all, nbins=baseline.nbins, **kwargs
        )


class LinearMultiheadSplit(torch.nn.Module):
    bias = None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        n_splits: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if bias:
            self.bias = EmbeddingTensor(
                n_heads,
                (out_features,),
                sparse=True,
            )
            self.bias.data.zero_()

        self.weight = EmbeddingTensor(
            n_heads,
            (
                in_features,
                out_features,
            ),
            sparse=True,
        )
        stdv = 1.0 / math.sqrt(self.weight.shape[-1])
        self.weight.data.uniform_(-stdv, stdv)
        self.delta_weight = EmbeddingTensor(
            n_heads * n_splits,
            (
                in_features,
                out_features,
            ),
            sparse=True,
        )
        stdv = 1.0 / math.sqrt(self.delta_weight.shape[-1])
        self.delta_weight.data.zero_()
        self.n_splits = n_splits

    def forward(self, input: torch.Tensor, head_ix: torch.Tensor, split_ix: torch.Tensor):
        weight = self.weight(head_ix) + self.delta_weight(head_ix * self.n_splits + split_ix) * 0.1
        # weight = (
        #     self.weight(head_ix)
        #     + torch.index_select(self.delta_weight.get_full_weight(), 0, head_ix * self.n_splits + split_ix) * 0.1
        # )
        if self.bias is not None:
            return torch.einsum("ab,abc->ac", input, weight) + self.bias(head_ix)
        return torch.einsum("ab,abc->ac", input, weight)

    def parameters_sparse(self):
        yield self.weight.weight
        yield self.delta_weight.weight
        if self.bias is not None:
            yield self.bias.weight


class FragmentsizeDistribution5(torch.nn.Module):
    def __init__(
        self,
        fragments,
        clustering,
        unnormalized_heights_all_baseline,
        nbins=(8, 8, 8, 2),
        width=1024,
        predict_layers=2,
        n_hidden_dimensions=10,
        logprob_inside=None,
        shuffle_clustering=False,
    ):
        super().__init__()

        self.register_buffer("nbins", torch.from_numpy(np.array(nbins)))
        self.total_width = fragments.regions.width
        self.width = width

        self.register_buffer("totalnbins", torch.cumprod(self.nbins, 0))
        self.register_buffer("totalbinwidths", torch.div(torch.tensor(width), self.totalnbins, rounding_mode="floor"))

        # set the baseline distribution
        self.unnormalized_heights_all_baseline = unnormalized_heights_all_baseline
        for i, unnormalized_height in enumerate(unnormalized_heights_all_baseline):
            self.register_buffer(f"unnormalized_height_all_baseline_{i}", unnormalized_height)
            # setattr(self, f"unnormalized_height_all_baseline_{i}", unnormalized_height)

        if logprob_inside is None:
            self.logprob_inside = torch.nn.Parameter(torch.logit(torch.tensor(0.9)))
        else:
            self.register_buffer("logprob_inside", logprob_inside)

        self.position_embedder = SineEncoder(5)
        self.n_hidden_dimensions = n_hidden_dimensions
        self.predictor_0 = torch.nn.Sequential(
            torch.nn.Linear(self.position_embedder.n_embedding_dimensions, self.n_hidden_dimensions),
            torch.nn.Sigmoid(),
        )
        self.predictor_0_final = LinearMultiheadSplit(
            self.n_hidden_dimensions, nbins[0], fragments.n_regions, clustering.n_clusters, bias=False
        )

        self.predict_layers = predict_layers
        self.predictor_1 = torch.nn.Sequential(
            torch.nn.Linear(self.position_embedder.n_embedding_dimensions * 2, self.n_hidden_dimensions),
            torch.nn.Sigmoid(),
        )
        self.predictor_1_final = LinearMultiheadSplit(
            self.n_hidden_dimensions, nbins[1], fragments.n_regions, clustering.n_clusters, bias=False
        )

        self.n_clusters = clustering.n_clusters
        self.shuffle_clustering = shuffle_clustering

    def log_prob(self, data):
        fragmentsizes = torch.abs(data.fragments.coordinates[:, 1] - data.fragments.coordinates[:, 0])
        inside = fragmentsizes < self.width

        prob_inside = torch.sigmoid(self.logprob_inside)
        logprob_outside = torch.log(1 - prob_inside) - (math.log(self.total_width - self.width))
        logprob_inside = torch.log(prob_inside)
        log_prob = torch.where(inside, logprob_inside, logprob_outside)
        fragmentsizes_inside = fragmentsizes[inside]

        unnormalized_heights_zooms = extract_unnormalized_heights(
            fragmentsizes_inside,
            self.totalbinwidths,
            [getattr(self, f"unnormalized_height_all_baseline_{i}") for i in range(2)],
        )

        if self.shuffle_clustering:
            data.clustering.labels = torch.randint(
                0, self.n_clusters, (data.clustering.labels.shape[0],), device=data.clustering.labels.device
            )

        # input : cat(left cut site position embedding + current zoom fragment size embedding + current probability (and/or any extra state variables coming from previous neural network))
        # output: unnormalized heights diff at this zoom for each fragment
        unnormalized_heights_zooms_diff_0 = self.predictor_0_final(
            self.predictor_0(self.position_embedder(data.fragments.coordinates[inside, 0].unsqueeze(-1))),
            torch.masked_select(data.fragments.regionmapping, inside),
            torch.index_select(data.clustering.labels, 0, torch.masked_select(data.fragments.local_cell_ix, inside)),
        )
        unnormalized_heights_zooms[0] = unnormalized_heights_zooms[0] + unnormalized_heights_zooms_diff_0

        if self.predict_layers > 1:
            bincenters_0 = (
                torch.div(fragmentsizes_inside, self.totalbinwidths[0], rounding_mode="floor") * self.totalbinwidths[0]
            )
            position_embedding = torch.cat(
                [
                    self.position_embedder(data.fragments.coordinates[inside, 0].unsqueeze(-1)),
                    self.position_embedder(bincenters_0.unsqueeze(-1)),
                ],
                dim=-1,
            )
            unnormalized_heights_zooms_diff_1 = self.predictor_1_final(
                self.predictor_1(position_embedding),
                torch.masked_select(data.fragments.regionmapping, inside),
                torch.index_select(
                    data.clustering.labels, 0, torch.masked_select(data.fragments.local_cell_ix, inside)
                ),
            )
            unnormalized_heights_zooms[1] = unnormalized_heights_zooms[1] + unnormalized_heights_zooms_diff_1

        log_prob[inside] += calculate_logprob(fragmentsizes_inside, self.nbins, self.width, unnormalized_heights_zooms)

        return log_prob

    def parameters_sparse(self):
        return itertools.chain(self.predictor_0_final.parameters_sparse(), self.predictor_1_final.parameters_sparse())

    @classmethod
    def from_baseline(cls, fragments, clustering, reference, **kwargs):
        self = cls(
            fragments,
            clustering,
            unnormalized_heights_all_baseline=reference.unnormalized_heights_all_baseline,
            nbins=reference.nbins,
            logprob_inside=reference.logprob_inside.clone().detach(),
            predict_layers=reference.predict_layers,
            n_hidden_dimensions=reference.n_hidden_dimensions,
            **kwargs,
        )
        self.predictor_0 = copy.deepcopy(reference.predictor_0)
        for p in self.predictor_0.parameters():
            p.requires_grad = False
        self.predictor_0_final.weight = EmbeddingTensor.from_pretrained(reference.predictor_0_final.weight)
        self.predictor_1 = copy.deepcopy(reference.predictor_1)
        for p in self.predictor_1.parameters():
            p.requires_grad = False
        self.predictor_1_final.weight = EmbeddingTensor.from_pretrained(reference.predictor_1_final.weight)
        return self

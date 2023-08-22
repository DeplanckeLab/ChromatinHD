import torch
import tqdm.auto as tqdm
import math
from chromatinhd.models.diff.model import splines
from chromatinhd.embedding import EmbeddingTensor

from chromatinhd import get_default_device


class TransformedDistribution(torch.nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def log_prob(self, x, return_transformed=False, **kwargs):
        log_prob = torch.zeros_like(x)
        x_ = x
        x_, logabsdet = self.transform.transform_forward(x_, **kwargs)
        log_prob = log_prob + logabsdet
        if return_transformed:
            return log_prob, x_
        return log_prob

    def sample(self, *args, sample_shape=torch.Size(), device=None, **kwargs):
        y = torch.rand(sample_shape, device=device)
        y, _ = self.transform.transform_inverse(y, *args, **kwargs)
        return y

    def parameters_sparse(self):
        return self.transform.parameters_sparse()

    def parameters_dense(self):
        return self.transform.parameters_dense()

    def forward(self, input):
        pass


class QuadraticSplineTransform(torch.nn.Module):
    bijective = True
    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.real

    def __init__(self, unnormalized_widths, unnormalized_heights):
        super().__init__()
        self.unnormalized_widths = torch.nn.Parameter(unnormalized_widths)
        self.unnormalized_heights = torch.nn.Parameter(unnormalized_heights)

    def transform_forward(self, x, local_region_ix, inverse=False):
        widths = splines.quadratic.calculate_widths(self.unnormalized_widths)
        heights = splines.quadratic.calculate_heights(self.unnormalized_heights, widths)
        bin_left_cdf = splines.quadratic.calculate_bin_left_cdf(heights, widths)
        bin_locations = splines.quadratic.calculate_bin_locations(widths)

        outputs, logabsdet = splines.quadratic.quadratic_spline(
            x,
            widths=widths[local_region_ix],
            heights=heights[local_region_ix],
            bin_left_cdf=bin_left_cdf[local_region_ix],
            bin_locations=bin_locations[local_region_ix],
            inverse=inverse,
        )
        return outputs, logabsdet

    def transform_inverse(self, y, local_region_ix):
        return self.transform_inverse(y, local_region_ix, inverse=True)


def prioritize(x, n, k=2):
    possible = torch.linspace(0, 1, n * k + 2)[1:-1]

    y_ = torch.linspace(0, 1, n * k)
    dist = torch.distributions.Normal(0.0, scale=(1 / (n * k * 2)))

    weights = dist.log_prob(torch.sqrt((x - y_.unsqueeze(1)) ** 2))
    density = torch.logsumexp(weights, 1)

    diff1 = torch.nn.functional.pad(torch.exp(density).diff(), (1, 0))
    diff1 = torch.nn.functional.pad(torch.exp(density).diff().diff(), (1, 1))
    possible_scores = diff1.reshape((n * k, 1)).mean(1)

    chosen_idx = torch.argsort(-possible_scores.abs())[: (n - 2)]
    chosen = torch.nn.functional.pad(torch.sort(possible[chosen_idx])[0], (1, 1))
    chosen[..., -1] = 1.0
    return chosen


def initialize_from_previous(x, n, local_region_ix, n_regions, transforms, device=None):
    q2_orig = torch.linspace(0, 1, n).expand(n_regions, -1)
    # q2_orig = prioritize(x, n).expand(n_regions, -1)
    q2 = q2_orig

    # calculate bin widths
    for transform in transforms:
        q2 = (
            transform.transform_forward(
                q2.flatten(),
                local_region_ix=torch.repeat_interleave(torch.ones(n_regions, dtype=int) * n),
            )[0]
            .detach()
            .reshape(q2.shape)
        )
        # transforms may lead to equivalent consecutive bins
        # which leads to NaN log(width)
        # therefore add some eps to these bins
        if (q2.diff() <= 0).any():
            eps = 1e-6
            q2 = torch.nn.functional.pad(torch.cumsum((q2.diff() + eps), 1), (1, 0))
            q2 = q2 / q2.diff().sum(1, keepdim=True)
            assert (q2.diff() > 0).all()

    unnormalized_widths = torch.log(q2.diff())

    # calculate the bincount in chunks
    # runs on a device
    chunk_width = int(1e6)
    bincount = torch.zeros((n_regions, n - 1), dtype=int)

    transforms = [transform.to(device) for transform in transforms]

    q2 = q2.to(device)

    for x2, local_region_ix2 in tqdm.tqdm(
        zip(x.split(chunk_width, 0), local_region_ix.split(chunk_width, 0)),
        total=math.ceil(x.shape[0] / chunk_width),
    ):
        x2 = x2.to(device)
        local_region_ix2 = local_region_ix2.to(device)
        for transform in transforms:
            x2 = transform.transform_forward(x2, local_region_ix2)[0].detach()

        digitized = torch.clamp(
            torch.searchsorted(q2[local_region_ix2], x2.unsqueeze(-1), right=True).squeeze(-1),
            0,
            n - 1,
        )
        bincount += (
            torch.bincount((digitized + local_region_ix2 * n), minlength=n * n_regions)
            .reshape((n_regions, n))[:, 1:]
            .cpu()
        )

    q2 = q2.to("cpu")
    transforms = [transform.to("cpu") for transform in transforms]

    # calculate the initial bin height (=pdf) by taking the average bincount around each knot
    aroundcounts = torch.nn.functional.pad(bincount / q2.diff(), (1, 0)) + torch.nn.functional.pad(
        bincount / q2.diff(), (0, 1)
    )
    unnormalized_heights = torch.log(aroundcounts + 1e-2)  # small pseudocount for those bins without a single count

    if unnormalized_heights.isnan().any():
        raise ValueError("NaNs in initialized pdf")

    return unnormalized_heights, unnormalized_widths


class QuadraticSplineStack(torch.nn.Module):
    def __init__(self, x, nbins, local_region_ix, n_regions):
        super().__init__()
        splits = []
        unnormalized = []
        transforms = []
        for n in nbins:
            unnormalized_heights, unnormalized_widths = initialize_from_previous(
                x, n, local_region_ix, n_regions, transforms=transforms
            )
            unnormalized.extend([unnormalized_heights, unnormalized_widths])
            splits.extend([unnormalized_heights.shape[-1], unnormalized_widths.shape[-1]])
            transforms.append(QuadraticSplineTransform(unnormalized_widths, unnormalized_heights))
        self.unnormalized = torch.nn.Parameter(torch.cat(unnormalized, -1))
        self.splits = splits

    def _split_parameters(self, x):
        split = x.split(self.splits, -1)
        return zip(split[0::2], split[1::2])

    def transform_forward(self, x, local_region_ix, inverse=False):
        assert x.shape == local_region_ix.shape

        logabsdet = None
        outputs = x
        for unnormalized_heights, unnormalized_widths in self._split_parameters(self.unnormalized):
            widths = splines.quadratic.calculate_widths(unnormalized_widths)
            bin_locations = splines.quadratic.calculate_bin_locations(widths)

            heights = splines.quadratic.calculate_heights(unnormalized_heights, widths)
            bin_left_cdf = splines.quadratic.calculate_bin_left_cdf(heights, widths)

            outputs, logabsdet_ = splines.quadratic.quadratic_spline(
                outputs,
                widths=widths[local_region_ix],
                heights=heights[local_region_ix],
                bin_left_cdf=bin_left_cdf[local_region_ix],
                bin_locations=bin_locations[local_region_ix],
                inverse=inverse,
            )
            if logabsdet is None:
                logabsdet = logabsdet_
            else:
                logabsdet = logabsdet + logabsdet_
        return outputs, logabsdet

    def transform_inverse(self, y, local_region_ix):
        return self.transform_forward(y, local_region_ix=local_region_ix, inverse=True)


class DifferentialQuadraticSplineStack(torch.nn.Module):
    def __init__(self, nbins, n_regions):
        self.nbins = nbins

        super().__init__()
        splits_heights = []
        splits_widths = []
        split_deltas = []
        for n in nbins:
            n_heights = n
            n_widths = n - 1

            splits_heights.append(n_heights)
            splits_widths.append(n_widths)
            split_deltas.append(n_heights)

        self.unnormalized_heights = EmbeddingTensor(n_regions, (sum(splits_heights),), sparse=True)

        self.unnormalized_widths = EmbeddingTensor(n_regions, (sum(splits_widths),), sparse=True)

        self.unnormalized_heights.data[:] = 0.0
        self.unnormalized_widths.data[:] = 0.0

        self.splits_heights = splits_heights
        self.splits_widths = splits_widths
        self.split_deltas = split_deltas

    def _split_parameters(self, x, splits):
        return x.split(splits, -1)

    def transform(self, x, regions_oi, local_region_ix, delta, inverse=False):
        assert x.shape == local_region_ix.shape

        logabsdet = None
        outputs = x

        stride = 1 if not inverse else -1

        unnormalized_widths = self.unnormalized_widths(regions_oi)
        unnormalized_heights = self.unnormalized_heights(regions_oi)
        for unnormalized_heights, unnormalized_widths, delta_heights in zip(
            self._split_parameters(unnormalized_heights, self.splits_heights)[::stride],
            self._split_parameters(unnormalized_widths, self.splits_widths)[::stride],
            self._split_parameters(delta, self.split_deltas)[::stride],
        ):
            # calculate widths for all regions as they do not depend on the cell
            widths = splines.quadratic.calculate_widths(unnormalized_widths)
            bin_locations = splines.quadratic.calculate_bin_locations(widths)

            # select and calculate widths for each cut site (i.e. cellxregion)
            # use index_select here as it is much faster in backwards than regular indexing
            widths = widths.index_select(0, local_region_ix)
            bin_locations = bin_locations.index_select(0, local_region_ix)

            unnormalized_heights = unnormalized_heights.index_select(0, local_region_ix) + delta_heights
            heights = splines.quadratic.calculate_heights(unnormalized_heights, widths)
            bin_left_cdf = splines.quadratic.calculate_bin_left_cdf(heights, widths)

            outputs, logabsdet_ = splines.quadratic.quadratic_spline(
                outputs,
                widths=widths,
                heights=heights,
                bin_left_cdf=bin_left_cdf,
                bin_locations=bin_locations,
                inverse=inverse,
            )
            if logabsdet is None:
                logabsdet = logabsdet_
            else:
                logabsdet = logabsdet + logabsdet_
        return outputs, logabsdet

    def transform_forward(self, x, regions_oi, local_region_ix, delta):
        return self.transform(x, regions_oi=regions_oi, local_region_ix=local_region_ix, delta=delta, inverse=False)

    def transform_inverse(self, y, regions_oi, local_region_ix, delta):
        return self.transform(y, regions_oi=regions_oi, local_region_ix=local_region_ix, delta=delta, inverse=True)

    def parameters_sparse(self):
        return [self.unnormalized_heights.weight, self.unnormalized_widths.weight]

    def parameters_dense(self):
        return []

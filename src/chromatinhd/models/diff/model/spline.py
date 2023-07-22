import torch
import tqdm.auto as tqdm
import math
from chromatinhd.models.diff.model import splines
from chromatinhd.embedding import EmbeddingTensor


class TransformedDistribution(torch.nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def log_prob(self, x, *args, **kwargs):
        log_prob = torch.zeros_like(x)
        x_ = x
        x_, logabsdet = self.transform.transform_forward(x_, *args, **kwargs)
        log_prob = log_prob + logabsdet
        return log_prob

    def sample(self, sample_shape=torch.Size(), *args, device=None, **kwargs):
        y = torch.rand(sample_shape, device=device)
        y, _ = self.transform.transform_inverse(y, *args, **kwargs)
        return y

    def parameters_sparse(self):
        return self.transform.parameters_sparse()

    def parameters_dense(self):
        return self.transform.parameters_dense()


class QuadraticSplineTransform(torch.nn.Module):
    bijective = True
    domain = torch.distributions.constraints.real
    codomain = torch.distributions.constraints.real

    def __init__(self, unnormalized_widths, unnormalized_heights):
        super().__init__()
        self.unnormalized_widths = torch.nn.Parameter(unnormalized_widths)
        self.unnormalized_heights = torch.nn.Parameter(unnormalized_heights)

    def transform_forward(self, x, local_gene_ix, inverse=False):
        widths = splines.quadratic.calculate_widths(self.unnormalized_widths)
        heights = splines.quadratic.calculate_heights(self.unnormalized_heights, widths)
        bin_left_cdf = splines.quadratic.calculate_bin_left_cdf(heights, widths)
        bin_locations = splines.quadratic.calculate_bin_locations(widths)

        outputs, logabsdet = splines.quadratic.quadratic_spline(
            x,
            widths=widths[local_gene_ix],
            heights=heights[local_gene_ix],
            bin_left_cdf=bin_left_cdf[local_gene_ix],
            bin_locations=bin_locations[local_gene_ix],
            inverse=inverse,
        )
        return outputs, logabsdet

    def transform_inverse(self, y, local_gene_ix):
        return self.transform_inverse(y, local_gene_ix, inverse=True)


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


def initialize_from_previous(x, n, local_gene_ix, n_genes, transforms, device="cuda"):
    q2_orig = torch.linspace(0, 1, n).expand(n_genes, -1)
    # q2_orig = prioritize(x, n).expand(n_genes, -1)
    q2 = q2_orig

    # calculate bin widths
    for transform in transforms:
        q2 = (
            transform.transform_forward(
                q2.flatten(),
                local_gene_ix=torch.repeat_interleave(
                    torch.ones(n_genes, dtype=int) * n
                ),
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
    bincount = torch.zeros((n_genes, n - 1), dtype=int)

    transforms = [transform.to(device) for transform in transforms]

    q2 = q2.to(device)

    for x2, local_gene_ix2 in tqdm.tqdm(
        zip(x.split(chunk_width, 0), local_gene_ix.split(chunk_width, 0)),
        total=math.ceil(x.shape[0] / chunk_width),
    ):
        x2 = x2.to(device)
        local_gene_ix2 = local_gene_ix2.to(device)
        for transform in transforms:
            x2 = transform.transform_forward(x2, local_gene_ix2)[0].detach()

        digitized = torch.clamp(
            torch.searchsorted(
                q2[local_gene_ix2], x2.unsqueeze(-1), right=True
            ).squeeze(-1),
            0,
            n - 1,
        )
        bincount += (
            torch.bincount((digitized + local_gene_ix2 * n), minlength=n * n_genes)
            .reshape((n_genes, n))[:, 1:]
            .cpu()
        )

    q2 = q2.to("cpu")
    transforms = [transform.to("cpu") for transform in transforms]

    # calculate the initial bin height (=pdf) by taking the average bincount around each knot
    aroundcounts = torch.nn.functional.pad(
        bincount / q2.diff(), (1, 0)
    ) + torch.nn.functional.pad(bincount / q2.diff(), (0, 1))
    unnormalized_heights = torch.log(
        aroundcounts + 1e-2
    )  # small pseudocount for those bins without a single count

    if unnormalized_heights.isnan().any():
        raise ValueError("NaNs in initialized pdf")

    return unnormalized_heights, unnormalized_widths


class QuadraticSplineStack(torch.nn.Module):
    def __init__(self, x, nbins, local_gene_ix, n_genes):
        super().__init__()
        splits = []
        unnormalized = []
        transforms = []
        for n in nbins:
            unnormalized_heights, unnormalized_widths = initialize_from_previous(
                x, n, local_gene_ix, n_genes, transforms=transforms
            )
            unnormalized.extend([unnormalized_heights, unnormalized_widths])
            splits.extend(
                [unnormalized_heights.shape[-1], unnormalized_widths.shape[-1]]
            )
            transforms.append(
                QuadraticSplineTransform(unnormalized_widths, unnormalized_heights)
            )
        self.unnormalized = torch.nn.Parameter(torch.cat(unnormalized, -1))
        self.splits = splits

    def _split_parameters(self, x):
        split = x.split(self.splits, -1)
        return zip(split[0::2], split[1::2])

    def transform_forward(self, x, local_gene_ix, inverse=False):
        assert x.shape == local_gene_ix.shape

        logabsdet = None
        outputs = x
        for unnormalized_heights, unnormalized_widths in self._split_parameters(
            self.unnormalized
        ):
            widths = splines.quadratic.calculate_widths(unnormalized_widths)
            bin_locations = splines.quadratic.calculate_bin_locations(widths)

            heights = splines.quadratic.calculate_heights(unnormalized_heights, widths)
            bin_left_cdf = splines.quadratic.calculate_bin_left_cdf(heights, widths)

            outputs, logabsdet_ = splines.quadratic.quadratic_spline(
                outputs,
                widths=widths[local_gene_ix],
                heights=heights[local_gene_ix],
                bin_left_cdf=bin_left_cdf[local_gene_ix],
                bin_locations=bin_locations[local_gene_ix],
                inverse=inverse,
            )
            if logabsdet is None:
                logabsdet = logabsdet_
            else:
                logabsdet = logabsdet + logabsdet_
        return outputs, logabsdet

    def transform_inverse(self, y, local_gene_ix):
        return self.transform_forward(y, local_gene_ix=local_gene_ix, inverse=True)


class DifferentialQuadraticSplineStack(torch.nn.Module):
    def __init__(self, nbins, n_genes, local_gene_ix=None, x=None):
        self.nbins = nbins

        super().__init__()
        unnormalized_heights = []
        unnormalized_widths = []
        splits_heights = []
        splits_widths = []
        transforms = []
        split_deltas = []
        for n in nbins:
            n_heights = n
            n_widths = n - 1

            if x is not None:
                unnormalized_heights_, unnormalized_widths_ = initialize_from_previous(
                    x, n, local_gene_ix, n_genes, transforms=transforms
                )
                unnormalized_heights.append(unnormalized_heights_)
                unnormalized_widths.append(unnormalized_widths_)

                assert unnormalized_heights_.shape[-1] == n_heights
                assert unnormalized_widths_.shape[-1] == n_widths

                transforms.append(
                    QuadraticSplineTransform(
                        unnormalized_widths_, unnormalized_heights_
                    )
                )
            splits_heights.append(n_heights)
            splits_widths.append(n_widths)
            split_deltas.append(n_heights)

        self.unnormalized_heights = EmbeddingTensor(
            n_genes, (sum(splits_heights),), sparse=True
        )

        self.unnormalized_widths = EmbeddingTensor(
            n_genes, (sum(splits_widths),), sparse=True
        )

        if x is not None:
            unnormalized_heights = torch.cat(unnormalized_heights, -1)
            unnormalized_widths = torch.cat(unnormalized_widths, -1)
            self.unnormalized_heights.data = unnormalized_heights
            self.unnormalized_widths.data = unnormalized_widths
        else:
            self.unnormalized_heights.data[:] = 0.0
            self.unnormalized_widths.data[:] = 0.0

        self.splits_heights = splits_heights
        self.splits_widths = splits_widths
        self.split_deltas = split_deltas

    def _split_parameters(self, x, splits):
        return x.split(splits, -1)

    def transform_forward(self, x, genes_oi, local_gene_ix, delta, inverse=False):
        assert x.shape == local_gene_ix.shape

        logabsdet = None
        outputs = x

        stride = 1 if not inverse else -1

        unnormalized_widths = self.unnormalized_widths(genes_oi)
        unnormalized_heights = self.unnormalized_heights(genes_oi)
        for unnormalized_heights, unnormalized_widths, delta_heights in zip(
            self._split_parameters(unnormalized_heights, self.splits_heights)[::stride],
            self._split_parameters(unnormalized_widths, self.splits_widths)[::stride],
            self._split_parameters(delta, self.split_deltas)[::stride],
        ):
            widths = splines.quadratic.calculate_widths(unnormalized_widths)
            bin_locations = splines.quadratic.calculate_bin_locations(widths)

            # use index_select here as it is much faster in backwards than regular indexing
            widths = widths.index_select(0, local_gene_ix)
            bin_locations = bin_locations.index_select(0, local_gene_ix)

            unnormalized_heights = (
                unnormalized_heights.index_select(0, local_gene_ix) + delta_heights
            )
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

    def transform_inverse(self, y, genes_oi, local_gene_ix, delta):
        return self.transform_forward(
            y, genes_oi=genes_oi, local_gene_ix=local_gene_ix, delta=delta, inverse=True
        )

    def parameters_sparse(self):
        return [self.unnormalized_heights.weight, self.unnormalized_widths.weight]

    def parameters_dense(self):
        return []

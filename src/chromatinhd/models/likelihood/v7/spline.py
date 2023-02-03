import torch
import tqdm.auto as tqdm
import math
from . import quadratic
from chromatinhd.embedding import EmbeddingTensor
import numpy as np


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
        widths = quadratic.calculate_widths(self.unnormalized_widths)
        heights = quadratic.calculate_heights(self.unnormalized_heights, widths)
        bin_left_cdf = quadratic.calculate_bin_left_cdf(heights, widths)
        bin_locations = quadratic.calculate_bin_locations(widths)

        outputs, logabsdet = quadratic.quadratic_spline(
            x,
            widths=widths[local_gene_ix],
            heights=heights[local_gene_ix],
            bin_left_cdf=bin_left_cdf[local_gene_ix],
            bin_locations=bin_locations[local_gene_ix],
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
        aroundcounts + 1
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
            widths = quadratic.calculate_widths(unnormalized_widths)
            bin_locations = quadratic.calculate_bin_locations(widths)

            heights = quadratic.calculate_heights(unnormalized_heights, widths)
            bin_left_cdf = quadratic.calculate_bin_left_cdf(heights, widths)

            outputs, logabsdet_ = quadratic.quadratic_spline(
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

        self.unnormalized_heights = torch.nn.Parameter(
            torch.zeros(n_genes, sum(splits_heights), requires_grad=True)
        )

        self.unnormalized_widths = torch.nn.Parameter(
            torch.zeros(n_genes, sum(splits_widths), requires_grad=True)
        )

        if x is not None:
            unnormalized_heights = torch.cat(unnormalized_heights, -1)
            unnormalized_widths = torch.cat(unnormalized_widths, -1)
            self.unnormalized_heights.data = unnormalized_heights
            self.unnormalized_widths.data = unnormalized_widths

        self.splits_heights = splits_heights
        self.splits_widths = splits_widths
        self.split_deltas = split_deltas

    def _split_parameters(self, x, splits, inverse=False):
        if inverse:
            return x.split(splits, -1)[::-1]
        else:
            return x.split(splits, -1)

    def transform_progressive(
        self,
        cut_positions,
        cut_local_reflatentxgene_ix,
        cut_local_gene_ix,
        cut_local_reflatent_ix,
        mixture_delta_reflatentxgene,
        inverse=False,
    ):
        assert cut_positions.shape == cut_local_reflatentxgene_ix.shape
        assert cut_positions.shape == cut_local_gene_ix.shape

        n_genes = mixture_delta_reflatentxgene.shape[1]
        n_reflatent = mixture_delta_reflatentxgene.shape[0]

        genespacing = (
            torch.ones(
                (n_reflatent, n_genes),
                dtype=cut_positions.dtype,
                device=cut_positions.device,
            )
            / n_genes
        )

        # calculate heights and weights
        transformation_data = []
        for unnormalized_heights, unnormalized_widths, delta_heights, num_bins in zip(
            self._split_parameters(self.unnormalized_heights, self.splits_heights),
            self._split_parameters(self.unnormalized_widths, self.splits_widths),
            self._split_parameters(mixture_delta_reflatentxgene, self.split_deltas),
            self.nbins,
        ):
            # calculate flattened widths per reflatent
            gene_bin_positions = (
                torch.arange(n_genes) + 1
            ) * unnormalized_heights.shape[-1] - 1
            widths = torch.nn.functional.pad(
                (
                    quadratic.calculate_widths(unnormalized_widths).unsqueeze(0)
                    * genespacing.unsqueeze(-1)
                ),
                (0, 1),
                value=0,
            ).flatten(-2, -1)[..., :-1]
            bin_locations = quadratic.calculate_bin_locations(widths)

            unnormalized_heights = (unnormalized_heights + delta_heights).flatten(1, 2)
            heights = quadratic.calculate_heights(unnormalized_heights, widths)

            bin_left_cdf = quadratic.calculate_bin_left_cdf(heights, widths)

            # update genespacing based on the gene boundaries in the cdf
            # the first gene always starts at 0
            genespacing = torch.diff(
                torch.nn.functional.pad(bin_left_cdf[..., gene_bin_positions], (1, 0)),
                dim=-1,
            )

            transformation_data.append(
                (widths, heights, bin_left_cdf, bin_locations, num_bins)
            )

        output = cut_positions
        logabsdets = []
        outputs = []

        logabsdet = torch.zeros_like(cut_positions)

        logabsdets.append(logabsdet)
        outputs.append(output)

        step = 1
        if inverse:
            step = -1

        for (
            widths,
            heights,
            bin_left_cdf,
            bin_locations,
            num_bins,
        ) in transformation_data[::step]:

            # select the widths, heights, left_cdf and bin_locations for each fragment
            # use index_select here as it is much faster in backwards than regular indexing
            print(widths.shape)
            print(cut_local_reflatent_ix.shape)
            cut_widths = torch.index_select(widths, 0, cut_local_reflatent_ix)
            cut_heights = torch.index_select(heights, 0, cut_local_reflatent_ix)
            cut_bin_left_cdf = torch.index_select(
                bin_left_cdf, 0, cut_local_reflatent_ix
            )
            cut_bin_locations = torch.index_select(
                bin_locations, 0, cut_local_reflatent_ix
            )

            if inverse:
                # alternative way that does a full search on all data
                # makes mistakes at the boundaries of two genes
                bin_idx = (
                    torch.searchsorted(cut_bin_left_cdf, output.unsqueeze(-1)).squeeze(
                        -1
                    )
                    - 1
                )
                bin_idx = torch.clamp(bin_idx, 0)

                # following way runs but does not work correctly
                # I guess it has to do with that we cannot know beforehand which gene a sample belongs to
                #! this makes the inverse quite slow, especially with many genes!
                # bin_left_cdf_genewise = bin_left_cdf.reshape(
                #     (n_reflatent * n_genes, num_bins)
                # )
                # cut_left_cdf_genewise = torch.index_select(
                #     bin_left_cdf_genewise, 0, cut_local_reflatentxgene_ix
                # )
                # bin_idx = cut_local_gene_ix * num_bins + torch.clamp(
                #     torch.searchsorted(
                #         cut_left_cdf_genewise, output.unsqueeze(-1)
                #     ).squeeze(-1)
                #     - 1,
                #     0,
                #     num_bins - 1,
                # )
                # bin_idx = torch.clamp(bin_idx, 0, widths.shape[-1] - 1)

                # import matplotlib.pyplot as plt

                # plt.scatter(bin_idx2, bin_idx)

            else:
                # actual way
                bin_locations_genewise = bin_locations.reshape(
                    (n_reflatent * n_genes, num_bins)
                )
                cut_bin_locatons_genewise = torch.index_select(
                    bin_locations_genewise, 0, cut_local_reflatentxgene_ix
                )
                bin_idx = cut_local_gene_ix * num_bins + torch.clamp(
                    torch.searchsorted(
                        cut_bin_locatons_genewise, output.unsqueeze(-1)
                    ).squeeze(-1)
                    - 1,
                    0,
                    num_bins - 2,
                )

                # alternative way that does a full search on all data
                # makes mistakes at the boundaries of two genes
                # bin_idx2 = (
                #     torch.searchsorted(cut_bin_locations, output.unsqueeze(-1)).squeeze(
                #         -1
                #     )
                #     - 1
                # )
                # bin_idx2 = torch.clamp(bin_idx2, 0, widths.shape[-1] - 1)

            output, logabsdet_ = quadratic.quadratic_spline(
                output,
                widths=cut_widths,
                heights=cut_heights,
                bin_left_cdf=cut_bin_left_cdf,
                bin_locations=cut_bin_locations,
                bin_idx=bin_idx,
                inverse=inverse,
            )

            logabsdet = logabsdet + logabsdet_

            logabsdets.append(logabsdet)
            outputs.append(output)

        return logabsdets, outputs

    def transform_forward(
        self,
        cut_positions,
        cut_local_reflatentxgene_ix,
        cut_local_gene_ix,
        cut_local_reflatent_ix,
        mixture_delta_reflatentxgene,
        inverse=False,
    ):
        logabsdets, outputs = self.transform_progressive(
            cut_positions,
            cut_local_reflatentxgene_ix,
            cut_local_gene_ix,
            cut_local_reflatent_ix,
            mixture_delta_reflatentxgene,
            inverse=inverse,
        )
        return (
            outputs[-1],
            logabsdets[-1],
        )

    def transform_inverse(self, y, local_gene_ix):
        return self.transform_forward(y, local_gene_ix=local_gene_ix, inverse=True)

    def parameters_sparse(self):
        return []

    def parameters_dense(self):
        return [
            self.unnormalized_heights,
            # self.unnormalized_widths,
        ]

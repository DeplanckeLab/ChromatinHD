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

        outputs, logabsdet = quadratic.quadratic_spline2(
            x,
            widths=widths[local_gene_ix],
            heights=heights[local_gene_ix],
            bin_left_cdf=bin_left_cdf[local_gene_ix],
            bin_locations=bin_locations[local_gene_ix],
        )
        return outputs, logabsdet

    def transform_inverse(self, y, local_gene_ix):
        return self.transform_inverse(y, local_gene_ix, inverse=True)


def initialize_from_previous(x, n, local_gene_ix, n_genes, transforms, device="cuda"):
    q2_orig = torch.linspace(0, 1, n, dtype=torch.float64).expand(n_genes, -1)
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
            torch.zeros(
                n_genes, sum(splits_heights), requires_grad=True, dtype=torch.float64
            )
        )

        self.unnormalized_widths = torch.nn.Parameter(
            torch.zeros(
                n_genes, sum(splits_widths), requires_grad=True, dtype=torch.float64
            )
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
            # calculate bin_idx
            if inverse:
                raise NotImplementedError(
                    "This is creatng massive tensors (and will make errors at the gene boundaries)"
                    ", check v7 on an initial approach (which does not work) on how to do this on a per-gene basis"
                )
                # bin_idx = (
                #     torch.searchsorted(cut_bin_left_cdf, output.unsqueeze(-1)).squeeze(
                #         -1
                #     )
                #     - 1
                # )
                # bin_idx = torch.clamp(bin_idx, 0)

            else:
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

            # select the widths, heights, left_cdf and bin_locations for each fragment
            # use index_select here as it is much faster in backwards than regular indexing
            cut_local_reflatentxgenexbin_ix = (
                bin_idx + cut_local_reflatent_ix * num_bins * n_genes
            )
            input_bin_locations = bin_locations.flatten().index_select(
                0, cut_local_reflatentxgenexbin_ix
            )
            input_bin_left_cdf = bin_left_cdf.flatten().index_select(
                0, cut_local_reflatentxgenexbin_ix
            )
            input_left_heights = heights.flatten().index_select(
                0, cut_local_reflatentxgenexbin_ix
            )
            input_right_heights = heights.flatten().index_select(
                0, cut_local_reflatentxgenexbin_ix + 1
            )

            cut_local_reflatentxgenexbin2_ix = bin_idx + cut_local_reflatent_ix * (
                num_bins * n_genes - 1
            )
            input_bin_widths = widths.flatten().index_select(
                0, cut_local_reflatentxgenexbin2_ix
            )

            output, logabsdet_ = quadratic.quadratic_spline(
                output,
                input_bin_widths=input_bin_widths,
                input_left_heights=input_left_heights,
                input_right_heights=input_right_heights,
                input_bin_locations=input_bin_locations,
                input_bin_left_cdf=input_bin_left_cdf,
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

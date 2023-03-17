import torch
from chromatinhd import splines
from chromatinhd.embedding import EmbeddingTensor


class TransformedDistribution(torch.nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def log_prob(self, x, *args, **kwargs):
        # uniform distribution [0, 1] has likelihood of 1 everywhere
        # so that means it has log_prob of 0
        log_prob = torch.zeros_like(x)

        # apply transform and update logprob with log abs determinant jacobian
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
    def __init__(self, nbins, n_genes):
        self.nbins = nbins

        super().__init__()

        # calculate how many heights, widths and (height_)deltas we will need
        splits_heights = []
        splits_widths = []
        split_deltas = []
        for n in nbins:
            n_heights = n
            n_widths = n - 1

            splits_heights.append(n_heights)
            splits_widths.append(n_widths)
            split_deltas.append(n_heights)

        # set up the baseline height and width
        self.unnormalized_heights = EmbeddingTensor(
            n_genes, (sum(splits_heights),), sparse=True
        )

        self.unnormalized_widths = EmbeddingTensor(
            n_genes, (sum(splits_widths),), sparse=True
        )

        self.unnormalized_heights.data.zero_()
        self.unnormalized_widths.data.zero_()

        self.splits_heights = splits_heights
        self.splits_widths = splits_widths
        self.split_deltas = split_deltas

    def _split_parameters(self, x, splits):
        return x.split(splits, -1)

    def transform_forward(self, x, genes_oi, local_gene_ix, delta, inverse=False):
        assert x.shape == local_gene_ix.shape

        logabsdet = None
        outputs = x

        unnormalized_widths = self.unnormalized_widths(genes_oi)
        unnormalized_heights = self.unnormalized_heights(genes_oi)

        # apply the consecutive transformations
        for unnormalized_heights, unnormalized_widths, delta_heights in zip(
            self._split_parameters(unnormalized_heights, self.splits_heights),
            self._split_parameters(unnormalized_widths, self.splits_widths),
            self._split_parameters(delta, self.split_deltas),
        ):
            # calculate widths for all genes
            widths = splines.quadratic.calculate_widths(unnormalized_widths)
            bin_locations = splines.quadratic.calculate_bin_locations(widths)

            # use index_select here as it is much faster in backwards than regular indexing
            # get widths and bin_locations for each cut
            widths = widths.index_select(0, local_gene_ix)
            bin_locations = bin_locations.index_select(0, local_gene_ix)

            # get heights and bin_left_cdf for each cut
            unnormalized_heights = (
                unnormalized_heights.index_select(0, local_gene_ix) + delta_heights
            )
            heights = splines.quadratic.calculate_heights(unnormalized_heights, widths)
            bin_left_cdf = splines.quadratic.calculate_bin_left_cdf(heights, widths)

            # apply the spline transformation
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

    def transform_inverse(self, y, local_gene_ix):
        return self.transform_forward(y, local_gene_ix=local_gene_ix, inverse=True)

    def parameters_sparse(self):
        return [self.unnormalized_heights.weight, self.unnormalized_widths.weight]

    def parameters_dense(self):
        return []

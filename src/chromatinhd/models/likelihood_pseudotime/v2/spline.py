import torch
from chromatinhd import splines
from chromatinhd.embedding import EmbeddingTensor


class TransformedDistribution(torch.nn.Module):
    def __init__(self, transform):
        print("---  TransformedDistribution.__init__()  ---")
        super().__init__()
        self.transform = transform

    def log_prob(self, x, *args, **kwargs):
        print("---  TransformedDistribution.log_prob()  ---")
        # uniform distribution [0, 1] has likelihood of 1 everywhere
        # so that means it has log_prob of 0
        log_prob = torch.zeros_like(x)

        # apply transform and update logprob with log abs determinant jacobian
        x_, logabsdet = self.transform.transform_forward(x, *args, **kwargs)
        log_prob = log_prob + logabsdet
        return log_prob

    def sample(self, sample_shape=torch.Size(), *args, device=None, **kwargs):
        print("---  TransformedDistribution.sample()  ---")
        y = torch.rand(sample_shape, device=device)
        y, _ = self.transform.transform_inverse(y, *args, **kwargs)
        return y

    def parameters_sparse(self):
        return self.transform.parameters_sparse()

    def parameters_dense(self):
        return self.transform.parameters_dense()


class DifferentialQuadraticSplineStack(torch.nn.Module):
    def __init__(self, nbins, n_genes):
        print("---  DifferentialQuadraticSplineStack.__init__()  ---")
        super().__init__()

        # calculate how many heights, widths and (height_)deltas we will need
        splits_heights, splits_widths, split_deltas = [n for n in nbins], [n-1 for n in nbins], [n for n in nbins]
        
        self.nbins = nbins
        self.splits_heights = splits_heights
        self.splits_widths = splits_widths
        self.split_deltas = split_deltas
        # set up the baseline height and width
        self.unnormalized_heights = EmbeddingTensor(n_genes, (sum(splits_heights),), sparse=True)
        self.unnormalized_widths = EmbeddingTensor(n_genes, (sum(splits_widths),), sparse=True)
        self.unnormalized_heights.data.zero_()
        self.unnormalized_widths.data.zero_()

    def _split_parameters(self, x, splits):
        print("---  DifferentialQuadraticSplineStack._split_parameters()  ---")
        return x.split(splits, -1)

    def transform_forward(self, x, genes_oi, local_gene_ix, delta, inverse=False):
        print("---  DifferentialQuadraticSplineStack.transform_forward()  ---")
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

            # print("widths", widths.shape)
            # print("bin_locations", bin_locations.shape)
            # print("local_gene_ix", local_gene_ix)

            # use index_select here as it is much faster in backwards than regular indexing
            # get widths and bin_locations for each cut
            widths = widths.index_select(0, local_gene_ix)
            bin_locations = bin_locations.index_select(0, local_gene_ix)

            # print("widths", widths.shape)
            # print("bin_locations", bin_locations.shape)

            # get heights and bin_left_cdf for each cut
            unnormalized_heights = (unnormalized_heights.index_select(0, local_gene_ix) + delta_heights)
            heights = splines.quadratic.calculate_heights(unnormalized_heights, widths)
            bin_left_cdf = splines.quadratic.calculate_bin_left_cdf(heights, widths)

            # print("unnormalized_heights", unnormalized_heights.shape)
            # print("heights", heights.shape)
            # print("bin_left_cdf", bin_left_cdf.shape)

            # apply the spline transformation
            outputs, logabsdet_ = splines.quadratic.quadratic_spline(
                outputs,
                widths=widths,
                heights=heights,
                bin_left_cdf=bin_left_cdf,
                bin_locations=bin_locations,
                inverse=inverse,
            )

            # print("outputs", outputs.shape)
            # print("logabsdet_", logabsdet_.shape)

            if logabsdet is None:
                logabsdet = logabsdet_
            else:
                logabsdet = logabsdet + logabsdet_

            # print("logabsdet", logabsdet.shape)
        return outputs, logabsdet

    def transform_inverse(self, y, local_gene_ix):
        print("---  DifferentialQuadraticSplineStack.transform_inverse()  ---")
        return self.transform_forward(y, local_gene_ix=local_gene_ix, inverse=True)

    def parameters_sparse(self):
        return [self.unnormalized_heights.weight, self.unnormalized_widths.weight]

    def parameters_dense(self):
        return []

import torch
import tqdm.auto as tqdm
import math
from chromatinhd.models.diff.model import splines
from chromatinhd.embedding import EmbeddingTensor

from chromatinhd import get_default_device


class WindowStack(torch.nn.Module):
    def __init__(self, nbins, n_genes, window_width=20000):
        self.nbins = nbins

        super().__init__()
        splits_heights = []
        n_nf_params = 0
        bins = []
        for n in nbins:
            assert 20000 % n == 0
            n_heights = n

            bins.append(torch.linspace(0, window_width, n + 1)[1:-1].contiguous())

            splits_heights.append(n_heights)

            n_nf_params += n_heights

        self.splits_heights = splits_heights
        self.n_nf_params = n_nf_params

        self.window_width = window_width
        self.bins = bins

        self.scale = torch.nn.Parameter(torch.tensor(10.0))

    def _split_parameters(self, x, splits):
        return x.split(splits, -1)

    def log_prob(self, x, genes_oi, local_gene_ix, unnormalized_heights_gene, inverse=False):
        assert x.shape == local_gene_ix.shape

        stride = 1 if not inverse else -1

        probs = torch.zeros((genes_oi.shape[0], self.window_width), device=x.device)

        for (unnormalized_heights,) in zip(
            self._split_parameters(unnormalized_heights_gene, self.splits_heights)[::stride],
        ):
            probs += torch.repeat_interleave(
                unnormalized_heights,
                self.window_width // unnormalized_heights.shape[1],
                dim=1,
            )

        probs = torch.nn.functional.log_softmax(probs, 1)

        genexposition_ix = local_gene_ix * self.window_width + x

        # if not self.training:
        #     import matplotlib.pyplot as plt
        #     import numpy as np

        #     print(torch.exp(probs)[0].sum())

        #     fig, ax = plt.subplots(figsize=(3, 3))
        #     ax.plot(np.exp(probs[0].detach().cpu().numpy()))
        #     ax.set_ylim(0)
        #     ax.set_xlim(0, self.window_width)

        logprob = probs.flatten()[genexposition_ix]

        return logprob


class FragmentCountDistribution(torch.nn.Module):
    def log_prob(self, data):
        bincount = torch.bincount(data.motifs.local_gene_ix, minlength=data.minibatch.n_genes) / self.window_width
        logits_count_genes = self.nn_logit(bincount[:, None].float()).reshape((data.minibatch.n_genes,))
        count = torch.bincount(
            data.fragments.local_cellxgene_ix, minlength=data.minibatch.n_cells * data.minibatch.n_genes
        ).reshape((data.minibatch.n_cells, data.minibatch.n_genes))
        likelihood_count = torch.distributions.Geometric(logits=logits_count_genes).log_prob(count)
        return likelihood_count


class FragmentCountDistribution1(FragmentCountDistribution):
    def __init__(self, fragments, motifscan):
        super().__init__()
        self.nn_logit = torch.nn.Sequential(
            torch.nn.Linear(1, 1),
        )
        self.window_width = fragments.regions.window[1] - fragments.regions.window[0]


class FragmentCountDistribution2(FragmentCountDistribution):
    def __init__(self, fragments, motifscan):
        super().__init__()
        self.nn_logit = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )
        self.window_width = fragments.regions.window[1] - fragments.regions.window[0]


class FragmentCountDistributionBaseline(FragmentCountDistribution):
    def __init__(
        self,
        fragments,
        motifscan,
    ):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.zeros((1,)))

    def log_prob(self, data):
        count = torch.bincount(
            data.fragments.local_cellxgene_ix, minlength=data.minibatch.n_cells * data.minibatch.n_genes
        ).reshape((data.minibatch.n_cells, data.minibatch.n_genes))
        likelihood_count = torch.distributions.Geometric(logits=self.logit).log_prob(count)
        return likelihood_count


class FragmentPositionDistribution(torch.nn.Module):
    def log_prob(self, data):
        nf_params = []
        coordinates = data.fragments.coordinates[:, 0] - self.window[0]
        for predictor, binset in zip(self.predictors, self.position_dist.bins):
            bin_width = binset[1] - binset[0]
            localgenexbin = data.motifs.local_gene_ix * (len(binset) + 1) + torch.searchsorted(
                binset.to(coordinates.device), data.motifs.positions - self.window[0]
            )
            bincount = torch.bincount(localgenexbin, minlength=data.minibatch.n_genes * (len(binset) + 1)) / bin_width
            nf_params_binset = predictor(bincount[:, None].float()).reshape((data.minibatch.n_genes, len(binset) + 1))
            nf_params.append(nf_params_binset)
        nf_params = torch.cat(nf_params, dim=-1)

        unnormalized_heights_gene = nf_params

        # p(pos|gene,motifs_gene)
        likelihood_position = self.position_dist.log_prob(
            coordinates,
            genes_oi=data.minibatch.genes_oi_torch,
            local_gene_ix=data.fragments.local_gene_ix,
            unnormalized_heights_gene=unnormalized_heights_gene,
        )
        return likelihood_position


class FragmentPositionDistribution1(FragmentPositionDistribution):
    def __init__(self, fragments, motifscan, nbins=(10, 20, 50)):
        super().__init__()
        self.window = fragments.regions.window
        self.window_width = fragments.regions.window[1] - fragments.regions.window[0]
        self.position_dist = WindowStack(
            nbins=nbins,
            n_genes=fragments.n_genes,
            window_width=self.window_width,
        )

        self.predictors = [torch.nn.Sequential(torch.nn.Linear(1, 1)) for _ in range(len(self.position_dist.bins))]
        for i, predictor in enumerate(self.predictors):
            setattr(self, f"predictor_{i}", predictor)


class FragmentPositionDistribution2(FragmentPositionDistribution):
    def __init__(self, fragments, motifscan, nbins=(10, 20, 50)):
        super().__init__()
        self.window = fragments.regions.window
        self.window_width = fragments.regions.window[1] - fragments.regions.window[0]
        self.position_dist = WindowStack(
            nbins=nbins,
            n_genes=fragments.n_genes,
            window_width=self.window_width,
        )

        self.predictors = [
            torch.nn.Sequential(
                torch.nn.Linear(1, 10),
                torch.nn.Dropout(0.1),
                torch.nn.ReLU(),
                torch.nn.Linear(10, 1),
            )
            for _ in range(len(self.position_dist.bins))
        ]
        for i, predictor in enumerate(self.predictors):
            setattr(self, f"predictor_{i}", predictor)


class FragmentPositionDistributionBaseline(FragmentPositionDistribution):
    def __init__(self, fragments, motifscan, nbins=(10, 20, 50)):
        super().__init__()
        self.region_width = fragments.regions.window[1] - fragments.regions.window[0]

    def log_prob(self, data):
        likelihood_position = torch.zeros(
            data.fragments.n_fragments, device=data.fragments.coordinates.device
        ) - math.log(self.region_width)
        return likelihood_position


class FragmentSizeDistribution(torch.nn.Module):
    def log_prob(self, data):
        bincount = torch.bincount(data.motifs.local_gene_ix, minlength=data.minibatch.n_genes) / self.window_width
        logits_count_genes = self.nn_logit(bincount[:, None].float()).reshape((data.minibatch.n_genes,))
        count = torch.bincount(
            data.fragments.local_cellxgene_ix, minlength=data.minibatch.n_cells * data.minibatch.n_genes
        ).reshape((data.minibatch.n_cells, data.minibatch.n_genes))
        likelihood_count = torch.distributions.Geometric(logits=logits_count_genes).log_prob(count)
        return likelihood_count


class FragmentSizeDistributionBaseline(FragmentPositionDistribution):
    def __init__(self, fragments, motifscan, nbins=(10, 20, 50)):
        super().__init__()
        self.region_width = fragments.regions.window[1] - fragments.regions.window[0]

    def log_prob(self, data):
        likelihood_position = torch.zeros(
            data.fragments.n_fragments, device=data.fragments.coordinates.device
        ) - math.log(self.region_width)
        return likelihood_position

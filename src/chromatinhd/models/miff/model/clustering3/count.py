import torch
import numpy as np
import math
from typing import Union
from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.data.fragments import Fragments, FragmentsView


class FragmentCountDistribution(torch.nn.Module):
    pass


class Baseline(FragmentCountDistribution):
    def __init__(self, fragments, motifcounts, clustering, baseline=None):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.zeros((1,)))
        if baseline is not None:
            self.baseline = baseline
        else:
            self.baseline = EmbeddingTensor(fragments.n_regions, (1,), sparse=True)
            init = torch.from_numpy(fragments.regionxcell_counts.sum(1).astype(np.float) / fragments.n_cells)
            init = torch.log(init)
            self.baseline.weight.data[:] = init.unsqueeze(-1)

        lib = torch.from_numpy(fragments.regionxcell_counts.sum(0).astype(np.float) / fragments.n_regions)
        lib = torch.log(lib)
        self.register_buffer("lib", lib)

    def log_prob(self, data):
        count = torch.bincount(
            data.fragments.local_cellxregion_ix, minlength=data.minibatch.n_cells * data.minibatch.n_regions
        ).reshape((data.minibatch.n_cells, data.minibatch.n_regions))
        logits = (
            self.baseline(data.minibatch.regions_oi_torch).squeeze(1).unsqueeze(0)
            + self.lib.unsqueeze(1)[data.minibatch.cells_oi]
        )
        likelihood_count = torch.distributions.Poisson(rate=torch.exp(logits)).log_prob(count)
        return likelihood_count

    def parameters_sparse(self):
        yield self.baseline.weight


class FragmentCountDistribution1(FragmentCountDistribution):
    def __init__(self, fragments, motifcounts, clustering, baseline=None):
        super().__init__()

        if baseline is not None:
            self.baseline = baseline
        else:
            self.baseline = EmbeddingTensor(fragments.n_regions, (1,), sparse=True)

        n_final_channels = 10

        self.nn_logit = torch.nn.Sequential(
            torch.nn.Conv1d(1, n_final_channels, kernel_size=3, padding="same"),
            torch.nn.MaxPool1d(kernel_size=motifcounts.motifcount_sizes[0], stride=motifcounts.motifcount_sizes[0]),
            torch.nn.Flatten(),
        )
        self.window_width = fragments.regions.window[1] - fragments.regions.window[0]
        self.weight = torch.nn.Embedding(clustering.n_clusters, n_final_channels, sparse=True)
        self.weight.weight.data.zero_()

        self.motif_binsizes = motifcounts.motif_binsizes

        lib = torch.from_numpy(fragments.regionxcell_counts.sum(0).astype(np.float) / fragments.n_regions)
        lib = torch.log(lib)
        self.register_buffer("lib", lib)

    def log_prob(self, data):
        motif_binsize = self.motif_binsizes[0]
        motif_count_regions = data.motifcounts.regioncounts / motif_binsize * 10

        nn_outcome = self.nn_logit(motif_count_regions.unsqueeze(1)).squeeze(1)
        logits = (
            torch.einsum("ab,cb->ac", self.weight(data.clustering.labels), nn_outcome)
            + self.baseline(data.minibatch.regions_oi_torch).squeeze(1).unsqueeze(0)
            + self.lib.unsqueeze(1)[data.minibatch.cells_oi]
        )

        count = torch.bincount(
            data.fragments.local_cellxregion_ix, minlength=data.minibatch.n_cells * data.minibatch.n_regions
        ).reshape((data.minibatch.n_cells, data.minibatch.n_regions))
        likelihood_count = torch.distributions.Poisson(rate=torch.exp(logits)).log_prob(count)
        return likelihood_count

    def parameters_sparse(self):
        yield self.baseline.weight
        yield self.weight.weight

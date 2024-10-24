import torch
import numpy as np
import math
from typing import Union
from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.data.fragments import Fragments, FragmentsView


class FragmentCountDistribution(torch.nn.Module):
    pass


def count_fragments(data):
    count = torch.bincount(
        data.fragments.local_cellxregion_ix, minlength=data.minibatch.n_cells * data.minibatch.n_regions
    ).reshape((data.minibatch.n_cells, data.minibatch.n_regions))
    return count


class Baseline(FragmentCountDistribution):
    def __init__(self, fragments, clustering, baseline=None):
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
        count = count_fragments(data)

        if data.fragments.lib is not None:
            lib = data.fragments.lib
        else:
            lib = self.lib[data.minibatch.cells_oi]
        logits = self.baseline(data.minibatch.regions_oi_torch).squeeze(1).unsqueeze(0) + lib.unsqueeze(1)
        likelihood_count = torch.distributions.Poisson(rate=torch.exp(logits)).log_prob(count)
        return likelihood_count

    def parameters_sparse(self):
        yield self.baseline.weight


class FragmentCountDistribution1(FragmentCountDistribution):
    def __init__(self, fragments, clustering, baseline=None, delta_logit=None):
        super().__init__()
        if baseline is not None:
            self.baseline = baseline
        else:
            self.baseline = EmbeddingTensor(fragments.n_regions, (1,), sparse=True)
            init = torch.from_numpy(fragments.regionxcell_counts.sum(1).astype(np.float) / fragments.n_cells)
            init = torch.log(init)
            self.baseline.weight.data[:] = init.unsqueeze(-1)

        if delta_logit is not None:
            self.delta_logit = delta_logit
        else:
            self.delta_logit = EmbeddingTensor(fragments.n_regions, (clustering.n_clusters,), sparse=True)
            self.delta_logit.weight.data[:] = 0

        lib = torch.from_numpy(fragments.regionxcell_counts.sum(0).astype(np.float) / fragments.n_regions)
        lib = torch.log(lib)
        self.register_buffer("lib", lib)

    def log_prob(self, data):
        count = count_fragments(data)

        if data.fragments.lib is not None:
            lib = data.fragments.lib
        else:
            lib = self.lib[data.minibatch.cells_oi]
        logits = (
            +self.baseline(data.minibatch.regions_oi_torch).squeeze(1).unsqueeze(0)
            + lib.unsqueeze(1)
            + self.delta_logit(data.minibatch.regions_oi_torch)[:, data.clustering.labels].transpose(0, 1)
        )

        count = torch.bincount(
            data.fragments.local_cellxregion_ix, minlength=data.minibatch.n_cells * data.minibatch.n_regions
        ).reshape((data.minibatch.n_cells, data.minibatch.n_regions))
        likelihood_count = torch.distributions.Poisson(rate=torch.exp(logits)).log_prob(count)
        return likelihood_count

    def parameters_sparse(self):
        yield self.baseline.weight
        yield self.delta_logit.weight

    @classmethod
    def from_baseline(cls, fragments, clustering, count_reference):
        baseline = torch.nn.Embedding.from_pretrained(count_reference.baseline.weight.data)
        delta_logit = EmbeddingTensor.from_pretrained(count_reference.delta_logit)

        return cls(fragments=fragments, clustering=clustering, delta_logit=delta_logit, baseline=baseline)

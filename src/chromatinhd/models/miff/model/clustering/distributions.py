import torch
import numpy as np
import math
from chromatinhd.embedding import EmbeddingTensor


class WindowStack(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def log_prob(self, bin_ixs, unnormalized_heights, inverse=False):
        logprob = torch.zeros(bin_ixs.shape[0], device=bin_ixs.device)

        for i, (unnormalized_heights_scale,) in enumerate(
            zip(
                unnormalized_heights,
            )
        ):
            heights_scale = torch.nn.functional.log_softmax(unnormalized_heights_scale, 1) + math.log(
                unnormalized_heights_scale.shape[1]
            )
            logprob += heights_scale.gather(1, bin_ixs[:, i].unsqueeze(1)).squeeze(1)

        return logprob


class FragmentCountDistribution(torch.nn.Module):
    pass


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


class FragmentCountDistributionBaseline(FragmentCountDistribution):
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


class FragmentPositionDistribution(torch.nn.Module):
    pass


class Identity(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class FragmentPositionDistribution1(FragmentPositionDistribution):
    def __init__(
        self,
        fragments,
        motifcounts,
        clustering,
        kernel_size=5,
        final_channels=10,
        n_hidden_dimensions=10,
        n_motifs=1,
        baselines=None,
    ):
        super().__init__()
        self.window = fragments.regions.window
        self.window_width = fragments.regions.window[1] - fragments.regions.window[0]
        self.position_dist = WindowStack()
        self.motif_binsizes = motifcounts.motif_binsizes

        self.predictors = []
        self.predictor2s = []
        for i, (fragmentprob_size, motifcount_size) in enumerate(
            zip(motifcounts.fragmentprob_sizes, motifcounts.motifcount_sizes)
        ):
            if not motifcount_size % fragmentprob_size == 0:
                raise ValueError("motifcount_size must be a multiple of fragmentprob_size")
            int(motifcount_size // fragmentprob_size)
            predictor = torch.nn.Sequential(
                # torch.nn.Linear(n_motifs, final_channels),
                torch.nn.Conv1d(n_motifs, final_channels // 2, kernel_size=kernel_size, padding="same"),
                torch.nn.Conv1d(final_channels // 2, final_channels, kernel_size=kernel_size, padding="same"),
                # torch.nn.AvgPool1d(kernel_size=stride, stride=stride),
            )
            self.final_channels = final_channels
            setattr(self, f"predictor_{i}", predictor)
            self.predictors.append(predictor)

            predictor2 = torch.nn.Sequential(
                # Identity(),
                torch.nn.Linear(final_channels, n_hidden_dimensions),
                # torch.nn.Dropout(0.1),
                torch.nn.ReLU(),
                torch.nn.Linear(n_hidden_dimensions, n_hidden_dimensions),
                torch.nn.ReLU(),
            )
            setattr(self, f"predictor2_{i}", predictor2)
            self.predictor2s.append(predictor2)

        # create/check baselines and differential
        baselines_pretrained = baselines if baselines is not None else None

        baselines = []
        differentials = []
        for i, (parent_fragment_width, fragmentprob_size) in enumerate(
            zip([1, *motifcounts.fragment_widths[:-1]], motifcounts.fragmentprob_sizes)
        ):
            if baselines_pretrained is not None:
                baseline = baselines_pretrained[i]
            else:
                baseline = torch.nn.Embedding(
                    fragments.n_regions * parent_fragment_width,
                    fragmentprob_size,
                    sparse=True,
                )
                baseline.weight.data[:] = 0.0

            baselines.append(baseline)
            setattr(self, f"baseline_{i}", baseline)

            differential = EmbeddingTensor(
                clustering.n_clusters,
                (n_hidden_dimensions,),
            )
            differential.weight.data[:] = 0.0
            differentials.append(differential)
            setattr(self, f"differential_{i}", differential)

        self.baselines = baselines
        self.differentials = differentials
        self.n_clusters = clustering.n_clusters

    def log_prob(self, data):
        predictor = self.predictors[0]
        predictor2 = self.predictor2s[0]
        bincounts = data.motifcounts.regioncounts
        self.motif_binsizes[0]
        baseline = self.baselines[0]
        differential = self.differentials[0]

        # bincounts_input = (bincounts.float() / motif_binsize * 100)
        bincounts_input = bincounts.float().unsqueeze(1)

        # output: [fragment, channels, fragment_bin]
        # motifcount_embedding = bincounts_input.float().unsqueeze(1)
        # motifcount_embedding = (bincounts > 1).float().unsqueeze(1)
        motifcount_embedding = predictor(bincounts_input)  # unsqueeze to add a single channel
        fragmentprob_size = motifcount_embedding.shape[-1]

        # transpose to [fragmentxfragment_bin, channel]
        motifcount_embedding = motifcount_embedding.transpose(1, 2).reshape(-1, self.final_channels)

        # output: [fragmentxfragment_bin, 1] -> [fragment, fragment_bin, hidden_dimension]
        region_embedding = predictor2(motifcount_embedding).reshape((bincounts.shape[0], fragmentprob_size, -1))

        baseline_unnormalized_height = baseline(data.minibatch.regions_oi_torch)

        unnormalized_heights_scale = (
            torch.einsum("bcd,ad->abc", region_embedding, differential.get_full_weight()) + baseline_unnormalized_height
        )
        heights_scale = torch.nn.functional.log_softmax(unnormalized_heights_scale, -1) + math.log(
            unnormalized_heights_scale.shape[1]
        )

        likelihood_position = heights_scale[
            data.clustering.labels[data.fragments.local_cell_ix],
            data.fragments.local_region_ix,
            data.motifcounts.binixs[:, 0],
        ]
        return likelihood_position


class FragmentPositionDistribution2(FragmentPositionDistribution):
    def __init__(
        self,
        fragments,
        motifcounts,
        clustering,
        kernel_size=1,
        final_channels=1,
        n_hidden_dimensions=1,
        n_motifs=1,
        baselines=None,
    ):
        super().__init__()
        self.window = fragments.regions.window
        self.window_width = fragments.regions.window[1] - fragments.regions.window[0]
        self.position_dist = WindowStack()
        self.motif_binsizes = motifcounts.motif_binsizes

        self.predictors = []
        self.predictor2s = []
        for i, (fragmentprob_size, motifcount_size) in enumerate(
            zip(motifcounts.fragmentprob_sizes, motifcounts.motifcount_sizes)
        ):
            if not motifcount_size % fragmentprob_size == 0:
                raise ValueError("motifcount_size must be a multiple of fragmentprob_size")
            int(motifcount_size // fragmentprob_size)
            predictor = torch.nn.Sequential(
                # torch.nn.Linear(n_motifs, final_channels)
                torch.nn.Conv1d(n_motifs, final_channels, kernel_size=kernel_size, padding="same"),
                # torch.nn.AvgPool1d(kernel_size=stride, stride=stride),
            )
            self.final_channels = final_channels
            setattr(self, f"predictor_{i}", predictor)
            self.predictors.append(predictor)

            predictor2 = torch.nn.Sequential(
                Identity(),
                # torch.nn.Linear(final_channels, n_hidden_dimensions),
                # torch.nn.Dropout(0.1),
                # torch.nn.Sigmoid(),
                # torch.nn.Linear(n_hidden_dimensions, 1), # this is done in a head-specific way
            )
            setattr(self, f"predictor2_{i}", predictor2)
            self.predictor2s.append(predictor2)

        # create/check baselines and differential
        baselines_pretrained = baselines if baselines is not None else None

        baselines = []
        differentials = []
        for i, (parent_fragment_width, fragmentprob_size) in enumerate(
            zip([1, *motifcounts.fragment_widths[:-1]], motifcounts.fragmentprob_sizes)
        ):
            if baselines_pretrained is not None:
                baseline = baselines_pretrained[i]
            else:
                baseline = torch.nn.Embedding(
                    fragments.n_regions * parent_fragment_width,
                    fragmentprob_size,
                    sparse=True,
                )
                baseline.weight.data[:] = 0.0

            baselines.append(baseline)
            setattr(self, f"baseline_{i}", baseline)

            differential = EmbeddingTensor(
                clustering.n_clusters,
                (n_hidden_dimensions, 1),
            )
            differential.weight.data[:] = 0.0
            differentials.append(differential)
            setattr(self, f"differential_{i}", differential)

        self.baselines = baselines
        self.differentials = differentials
        self.n_clusters = clustering.n_clusters

    def log_prob(self, data):
        unnormalized_heights = []
        for i, (predictor, predictor2, bincounts, motif_binsize, baseline, differential) in enumerate(
            zip(
                self.predictors,
                self.predictor2s,
                data.motifcounts.bincounts,
                self.motif_binsizes,
                self.baselines,
                self.differentials,
            )
        ):
            bincounts_input = (bincounts.float() / motif_binsize * 100).unsqueeze(
                1
            )  # unsqueeze to add a single channel

            # output: [fragment, channels, fragment_bin]
            motifcount_embedding = bincounts_input.float().unsqueeze(1)
            # motifcount_embedding = bincounts.float().unsqueeze(1)
            motifcount_embedding = (bincounts > 1).float().unsqueeze(1)
            # motifcount_embedding = predictor(bincounts_input)
            fragmentprob_size = motifcount_embedding.shape[-1]

            # transpose to [fragmentxfragment_bin, channel]
            motifcount_embedding = motifcount_embedding.transpose(1, 2).reshape(-1, self.final_channels)

            # output: [fragmentxfragment_bin, 1] -> [fragment, fragment_bin, hidden_dimension]
            # fragment_embedding = bincounts.float().reshape((bincounts.shape[0], fragmentprob_size, -1))
            fragment_embedding = predictor2(motifcount_embedding).reshape((bincounts.shape[0], fragmentprob_size, -1))

            baseline_unnormalized_height = baseline(data.motifcounts.global_binixs[:, i])
            # differential_weights = differential(
            #     data.motifcounts.global_binixs[:, i] * self.n_clusters
            #     + data.clustering.labels[data.fragments.local_cell_ix]
            # )
            differential_weights = differential(data.clustering.labels[data.fragments.local_cell_ix])

            # fragment_embedding = fragment_embedding + baseline_unnormalized_height

            unnormalized_height = (
                torch.matmul(fragment_embedding, differential_weights).squeeze(-1) + baseline_unnormalized_height
            )
            unnormalized_heights.append(unnormalized_height)

        # p(pos|region,motifs_region)
        likelihood_position = self.position_dist.log_prob(
            bin_ixs=data.motifcounts.binixs,
            unnormalized_heights=unnormalized_heights,
        )
        return likelihood_position


class FragmentPositionDistributionBaseline(FragmentPositionDistribution):
    def __init__(self, fragments, motifcounts, clustering, baselines=None):
        super().__init__()

        baselines_pretrained = baselines if baselines is not None else None

        baselines = []
        for i, (parent_fragment_width, fragmentprob_size) in enumerate(
            zip([1, *motifcounts.fragment_widths[:-1]], motifcounts.fragmentprob_sizes)
        ):
            if baselines_pretrained is not None:
                baseline = baselines_pretrained[i]
            else:
                baseline = torch.nn.Embedding(
                    fragments.n_regions * parent_fragment_width,
                    fragmentprob_size,
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

            baselines.append(baseline)
            setattr(self, f"baseline_{i}", baseline)

        self.baselines = baselines
        self.position_dist = WindowStack()

    def log_prob(self, data):
        unnormalized_heights = []

        for i, (baseline,) in enumerate(zip(self.baselines)):
            # global_binixs contains in which bin, including the region ix, each fragment sits
            baseline_unnormalized_height = baseline(data.motifcounts.global_binixs[:, i])
            unnormalized_heights.append(baseline_unnormalized_height)

        # p(pos|region,motifs_region)
        likelihood_position = self.position_dist.log_prob(
            bin_ixs=data.motifcounts.binixs,
            unnormalized_heights=unnormalized_heights,
        )
        return likelihood_position

    def parameters_sparse(self):
        for baseline in self.baselines:
            yield baseline.weight

import torch
import math


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
            # print(heights_scale)
            logprob += heights_scale.gather(1, bin_ixs[:, i].unsqueeze(1)).squeeze(1)

        return logprob


class FragmentCountDistribution(torch.nn.Module):
    def log_prob(self, data):
        motif_binsize = data.motifcounts.motif_binsizes[0]
        motif_count_genes = data.motifcounts.genecounts / motif_binsize * 10
        logits_genes = self.nn_logit(motif_count_genes.unsqueeze(1)).squeeze(1)
        fragment_count_cellxgenes = torch.bincount(
            data.fragments.local_cellxgene_ix, minlength=data.minibatch.n_cells * data.minibatch.n_genes
        ).reshape((data.minibatch.n_cells, data.minibatch.n_genes))
        likelihood_count = torch.distributions.Geometric(logits=logits_genes).log_prob(fragment_count_cellxgenes)
        return likelihood_count


class FragmentCountDistribution1(FragmentCountDistribution):
    def __init__(self, fragments, motifcounts):
        super().__init__()
        self.nn_logit = torch.nn.Sequential(
            torch.nn.Conv1d(1, 10, kernel_size=3, padding="same"),
            torch.nn.MaxPool1d(kernel_size=motifcounts.motifcount_sizes[0], stride=motifcounts.motifcount_sizes[0]),
            torch.nn.Flatten(),
            torch.nn.Linear(10, 1),
        )
        self.window_width = fragments.regions.window[1] - fragments.regions.window[0]


class FragmentCountDistributionBaseline(FragmentCountDistribution):
    def __init__(
        self,
        fragments,
        motifcounts,
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
    pass


class FragmentPositionDistribution1(FragmentPositionDistribution):
    def __init__(self, fragments, motifcounts, kernel_size=5, final_channels=10, n_hidden_dimensions=10, n_motifs=1):
        super().__init__()
        self.window = fragments.regions.window
        self.window_width = fragments.regions.window[1] - fragments.regions.window[0]
        self.position_dist = WindowStack()

        self.predictors = []
        self.predictor2s = []
        for i, (fragmentprob_size, motifcount_size) in enumerate(
            zip(motifcounts.fragmentprob_sizes, motifcounts.motifcount_sizes)
        ):
            if not motifcount_size % fragmentprob_size == 0:
                raise ValueError("motifcount_size must be a multiple of fragmentprob_size")
            stride = int(motifcount_size // fragmentprob_size)
            predictor = torch.nn.Sequential(
                torch.nn.Conv1d(n_motifs, final_channels, kernel_size=kernel_size, padding="same"),
                torch.nn.AvgPool1d(kernel_size=stride, stride=stride),
            )
            self.final_channels = final_channels
            setattr(self, f"predictor_{i}", predictor)
            self.predictors.append(predictor)

            predictor2 = torch.nn.Sequential(
                torch.nn.Linear(final_channels, n_hidden_dimensions),
                torch.nn.Dropout(0.1),
                torch.nn.GELU(),
                torch.nn.Linear(n_hidden_dimensions, 1),
            )
            setattr(self, f"predictor2_{i}", predictor2)
            self.predictor2s.append(predictor2)

    def log_prob(self, data):
        unnormalized_heights = []
        for predictor, predictor2, bincounts, motif_binsize in zip(
            self.predictors,
            self.predictor2s,
            data.motifcounts.bincounts,
            data.motifcounts.motif_binsizes,
        ):
            bincounts_input = (bincounts.float() / motif_binsize * 100).unsqueeze(
                1
            )  # unsqueeze to add a single channel
            # bincounts_input = (bincounts.float()).unsqueeze(1)  # unsqueeze to add a single channel

            # output: [fragment, channels, fragment_bin]
            unnormalized_heights_bin = predictor(bincounts_input)
            n_fragment_bins = unnormalized_heights_bin.shape[-1]

            # transpose to [fragmentxfragment_bin, channel]
            unnormalized_heights_bin = unnormalized_heights_bin.transpose(1, 2).reshape(-1, self.final_channels)

            # output: [fragmentxfragment_bin, 1] -> [fragment, fragment_bin]
            unnormalized_heights_bin = predictor2(unnormalized_heights_bin).reshape(
                (bincounts.shape[0], n_fragment_bins)
            )
            unnormalized_heights.append(unnormalized_heights_bin)

        # p(pos|gene,motifs_gene)
        likelihood_position = self.position_dist.log_prob(
            bin_ixs=data.motifcounts.binixs,
            unnormalized_heights=unnormalized_heights,
        )
        return likelihood_position


class FragmentPositionDistributionBaseline(FragmentPositionDistribution):
    def __init__(self, fragments, motifcounts):
        super().__init__()
        self.window = fragments.regions.window
        self.window_width = fragments.regions.window[1] - fragments.regions.window[0]
        self.param = torch.nn.Parameter(torch.zeros((1,)))

    def log_prob(self, data):
        return torch.zeros((data.fragments.n_fragments,), device=data.fragments.coordinates.device) + self.param * 0.0


class FragmentPositionDistribution2(FragmentPositionDistribution):
    """
    This one doesn't pool information across bins avoiding any convolutions
    It therefore has to assume that bin size of fragments and motifs is the same
    """

    def __init__(self, fragments, motifcounts, n_hidden_dimensions=10, n_motifs=1):
        super().__init__()
        self.window = fragments.regions.window
        self.window_width = fragments.regions.window[1] - fragments.regions.window[0]
        self.position_dist = WindowStack()

        self.predictors = []
        self.predictor2s = []
        for i, (fragmentprob_size, motifcount_size) in enumerate(
            zip(motifcounts.fragmentprob_sizes, motifcounts.motifcount_sizes)
        ):
            assert (
                fragmentprob_size == motifcount_size
            ), "This assumes that the bin size of fragments and motifs is the same"
            predictor2 = torch.nn.Sequential(
                torch.nn.Linear(1, n_hidden_dimensions),
                torch.nn.Dropout(0.1),
                torch.nn.ReLU(),
                torch.nn.Linear(n_hidden_dimensions, 1),
            )
            setattr(self, f"predictor2_{i}", predictor2)
            self.predictor2s.append(predictor2)

    def log_prob(self, data):
        unnormalized_heights = []
        for predictor2, bincounts, motif_binsize in zip(
            self.predictor2s,
            data.motifcounts.bincounts,
            data.motifcounts.motif_binsizes,
        ):
            bincounts_input = bincounts.float() / motif_binsize * 100

            # output: [fragmentxfragment_bin, 1] -> [fragment, fragment_bin]
            unnormalized_heights_bin = predictor2(bincounts_input.flatten().unsqueeze(1)).reshape((bincounts.shape))
            unnormalized_heights.append(unnormalized_heights_bin)

        # p(pos|gene,motifs_gene)
        likelihood_position = self.position_dist.log_prob(
            bin_ixs=data.motifcounts.binixs,
            unnormalized_heights=unnormalized_heights,
        )
        return likelihood_position

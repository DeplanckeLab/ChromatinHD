import torch
import math
from chromatinhd.embedding import EmbeddingTensor


class FragmentPositionDistribution(torch.nn.Module):
    pass


class FragmentPositionDistributionBaseline(FragmentPositionDistribution):
    def __init__(self, fragments, motifcounts, clustering, baselines=None, cluster_modifier=None):
        super().__init__()

        baselines_pretrained = baselines if baselines is not None else None

        self.register_buffer(
            "lib", torch.bincount(fragments.mapping[:, 0], minlength=fragments.n_cells).float() / fragments.n_genes
        )

        baselines = []
        for i, (parent_fragment_width, fragmentprob_size) in enumerate(
            zip([1, *motifcounts.fragment_widths[:-1]], motifcounts.fragmentprob_sizes)
        ):
            if baselines_pretrained is not None:
                baseline = baselines_pretrained[i]
            else:
                binwidth = fragments.regions.region_width / fragmentprob_size
                binixs = (fragments.coordinates[:, 0].numpy() - fragments.regions.window[0]) // binwidth
                cellxgene_ix = fragments.mapping[:, 0] * fragments.n_genes + fragments.mapping[:, 1]
                count = torch.bincount(
                    (cellxgene_ix * fragmentprob_size + binixs).int(),
                    minlength=fragmentprob_size * (fragments.n_cells) * (fragments.n_genes),
                ).reshape((fragments.n_cells), (fragments.n_genes), fragmentprob_size)
                init_baseline = torch.log(count.float().mean(0) + 1e-5)

                baseline = torch.nn.Embedding(
                    fragments.n_genes * parent_fragment_width,
                    fragmentprob_size,
                    sparse=True,
                )
                baseline.weight.data = init_baseline

            baselines.append(baseline)
            setattr(self, f"baseline_{i}", baseline)

        self.baselines = baselines

        if cluster_modifier is None:
            self.cluster_modifier = torch.nn.Parameter(torch.zeros(clustering.n_clusters))
        else:
            self.register_buffer("cluster_modifier", cluster_modifier)

    def log_prob(self, data):
        unnormalized_heights_scale = (
            self.baselines[0](data.minibatch.genes_oi_torch) + self.cluster_modifier[data.clustering.labels, None, None]
        )
        n_bins = unnormalized_heights_scale.shape[-1]
        count = torch.bincount(
            data.fragments.local_cellxgene_ix * n_bins + data.motifcounts.binixs[:, 0],
            minlength=n_bins * len(data.minibatch.cells_oi) * len(data.minibatch.genes_oi),
        ).reshape(len(data.minibatch.cells_oi), len(data.minibatch.genes_oi), n_bins)

        dist = torch.distributions.Poisson(
            torch.exp(unnormalized_heights_scale)  # * self.lib[data.minibatch.cells_oi_torch, None]
        )
        likelihood_position = dist.log_prob(count).sum(-1)

        return likelihood_position

    def parameters_sparse(self):
        for baseline in self.baselines:
            yield baseline.weight


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
        kernel_size=1,
        final_channels=1,
        n_hidden_dimensions=1,
        n_motifs=1,
        baselines=None,
        cluster_modifier=None,
    ):
        super().__init__()
        self.window = fragments.regions.window
        self.window_width = fragments.regions.window[1] - fragments.regions.window[0]
        self.motif_binsizes = motifcounts.motif_binsizes

        self.register_buffer(
            "lib", torch.bincount(fragments.mapping[:, 0], minlength=fragments.n_cells).float() / fragments.n_genes
        )

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
                # torch.nn.ReLU(),
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
                    fragments.n_genes * parent_fragment_width,
                    fragmentprob_size,
                    sparse=True,
                )

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

        if cluster_modifier is None:
            self.cluster_modifier = torch.nn.Parameter(torch.zeros(clustering.n_clusters))
        else:
            self.register_buffer("cluster_modifier", cluster_modifier)

    def log_prob(self, data):
        self.predictors[0]
        predictor2 = self.predictor2s[0]
        bincounts = data.motifcounts.genecounts
        motif_binsize = self.motif_binsizes[0]
        baseline = self.baselines[0]
        differential = self.differentials[0]

        (bincounts.float() / motif_binsize * 100).unsqueeze(1)  # unsqueeze to add a single channel
        # import matplotlib.pyplot as plt
        # print(bincounts_input)

        # plt.plot(bincounts_input[0, 0, :].numpy())

        # output: [fragment, channels, fragment_bin]
        # motifcount_embedding = bincounts_input.float().unsqueeze(1)
        motifcount_embedding = bincounts.float().unsqueeze(1)
        # motifcount_embedding = (bincounts > 1).float().unsqueeze(1)
        # motifcount_embedding = predictor(bincounts_input)
        fragmentprob_size = motifcount_embedding.shape[-1]

        # transpose to [fragmentxfragment_bin, channel]
        motifcount_embedding = motifcount_embedding.transpose(1, 2).reshape(-1, self.final_channels)

        # output: [fragmentxfragment_bin, 1] -> [fragment, fragment_bin, hidden_dimension]
        # fragment_embedding = bincounts.float().reshape((bincounts.shape[0], fragmentprob_size, -1))
        gene_embedding = predictor2(motifcount_embedding).reshape((bincounts.shape[0], fragmentprob_size, -1))

        baseline_unnormalized_height = baseline(data.minibatch.genes_oi_torch)
        cell_embedding = differential(data.clustering.labels)

        unnormalized_heights_scale = (
            torch.matmul(gene_embedding.unsqueeze(0), cell_embedding.unsqueeze(1)).squeeze(-1)
            + baseline_unnormalized_height
        ) + self.cluster_modifier[data.clustering.labels, None, None]

        n_bins = unnormalized_heights_scale.shape[-1]

        count = torch.bincount(
            data.fragments.local_cellxgene_ix * n_bins + data.motifcounts.binixs[:, 0],
            minlength=n_bins * len(data.minibatch.cells_oi) * len(data.minibatch.genes_oi),
        ).reshape(len(data.minibatch.cells_oi), len(data.minibatch.genes_oi), n_bins)

        dist = torch.distributions.Poisson(
            torch.exp(unnormalized_heights_scale)  # * self.lib[data.minibatch.cells_oi_torch, None]
        )
        likelihood_position = dist.log_prob(count).sum(-1)

        return likelihood_position

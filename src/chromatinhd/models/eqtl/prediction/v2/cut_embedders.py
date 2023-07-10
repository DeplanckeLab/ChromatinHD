import torch
import torch_scatter


class PositionalEncoder(torch.nn.Module):
    def __init__(self, n_encoding_dimensions, window_size=1000):
        super().__init__()
        assert (n_encoding_dimensions % 2) == 0
        self.n_encoding_dimensions = n_encoding_dimensions
        self.freq = torch.pow(
            window_size,
            torch.arange(0, self.n_encoding_dimensions, 2, dtype=torch.float32)
            / self.n_encoding_dimensions,
        )
        self.register_buffer(
            "shifts",
            torch.tensor(
                [[0, torch.pi / 2] for _ in range(1, n_encoding_dimensions // 2 + 1)]
            ).flatten(-2),
        )
        self.register_buffer("freqs", torch.repeat_interleave(self.freq, 2))

    def forward(self, X):
        P = torch.sin(X / self.freqs + self.shifts)
        return P


class CutEmbedderPositional(torch.nn.Module):
    def __init__(self, window_size, positional_encoder=None):
        super().__init__()
        if positional_encoder is None:
            positional_encoder = PositionalEncoder(
                window_size=window_size, n_encoding_dimensions=20
            )
        self.positional_encoder = positional_encoder
        self.n_embedding_dimensions = positional_encoder.n_encoding_dimensions + 1

    def forward(self, x):
        z = torch.concat(
            [
                torch.ones_like(x, dtype=torch.float).unsqueeze(-1),
                self.positional_encoder(x.unsqueeze(-1)),
            ],
            -1,
        )
        return z


class CutEmbedderDummy(torch.nn.Module):
    n_embedding_dimensions = 1

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ones_like(x, dtype=torch.float).unsqueeze(-1)


class CutEmbedderBins(torch.nn.Module):
    def __init__(self, window_size, n_dimensions=10):
        super().__init__()
        self.register_buffer(
            "bins", torch.linspace(-window_size - 1, window_size + 1, n_dimensions)
        )
        self.n_embedding_dimensions = len(self.bins) - 1 + 1

    def forward(self, x):
        counts = torch.nn.functional.one_hot(
            torch.clamp(torch.searchsorted(self.bins, x) - 1, 0, len(self.bins) - 2),
            len(self.bins) - 1,
        )
        z = torch.concat(
            [
                torch.ones_like(x, dtype=torch.float).unsqueeze(-1),
                counts,
            ],
            -1,
        )
        return z


class VariantEmbedder(torch.nn.Module):
    def __init__(self, cluster_cut_lib):
        super().__init__()
        self.register_buffer(
            "cluster_cut_lib", cluster_cut_lib.to(torch.float) / 10**6
        )

    def forward(
        self,
        cut_embedding,
        local_clusterxvariant_indptr,
        n_variants,
        n_clusters,
    ):
        # variant_embedding [clusters, variants, components]
        variant_embedding = (
            torch_scatter.segment_sum_csr(cut_embedding, local_clusterxvariant_indptr)
            .reshape((n_clusters, n_variants, cut_embedding.shape[-1]))
            .to(torch.float)
        )

        # cluster_cut_lib [cluster] -> [cluster, variant, component]
        variant_embedding = variant_embedding / self.cluster_cut_lib.unsqueeze(
            -1
        ).unsqueeze(-1)
        # variant_embedding = torch.log1p(variant_embedding) - 2  # only for dummy

        relative_variant_embedding = (
            variant_embedding - (variant_embedding.mean(0, keepdim=True))
        ) / (variant_embedding.std(0, keepdim=True) + 1e-5)

        variant_full_embedding = torch.concat(
            [
                variant_embedding,
                relative_variant_embedding,
            ],
            -1,
        )

        return variant_full_embedding

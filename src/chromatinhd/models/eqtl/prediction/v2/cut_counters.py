import torch
import torch_scatter


def indptr_to_indices(x):
    n = len(x) - 1
    return torch.repeat_interleave(torch.arange(n, device=x.device), torch.diff(x))


def calculate_1d_conv_output_length(
    length_in, kernel_size, stride=1, padding=0, dilation=1
):
    if isinstance(kernel_size, tuple):
        kernel_size = kernel_size[0]
    if isinstance(stride, tuple):
        stride = stride[0]
    if isinstance(padding, tuple):
        padding = padding[0]
    if isinstance(dilation, tuple):
        dilation = dilation[0]
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class VariantEmbedder(torch.nn.Module):
    def __init__(
        self,
        window_size,
        cluster_cut_lib,
        dummy=False,
        dumb=False,
        n_bins=None,
    ):
        super().__init__()
        self.register_buffer(
            "cluster_cut_lib", cluster_cut_lib.to(torch.float) / 10**6
        )

        if n_bins is None:
            n_bins = 200

        self.register_buffer(
            "bins", torch.linspace(-window_size - 1, window_size + 1, n_bins)
        )
        self.n_bins = len(self.bins) - 1

        kernel_size = 10
        final_n_dimensions = 10
        stride = 2
        self.nn = torch.nn.Sequential(
            torch.nn.Conv1d(1, 5, kernel_size, stride=stride),
            torch.nn.Conv1d(5, 5, kernel_size, 1),
            torch.nn.Conv1d(5, 5, kernel_size, 1),
            torch.nn.Conv1d(5, 5, kernel_size, 1),
            torch.nn.Conv1d(5, final_n_dimensions, kernel_size),
        )

        output_size = self.n_bins
        for layer in self.nn:
            output_size = calculate_1d_conv_output_length(
                output_size,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
            )
        assert output_size > 0
        self.n_embedding_dimensions = output_size * final_n_dimensions + 2

        self.dummy = dummy
        self.dumb = dumb

    def forward(
        self,
        relative_coordinates,
        local_clusterxvariant_indptr,
        n_variants,
        n_clusters,
    ):
        counts = torch.clamp(
            torch.searchsorted(self.bins, relative_coordinates) - 1,
            0,
            self.n_bins - 1,
        )

        local_clusterxvariant_indices = indptr_to_indices(local_clusterxvariant_indptr)

        clusterxvariant_bin_counts = torch.bincount(
            counts + local_clusterxvariant_indices * self.n_bins,
            minlength=n_variants * n_clusters * self.n_bins,
        ).reshape((n_clusters, n_variants, self.n_bins))

        # cluster_cut_lib [cluster] -> [cluster, variant, bin]
        clusterxvariant_bin_counts = (
            clusterxvariant_bin_counts
            / self.cluster_cut_lib.unsqueeze(-1).unsqueeze(-1)
        )
        x_flattened = clusterxvariant_bin_counts.flatten(0, 1).unsqueeze(-2)

        variant_embedding = self.nn(x_flattened).reshape(n_clusters, n_variants, -1)

        clusterxvariant_counts = torch.log1p(
            clusterxvariant_bin_counts.sum(-1, keepdim=True)
        )

        # relative_variant_embedding = variant_embedding - (
        #     variant_embedding.mean(0, keepdim=True)
        # ) / (variant_embedding.std(0, keepdim=True) + 1e-5)

        if self.dummy:
            variant_embedding[:] = 0.0

        variant_full_embedding = torch.concat(
            [
                variant_embedding,
                clusterxvariant_counts,
                clusterxvariant_counts - clusterxvariant_counts.mean(0, keepdim=True),
            ],
            -1,
        )

        if self.dumb:
            variant_full_embedding[:] = 0.0

        return variant_full_embedding


class VariantEmbedder(torch.nn.Module):
    def __init__(
        self,
        window_size,
        cluster_cut_lib,
        dummy=False,
        dumb=False,
        n_bins=None,
    ):
        super().__init__()
        self.register_buffer(
            "cluster_cut_lib", cluster_cut_lib.to(torch.float) / 10**6
        )

        if n_bins is None:
            n_bins = 5

        self.register_buffer(
            "bins", torch.linspace(-window_size - 1, window_size + 1, n_bins)
        )
        self.n_bins = len(self.bins) - 1

        self.n_embedding_dimensions = self.n_bins * 2 + 4

        self.distance_scaler = window_size

        self.dummy = dummy
        self.dumb = dumb

    def forward(
        self,
        relative_coordinates,
        local_clusterxvariant_indptr,
        n_variants,
        n_clusters,
    ):
        counts = torch.clamp(
            torch.searchsorted(self.bins, relative_coordinates) - 1,
            0,
            self.n_bins - 1,
        )

        local_clusterxvariant_indices = indptr_to_indices(local_clusterxvariant_indptr)

        clusterxvariant_bin_counts = torch.bincount(
            counts + local_clusterxvariant_indices * self.n_bins,
            minlength=n_variants * n_clusters * self.n_bins,
        ).reshape((n_clusters, n_variants, self.n_bins))

        # cluster_cut_lib [cluster] -> [cluster, variant, bin]
        clusterxvariant_bin_counts = (
            clusterxvariant_bin_counts
            / self.cluster_cut_lib.unsqueeze(-1).unsqueeze(-1)
        )

        clusterxvariant_counts = torch.log1p(
            clusterxvariant_bin_counts.sum(-1, keepdim=True)
        )

        mean_relative_coords = (
            torch_scatter.segment_mean_csr(
                torch.abs(relative_coordinates), local_clusterxvariant_indptr
            ).reshape((n_clusters, n_variants, 1))
            / self.distance_scaler
        )

        # relative_variant_embedding = variant_embedding - (
        #     variant_embedding.mean(0, keepdim=True)
        # ) / (variant_embedding.std(0, keepdim=True) + 1e-5)

        if self.dummy or self.dumb:
            clusterxvariant_bin_counts[:] = 0.0
        if self.dumb:
            mean_relative_coords[:] = 0.0

        variant_full_embedding = torch.concat(
            [
                clusterxvariant_bin_counts,
                clusterxvariant_bin_counts
                - clusterxvariant_bin_counts.mean(0, keepdim=True),
                clusterxvariant_counts,
                clusterxvariant_counts - clusterxvariant_counts.mean(0, keepdim=True),
                mean_relative_coords - mean_relative_coords.mean(0, keepdim=True),
                mean_relative_coords,
            ],
            -1,
        )

        # if self.dumb:
        #     variant_full_embedding[:] = 0.0

        return variant_full_embedding

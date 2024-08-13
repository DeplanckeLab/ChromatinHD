import torch
from chromatinhd.embedding import EmbeddingTensor
import math
import pandas as pd
import itertools


class Shared(torch.nn.Module):
    """
    Encoder which applies only at the end.
    """

    def __init__(
        self,
        fragments,
        clustering,
        binwidths=(5000, 1000, 500, 200, 100, 50, 25),
        delta_initialization="zeros",
        bias_initialization="zeros",
        delta_regularization=True,
        delta_p_scale_free=False,
        delta_p_scale_dist="normal",
        delta_p_dist="normal",
        delta_p_scale=1.5,
        differential=True,
        bias_regularization=False,
        bias_p_dist="normal",
        bias_p_scale=1.5,
    ):
        super().__init__()
        self.n_clusters = clustering.n_clusters
        self.differential = differential

        # create bin delta
        self.register_buffer("binwidths", torch.tensor(binwidths))

        self.window = fragments.regions.window
        self.total_width = self.window[1] - self.window[0]
        current_width = self.total_width

        self.n_final_bins = (self.total_width // self.binwidths[-1]).item()

        self._parameters_sparse = []

        self.nsections = []
        self.nbins = []
        for level, binwidth in enumerate(self.binwidths):
            nbin_section = current_width // binwidth
            nsection = self.total_width // (binwidth * nbin_section)
            w_i = EmbeddingTensor(
                fragments.n_regions,
                (
                    nsection,
                    nbin_section,
                ),
                sparse=True,
            )
            if bias_initialization == "zeros":
                w_i.data[:] = 0.0
            w_delta_i = EmbeddingTensor(fragments.n_regions, (self.n_clusters, nsection, nbin_section), sparse=True)
            if delta_initialization == "zeros":
                w_delta_i.data[:] = 0.0

            setattr(self, f"w_{level}", w_i)
            setattr(self, f"w_delta_{level}", w_delta_i)

            self._parameters_sparse.append(w_i.weight)
            self._parameters_sparse.append(w_delta_i.weight)

            current_width = binwidth

            self.nsections.append(nsection)
            self.nbins.append((nbin_section * nsection).item())

        # delta regularization
        self.delta_regularization = delta_regularization
        if self.delta_regularization:
            if delta_p_scale_free:
                self.delta_p_scale = torch.nn.Parameter(torch.tensor(math.log(delta_p_scale), requires_grad=True))
            else:
                self.register_buffer(
                    "delta_p_scale",
                    torch.tensor(math.log(delta_p_scale)),
                )
        self.delta_p_scale_dist = delta_p_scale_dist
        self.delta_p_dist = delta_p_dist

        # bias regularization
        self.bias_regularization = bias_regularization
        if self.bias_regularization:
            self.register_buffer("bias_p_scale", torch.tensor(math.log(bias_p_scale)))
        self.bias_p_dist = bias_p_dist

    def _get_w_delta_p(self):
        if self.delta_p_dist == "normal":
            w_delta_p = torch.distributions.Normal(0.0, torch.exp(self.delta_p_scale))
        elif self.delta_p_dist == "laplace":
            w_delta_p = torch.distributions.Laplace(0.0, torch.exp(self.delta_p_scale))
        else:
            raise ValueError("incorrect delta_p_dist")
        return w_delta_p

    def parameters_sparse(self):
        return self._parameters_sparse

    def _calculate_w(self, regions_oi):
        kl = torch.tensor(0.0, device=regions_oi.device)
        # bin
        w = torch.zeros((len(regions_oi), self.n_clusters, self.n_final_bins), device=regions_oi.device)
        for level in range(len(self.binwidths)):
            w_delta = getattr(self, f"w_delta_{level}")(regions_oi)
            if not self.differential:
                w_delta = w_delta * 0.0
            w_bias = getattr(self, f"w_{level}")(regions_oi)

            w_delta_multiplied = w_delta
            # if level == 1:
            # w_delta_multiplied = w_delta * w_delta_multiplier
            w_level = (w_bias.unsqueeze(1) + w_delta_multiplied).view(len(regions_oi), self.n_clusters, -1)

            reshape = self.n_final_bins // self.nbins[level]

            # this reshapes in such a way that the "bigger" bin is broadcasted towards all "smaller" bins
            w = (w.view(*w.shape[:-1], -1, reshape) + w_level.unsqueeze(-1)).view(*w.shape)

            if self.delta_regularization:
                w_delta_p = self._get_w_delta_p()
                w_delta_kl = w_delta_p.log_prob(w_delta)
                kl = kl + w_delta_kl.sum() * 1.0

            if self.bias_regularization:
                w_bias_p = torch.distributions.Normal(0.0, torch.exp(self.bias_p_scale))
                w_bias_kl = w_bias_p.log_prob(w_bias)
                kl = kl + w_bias_kl.sum() * 1.0

        w = torch.nn.functional.log_softmax(w, dim=-1) - torch.log(
            self.binwidths[-1]
        )  # to make it a pmf, divide by final bin width

        return w, kl

    def forward(self, data, w_delta_multiplier=1.0):
        w, w_kl = self._calculate_w(data.minibatch.regions_oi_torch)

        coords = torch.clamp(data.cuts.coordinates, self.window[0], self.window[1] - 1) - self.window[0]
        bin_ix = (coords // self.binwidths[-1]).long()

        likelihood = w[data.cuts.local_region_ix, data.clustering.indices[data.cuts.local_cell_ix], bin_ix]

        return likelihood, w_kl


class Split(torch.nn.Module):
    """
    Encoder which applies the softmax per level separately. Useful if we want to do a scalable autoregressive flow model in the future.
    """

    def __init__(
        self,
        fragments,
        clustering,
        binwidths=(5000, 1000, 500, 200, 100, 50, 25),
        delta_initialization="zeros",
        bias_initialization="zeros",
        delta_regularization=True,
        delta_p_scale_free=False,
        delta_p_scale_dist="normal",
        delta_p_dist="normal",
        delta_p_scale=1.5,
        differential=True,
        bias_regularization=False,
        bias_p_dist="normal",
        bias_p_scale=1.5,
    ):
        super().__init__()
        self.n_clusters = clustering.n_clusters
        self.differential = differential

        # create bin delta
        self.register_buffer("binwidths", torch.tensor(binwidths))

        self.window = fragments.regions.window
        self.total_width = self.window[1] - self.window[0]
        current_width = self.total_width

        self.n_final_bins = (self.total_width // self.binwidths[-1]).item()

        self._parameters_sparse = []

        self.nsections = []
        self.nbins = []
        for level, binwidth in enumerate(self.binwidths):
            nbin_section = current_width // binwidth
            nsection = self.total_width // (binwidth * nbin_section)
            w_i = EmbeddingTensor(
                fragments.n_regions,
                (
                    nsection,
                    nbin_section,
                ),
                sparse=True,
            )
            if bias_initialization == "zeros":
                w_i.data[:] = 0.0
            w_delta_i = EmbeddingTensor(fragments.n_regions, (self.n_clusters, nsection, nbin_section), sparse=True)
            if delta_initialization == "zeros":
                w_delta_i.data[:] = 0.0

            setattr(self, f"w_{level}", w_i)
            setattr(self, f"w_delta_{level}", w_delta_i)

            self._parameters_sparse.append(w_i.weight)
            self._parameters_sparse.append(w_delta_i.weight)

            current_width = binwidth

            self.nsections.append(nsection)
            self.nbins.append((nbin_section * nsection).item())

        # delta regularization
        self.delta_regularization = delta_regularization
        if self.delta_regularization:
            if delta_p_scale_free:
                self.delta_p_scale = torch.nn.Parameter(torch.tensor(math.log(delta_p_scale), requires_grad=True))
            else:
                self.register_buffer(
                    "delta_p_scale",
                    torch.tensor(math.log(delta_p_scale)),
                )
        self.delta_p_scale_dist = delta_p_scale_dist
        self.delta_p_dist = delta_p_dist

        # bias regularization
        self.bias_regularization = bias_regularization
        if self.bias_regularization:
            self.register_buffer("bias_p_scale", torch.tensor(math.log(bias_p_scale)))
        self.bias_p_dist = bias_p_dist

    def _get_w_delta_p(self):
        if self.delta_p_dist == "normal":
            w_delta_p = torch.distributions.Normal(0.0, torch.exp(self.delta_p_scale))
        elif self.delta_p_dist == "laplace":
            w_delta_p = torch.distributions.Laplace(0.0, torch.exp(self.delta_p_scale))
        else:
            raise ValueError("incorrect delta_p_dist")
        return w_delta_p

    def parameters_sparse(self):
        return self._parameters_sparse

    def forward(self, data, w_delta_multiplier=1.0):
        w, w_kl = self._calculate_w(data.minibatch.regions_oi_torch)

        coords = torch.clamp(data.cuts.coordinates, self.window[0], self.window[1] - 1) - self.window[0]
        bin_ix = (coords // self.binwidths[-1]).long()

        likelihood = w[data.cuts.local_region_ix, data.clustering.indices[data.cuts.local_cell_ix], bin_ix]

        return likelihood, w_kl

    def _calculate_w(
        self,
        regions_oi,
    ):
        kl = torch.tensor(0.0, device=regions_oi.device)
        w = torch.zeros((len(regions_oi), self.n_clusters, self.n_final_bins), device=regions_oi.device) - math.log(
            self.total_width
        )

        for level in range(len(self.binwidths)):
            w_delta = getattr(self, f"w_delta_{level}")(regions_oi)
            if not self.differential:
                w_delta = w_delta * 0.0
            w_bias = getattr(self, f"w_{level}")(regions_oi)

            w_delta_multiplied = w_delta
            w_level = w_bias.unsqueeze(1) + w_delta_multiplied
            # print(w_level.shape)
            w_level = torch.nn.functional.log_softmax(w_level, dim=-1) + math.log(w_level.shape[-1])
            w_level = w_level.view(len(regions_oi), self.n_clusters, -1)

            reshape = self.n_final_bins // self.nbins[level]

            # this reshapes in such a way that the "bigger" bin is broadcasted towards all "smaller" bins
            w = (w.view(*w.shape[:-1], -1, reshape) + w_level.unsqueeze(-1)).view(*w.shape)

            if self.delta_regularization:
                w_delta_p = self._get_w_delta_p()
                w_delta_kl = w_delta_p.log_prob(w_delta)
                kl = kl + w_delta_kl.sum() * 1.0

            if self.bias_regularization:
                w_bias_p = torch.distributions.Normal(0.0, torch.exp(self.bias_p_scale))
                w_bias_kl = w_bias_p.log_prob(w_bias)
                kl = kl + w_bias_kl.sum() * 1.0

        return w, kl


####


class MultiLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        bias: bool = True,
        device=None,
        dtype=None,
        weight_constructor=None,
        bias_constructor=None,
    ):
        super().__init__()

        self.out_features = out_features

        self._parameters_sparse = []

        if bias:
            if bias_constructor is None:

                def bias_constructor(shape):
                    stdv = 1.0 / math.sqrt(shape[-1])
                    return torch.empty(shape, device=device, dtype=dtype).uniform_(-stdv, stdv)

            bias = EmbeddingTensor(n_heads, (out_features,), constructor=bias_constructor, sparse=True)

            self.register_module("bias", bias)
            self._parameters_sparse.append(bias.weight)
        else:
            self.bias = None

        if weight_constructor is None:

            def weight_constructor(shape):
                stdv = 1.0 / math.sqrt(shape[-1])
                return torch.empty(shape, device=device, dtype=dtype).uniform_(-stdv, stdv)

            torch.nn.Linear

        self.weight = EmbeddingTensor(
            n_heads,
            (
                in_features,
                out_features,
            ),
            constructor=weight_constructor,
            sparse=True,
        )

        self._parameters_sparse.append(self.weight.weight)

    def forward(self, input: torch.Tensor, regions_oi):
        if self.bias is not None:
            output = torch.einsum("ab,cad->cbd", input, self.weight(regions_oi)) + self.bias(regions_oi)[:, None, :]
        else:
            output = torch.einsum("ab,cad->cbd", input, self.weight(regions_oi))
        return output

    def parameters_sparse(self):
        return self._parameters_sparse


class SharedLora(torch.nn.Module):
    def __init__(
        self,
        fragments,
        clustering,
        transcriptome,
        binwidths=(5000, 1000, 500, 200, 100, 50, 25),
        delta_initialization="zeros",
        bias_initialization="zeros",
        delta_regularization=True,
        delta_p_scale_free=False,
        delta_p_scale_dist="normal",
        delta_p_dist="normal",
        delta_p_scale=1.5,
        differential=True,
        bias_regularization=False,
        bias_p_dist="normal",
        bias_p_scale=1.5,
    ):
        super().__init__()
        self.n_clusters = clustering.n_clusters
        self.differential = differential

        # create bin delta
        self.register_buffer("binwidths", torch.tensor(binwidths))

        self.window = fragments.regions.window
        self.total_width = self.window[1] - self.window[0]
        current_width = self.total_width

        self.n_final_bins = (self.total_width // self.binwidths[-1]).item()

        self._parameters_sparse = []

        self.nsections = []
        self.nbins = []
        for level, binwidth in enumerate(self.binwidths):
            nbin_section = current_width // binwidth
            nsection = self.total_width // (binwidth * nbin_section)
            w_i = EmbeddingTensor(
                fragments.n_regions,
                (
                    nsection,
                    nbin_section,
                ),
                sparse=True,
            )
            if bias_initialization == "zeros":
                w_i.data[:] = 0.0

            setattr(self, f"w_{level}", w_i)
            self._parameters_sparse.append(w_i.weight)

            w_i_delta_function = MultiLinear(
                clustering.n_clusters,
                (nsection * nbin_section),
                fragments.n_regions,
                bias=False,
                weight_constructor=torch.zeros if delta_initialization == "zeros" else None,
            )

            setattr(self, f"w_delta_{level}", w_i_delta_function)

            current_width = binwidth

            self.nsections.append(nsection)
            self.nbins.append((nbin_section * nsection).item())

        # delta regularization
        self.delta_regularization = delta_regularization
        if self.delta_regularization:
            if delta_p_scale_free:
                self.delta_p_scale = torch.nn.Parameter(torch.tensor(math.log(delta_p_scale), requires_grad=True))
            else:
                self.register_buffer(
                    "delta_p_scale",
                    torch.tensor(math.log(delta_p_scale)),
                )
        self.delta_p_scale_dist = delta_p_scale_dist
        self.delta_p_dist = delta_p_dist

        # bias regularization
        self.bias_regularization = bias_regularization
        if self.bias_regularization:
            self.register_buffer("bias_p_scale", torch.tensor(math.log(bias_p_scale)))
        self.bias_p_dist = bias_p_dist

        # create pca input
        cluster_counts = (
            pd.DataFrame(transcriptome.X[:], index=transcriptome.obs.index).groupby(clustering.labels).mean()
        )
        import sklearn.decomposition

        pca = sklearn.decomposition.PCA(n_components=cluster_counts.shape[0], whiten=True)
        # pca.fit(cluster_counts.T)
        components = pca.fit_transform(cluster_counts).T
        self.register_buffer("components", torch.from_numpy(components))

    def _get_w_delta_p(self):
        if self.delta_p_dist == "normal":
            w_delta_p = torch.distributions.Normal(0.0, torch.exp(self.delta_p_scale))
        elif self.delta_p_dist == "laplace":
            w_delta_p = torch.distributions.Laplace(0.0, torch.exp(self.delta_p_scale))
        else:
            raise ValueError("incorrect delta_p_dist")
        return w_delta_p

    def parameters_sparse(self):
        return itertools.chain(
            self._parameters_sparse,
            *[getattr(self, f"w_delta_{level}").parameters_sparse() for level in range(len(self.binwidths))],
        )

    def _calculate_w(self, regions_oi):
        kl = torch.tensor(0.0, device=regions_oi.device)
        # bin
        w = torch.zeros((len(regions_oi), self.n_clusters, self.n_final_bins), device=regions_oi.device)
        for level in range(len(self.binwidths)):
            w_bias = getattr(self, f"w_{level}")(regions_oi)
            w_delta = getattr(self, f"w_delta_{level}")(self.components, regions_oi).reshape(
                len(regions_oi), -1, *w_bias.shape[1:]
            )
            if not self.differential:
                w_delta = w_delta * 0.0

            w_delta_multiplied = w_delta
            # if level == 1:
            # w_delta_multiplied = w_delta * w_delta_multiplier
            w_level = (w_bias.unsqueeze(1) + w_delta_multiplied).view(len(regions_oi), self.n_clusters, -1)

            reshape = self.n_final_bins // self.nbins[level]

            # this reshapes in such a way that the "bigger" bin is broadcasted towards all "smaller" bins
            w = (w.view(*w.shape[:-1], -1, reshape) + w_level.unsqueeze(-1)).view(*w.shape)

            if self.delta_regularization:
                w_delta_p = self._get_w_delta_p()
                w_delta_kl = w_delta_p.log_prob(w_delta)
                kl = kl + w_delta_kl.sum() * 1.0

            if self.bias_regularization:
                w_bias_p = torch.distributions.Normal(0.0, torch.exp(self.bias_p_scale))
                w_bias_kl = w_bias_p.log_prob(w_bias)
                kl = kl + w_bias_kl.sum() * 1.0

        w = torch.nn.functional.log_softmax(w, dim=-1) - torch.log(
            self.binwidths[-1]
        )  # to make it a pmf, divide by final bin width

        return w, kl

    def forward(self, data, w_delta_multiplier=1.0):
        w, w_kl = self._calculate_w(data.minibatch.regions_oi_torch)

        coords = torch.clamp(data.cuts.coordinates, self.window[0], self.window[1] - 1) - self.window[0]
        bin_ix = (coords // self.binwidths[-1]).long()

        likelihood = w[data.cuts.local_region_ix, data.clustering.indices[data.cuts.local_cell_ix], bin_ix]

        return likelihood, w_kl

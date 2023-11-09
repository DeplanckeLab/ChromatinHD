import torch
import numpy as np
from chromatinhd.embedding import FeatureParameter


class SineEncoding(torch.nn.Module):
    def __init__(self, n_frequencies):
        super().__init__()

        self.register_buffer(
            "frequencies",
            torch.tensor([[1 / 100 ** (2 * i / n_frequencies)] * 2 for i in range(1, n_frequencies + 1)]).flatten(-2),
        )

        self.register_buffer(
            "shifts",
            torch.tensor([[0, torch.pi / 2] for _ in range(1, n_frequencies + 1)]).flatten(-2),
        )

        self.n_embedding_dimensions = n_frequencies * 2 * 2

    def forward(self, coordinates):
        embedding = torch.sin((coordinates[..., None] * self.frequencies + self.shifts).flatten(-2))
        return embedding


class SineEncoding2(torch.nn.Module):
    def __init__(self, n_frequencies, rate=0.2):
        super().__init__()

        self.register_buffer(
            "frequencies",
            torch.tensor([np.pi / 2 ** (i * rate) for i in range(0, n_frequencies)]),
        )

        self.n_embedding_dimensions = n_frequencies * 2

    def forward(self, coordinates):
        embedding = torch.cos((coordinates[..., None] * self.frequencies).flatten(-2))
        return embedding


class SineEncoding3(torch.nn.Module):
    def __init__(self, n_frequencies, rate=0.2):
        super().__init__()

        self.register_buffer(
            "frequencies",
            torch.tensor([[np.pi / 2 ** (i * rate)] * 2 for i in range(0, n_frequencies)]).flatten(-2),
        )

        self.register_buffer(
            "shifts",
            torch.tensor([[0, torch.pi / 2] for _ in range(1, n_frequencies + 1)]).flatten(-2),
        )

        self.n_embedding_dimensions = n_frequencies * 2 * 2

    def forward(self, coordinates):
        embedding = torch.sin((coordinates[..., None] * self.frequencies + self.shifts).flatten(-2))
        return embedding


class NoneEncoding(torch.nn.Sequential):
    """
    Returns zeros
    """

    def __init__(self):
        self.n_embedding_dimensions = 1
        super().__init__()

    def forward(self, coordinates):
        return torch.cat(
            [
                torch.zeros((*coordinates.shape[:-1], 1), device=coordinates.device, dtype=torch.float),
            ],
            dim=-1,
        )


class OneEncoding(torch.nn.Sequential):
    """
    Returns ones
    """

    def __init__(self):
        self.n_embedding_dimensions = 1
        super().__init__()

    def forward(self, coordinates):
        return torch.cat(
            [
                torch.ones((*coordinates.shape[:-1], 1), device=coordinates.device, dtype=torch.float),
            ],
            dim=-1,
        )


class DirectEncoding(torch.nn.Sequential):
    """
    Dummy encoding of fragments, simply providing the positions directly
    """

    def __init__(self, window=(-10000, 10000)):
        self.n_embedding_dimensions = 3
        self.window = window
        super().__init__()

    def forward(self, coordinates):
        return torch.cat(
            [
                torch.ones((*coordinates.shape[:-1], 1), device=coordinates.device, dtype=torch.float),
                coordinates / (self.window[1] - self.window[0]) * 2,
            ],
            dim=-1,
        )


class DistanceEncoding(torch.nn.Module):
    def __init__(self, n_frequencies, window=[-100000, 100000]):
        super().__init__()

        self.register_buffer("shifts", torch.linspace(0, 1, n_frequencies))

        self.scale = window[1] - window[0]
        self.shift = window[0]

        self.n_embedding_dimensions = n_frequencies * 2

    def forward(self, coordinates):
        embedding = (((coordinates[..., None] - self.shift) / self.scale) - self.shifts).flatten(-2)
        return embedding


class RadialEncoding(torch.nn.Module):
    def __init__(self, n_frequencies, window=[-100000, 100000], n_features=2, parameterize_scale=True):
        super().__init__()

        self.shift = window[0]
        self.scale = (window[1] - window[0]) / n_frequencies

        self.register_buffer("locs", torch.arange(n_frequencies))
        if parameterize_scale:
            self.scales = torch.nn.Parameter(torch.ones(n_frequencies))
        else:
            self.register_buffer("scales", torch.tensor([1] * n_frequencies))

        self.n_embedding_dimensions = n_frequencies * n_features

    def forward(self, coordinates):
        coordinates = (coordinates[..., None] - self.shift) / self.scale
        embedding = torch.exp(-((coordinates - self.locs) ** 2) / 2 * self.scales**2).flatten(-2)
        return embedding


class RadialBinaryEncoding(torch.nn.Module):
    requires_grad = False

    def __init__(self, n_frequencies, window=[-100000, 100000], n_features=2, scale=1, parameterize_loc=False):
        super().__init__()

        # create frequencies
        if isinstance(n_frequencies, (tuple, list, np.ndarray)):
            pass
        else:
            n = n_frequencies
            n_frequencies = []
            while n > 1:
                n_frequencies.append(n)
                n = n // 2

        locs = []
        scales = []

        for n in n_frequencies:
            locs.extend(torch.linspace(*window, n + 1))
            scales.extend(((window[1] - window[0]) / n * scale) * np.ones(n + 1))

        if parameterize_loc:
            self.locs = torch.nn.Parameter(torch.tensor(locs).float())
        else:
            self.register_buffer("locs", torch.tensor(locs).float())
        self.register_buffer("scales", torch.tensor(scales).float())

        self.n_embedding_dimensions = len(locs) * n_features

    def forward(self, coordinates):
        coordinates = coordinates[..., None]
        embedding = torch.exp(-(((coordinates - self.locs) / self.scales) ** 2) / 2).flatten(-2)

        if self.requires_grad:
            embedding.requires_grad = True
            self.embedding = embedding
        return embedding


class RadialBinaryCenterEncoding(torch.nn.Module):
    requires_grad = False

    def __init__(self, n_frequencies, window=[-100000, 100000], scale=1, parameterize_loc=False):
        super().__init__()

        # create frequencies
        if isinstance(n_frequencies, (tuple, list, np.ndarray)):
            pass
        else:
            n = n_frequencies
            n_frequencies = []
            while n > 1:
                n_frequencies.append(n)
                n = n // 2

        locs = []
        scales = []

        for n in n_frequencies:
            locs.extend(torch.linspace(*window, n + 1))
            scales.extend(((window[1] - window[0]) / n * scale) * np.ones(n + 1))

        if parameterize_loc:
            self.locs = torch.nn.Parameter(torch.tensor(locs).float())
        else:
            self.register_buffer("locs", torch.tensor(locs).float())
        self.register_buffer("scales", torch.tensor(scales).float())

        self.n_embedding_dimensions = len(locs)

    def forward(self, coordinates):
        coordinates = coordinates[..., None].float().mean(-2, keepdim=True)
        embedding = torch.exp(-(((coordinates - self.locs) / self.scales) ** 2) / 2).flatten(-2) * 2

        if self.requires_grad:
            embedding.requires_grad = True
            self.embedding = embedding
        return embedding


class LinearBinaryEncoding(torch.nn.Module):
    requires_grad = False

    def __init__(self, n_frequencies, window=[-100000, 100000], n_features=2, parameterize_loc=False):
        super().__init__()

        # create frequencies
        if isinstance(n_frequencies, (tuple, list, np.ndarray)):
            pass
        else:
            n = n_frequencies
            n_frequencies = []
            while n > 1:
                n_frequencies.append(n)
                n = n // 2

        locs = []
        scales = []

        for n in n_frequencies:
            locs.extend(torch.linspace(*window, n + 1))
            delta = (window[1] - window[0]) / n
            scales.extend(delta * np.ones(n + 1))

        if parameterize_loc:
            self.locs = torch.nn.Parameter(torch.tensor(locs).float())
        else:
            self.register_buffer("locs", torch.tensor(locs).float())
        self.register_buffer("scales", torch.tensor(scales).float())

        self.n_embedding_dimensions = len(locs) * n_features

    def forward(self, coordinates):
        coordinates = coordinates[..., None]
        embedding = torch.abs(torch.clamp(self.scales - torch.abs(coordinates - self.locs), 0) / self.scales).flatten(
            -2
        )

        if self.requires_grad:
            embedding.requires_grad = True
            self.embedding = embedding
        return embedding


class ExponentialEncoding(torch.nn.Module):
    def __init__(self, window=[-100000, 100000], n_features=2, parameterize_scale=True):
        super().__init__()

        self.scale = window[1]

        self.rate = torch.nn.Parameter(torch.ones(1))

        self.n_embedding_dimensions = n_features

    def forward(self, coordinates):
        coordinates = torch.abs((coordinates) / self.scale)
        embedding = self.rate * torch.exp(-coordinates * self.rate)
        return embedding


class TophatEncoding(torch.nn.Module):
    def __init__(self, n_frequencies, window=[-100000, 100000], n_features=2):
        super().__init__()

        self.shift = window[0]
        self.scale = (window[1] - window[0]) / n_frequencies

        self.register_buffer("locs", torch.arange(n_frequencies))
        self.register_buffer("scales", torch.tensor([1] * n_frequencies))

        self.n_embedding_dimensions = n_frequencies * n_features

    def forward(self, coordinates):
        coordinates = (coordinates[..., None] - self.shift) / self.scale
        embedding = ((coordinates - self.locs).abs() < self.scales).flatten(-2).float()
        return embedding


class DirectDistanceEncoding(torch.nn.Module):
    def __init__(self, max=1000):
        super().__init__()

        self.max = max

        self.n_embedding_dimensions = 1

    def forward(self, coordinates):
        return torch.clamp(coordinates[:, 1] - coordinates[:, 0], max=self.max).unsqueeze(-1) / self.max


class SplitDistanceEncoding(torch.nn.Module):
    def __init__(self, splits=[60, 170]):
        super().__init__()

        self.register_buffer("splits", torch.tensor(splits))

        self.n_embedding_dimensions = len(splits) + 1

    def forward(self, coordinates):
        return torch.nn.functional.one_hot(
            torch.searchsorted(self.splits, coordinates[:, 1] - coordinates[:, 0]), self.n_embedding_dimensions
        )


class LinearDistanceEncoding(torch.nn.Module):
    def __init__(self, locs=[0, 100, 200, 300, 400, 500, 600, 700, 800]):
        super().__init__()

        self.register_buffer("locs", torch.tensor(locs))
        self.register_buffer("scales", torch.ones(len(locs)) * (locs[1] - locs[0]))

        self.n_embedding_dimensions = len(locs)

    def forward(self, coordinates):
        distances = coordinates[:, 1] - coordinates[:, 0]
        embedding = torch.abs(
            torch.clamp(self.scales - torch.abs(distances.unsqueeze(-1) - self.locs), 0) / self.scales
        )
        return embedding


class SplineBinaryEncoding(torch.nn.Module):
    def __init__(
        self, binwidths=(100, 200, 500, 1000, 2000, 5000), window=(-100000, 100000), n_embedding_dimensions=100
    ):
        super().__init__()
        self.register_buffer("binwidths", torch.tensor(binwidths))
        self.register_buffer("binshifts", window[0] // self.binwidths)
        nbins = torch.tensor([(window[1] - window[0]) // binwidth + 1 for binwidth in self.binwidths])
        self.register_buffer(
            "bincumstarts", torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(nbins, 0)[:-1]])
        )

        self.w = torch.nn.Parameter(torch.zeros(nbins.sum(), n_embedding_dimensions))
        self.n_embedding_dimensions = n_embedding_dimensions
        self.window = window

    def forward(self, coordinates):
        coords = torch.clamp(coordinates, self.window[0], self.window[1] - 1)

        cumidxs = (
            coords // self.binwidths[:, None, None] - self.binshifts[:, None, None] + self.bincumstarts[:, None, None]
        )
        alphas = coords % self.binwidths[:, None, None] / self.binwidths[:, None, None]

        w0 = self.w.index_select(0, cumidxs.flatten()).reshape(cumidxs.shape + (self.n_embedding_dimensions,))
        w1 = self.w.index_select(0, cumidxs.flatten() + 1).reshape(cumidxs.shape + (self.n_embedding_dimensions,))

        # b = bin, f = fragment, c = left/right cut site, d = embedding dimension
        out = torch.einsum("bfcd,bfc->fd", w0, 1 - alphas) + torch.einsum("bfcd,bfc->fd", w1, alphas)
        return out


class SplineRegionalBinaryEncoding2(torch.nn.Module):
    def __init__(
        self,
        n_regions,
        # binwidths=(500,),
        binwidths=(100, 200, 500, 1000, 2000, 5000),
        window=(-100000, 100000),
        n_embedding_dimensions=100,
    ):
        super().__init__()
        self.register_buffer("binwidths", torch.tensor(binwidths))
        self.register_buffer("binshifts", window[0] // self.binwidths)
        nbins = torch.tensor([(window[1] - window[0]) // binwidth + 1 for binwidth in self.binwidths])
        self.n_weights = nbins.sum()
        self.register_buffer(
            "bincumstarts", torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(nbins, 0)[:-1]])
        )

        self.w = FeatureParameter(n_regions, (self.n_weights, n_embedding_dimensions), constructor=torch.zeros)
        self.n_embedding_dimensions = n_embedding_dimensions
        self.window = window
        self.n_regions = n_regions

        self.bias = torch.nn.Embedding(n_regions, n_embedding_dimensions, sparse=True)

        self.device = "cpu"

    def forward(self, coordinates, regions_oi, local_region_ix):
        coords = torch.clamp(coordinates, self.window[0], self.window[1] - 1)

        localregionxbin_ix = (
            coords // self.binwidths[:, None, None] - self.binshifts[:, None, None] + self.bincumstarts[:, None, None]
        ) + local_region_ix[None, :, None] * self.n_weights
        # print(localregionxbin_ix)
        alphas = coords % self.binwidths[:, None, None] / self.binwidths[:, None, None]

        w_genes = self.w(regions_oi.to("cpu").numpy()).reshape(-1, self.n_embedding_dimensions).to(self.device)

        w0 = w_genes[localregionxbin_ix.flatten()].reshape(localregionxbin_ix.shape + (self.n_embedding_dimensions,))
        w1 = w_genes[localregionxbin_ix.flatten() + 1].reshape(
            localregionxbin_ix.shape + (self.n_embedding_dimensions,)
        )

        # b = regionxbin, f = fragment, c = left/right cut site, d = embedding dimension
        out = (
            torch.einsum("bfcd,bfc->fd", w0, 1 - alphas)
            + torch.einsum("bfcd,bfc->fd", w1, alphas)
            + self.bias(
                regions_oi[local_region_ix]
            )  # you can activate the bias here again once the first linear layer is gone
        )
        return out

    def parameters_sparse(self):
        yield self.w
        yield self.bias

    def _apply(self, fn, recurse=True):
        w = self._modules["w"]
        del self._modules["w"]
        super()._apply(fn)
        self._modules["w"] = w
        self.device = self.bias.weight.device
        return self


class SplineRegionalBinaryEncoding2(torch.nn.Module):
    def __init__(
        self,
        n_regions,
        # binwidths=(500,),
        binwidths=(100, 200, 500, 1000, 2000, 5000),
        window=(-100000, 100000),
        n_embedding_dimensions=100,
    ):
        super().__init__()
        self.register_buffer("binwidths", torch.tensor(binwidths))
        self.register_buffer("binshifts", window[0] // self.binwidths)
        nbins = torch.tensor([(window[1] - window[0]) // binwidth + 1 for binwidth in self.binwidths])
        self.n_weights = nbins.sum()
        self.register_buffer(
            "bincumstarts", torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(nbins, 0)[:-1]])
        )

        self.w = FeatureParameter(
            n_regions, (self.n_weights, n_embedding_dimensions), constructor=torch.zeros, pin_memory=True
        )
        self.n_embedding_dimensions = n_embedding_dimensions
        self.window = window
        self.n_regions = n_regions

        self.bias = FeatureParameter(n_regions, n_embedding_dimensions)

        self.device = "cpu"

    def forward(self, coordinates, head_mappings):
        coords = torch.clamp(coordinates, self.window[0], self.window[1] - 1)

        bin_ix = (
            coords // self.binwidths[:, None, None] - self.binshifts[:, None, None] + self.bincumstarts[:, None, None]
        )
        alphas = coords % self.binwidths[:, None, None] / self.binwidths[:, None, None]

        output = torch.zeros(
            coordinates.shape[:-1] + (self.n_embedding_dimensions,), device=coordinates.device, dtype=torch.float
        )

        for ix, idx in head_mappings.items():
            w = self.w[ix].to("cuda")
            bin_ix_ = bin_ix[:, idx]
            w0 = w[bin_ix_.flatten()].reshape(bin_ix_.shape + (self.n_embedding_dimensions,))
            w1 = w[bin_ix_.flatten() + 1].reshape(bin_ix_.shape + (self.n_embedding_dimensions,))
            output[idx] = (
                torch.einsum("bfcd,bfc->fd", w0, 1 - alphas[:, idx])
                + torch.einsum("bfcd,bfc->fd", w1, alphas[:, idx])
                + self.bias[ix]
            )
        return output

    def parameters_sparse(self):
        yield self.w
        yield self.bias

    def _apply(self, fn, recurse=True):
        w = self._modules["w"]
        del self._modules["w"]
        super()._apply(fn)
        self._modules["w"] = w
        self.device = self.bias[0].device
        return self

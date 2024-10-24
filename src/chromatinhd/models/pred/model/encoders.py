import torch
import numpy as np
from chromatinhd.embedding import FeatureParameter
import math


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

    def __init__(
        self,
        n_frequencies,
        window=[-100000, 100000],
        n_features=2,
        scale=1,
        parameterize_loc=False,
    ):
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


class RadialBinaryEncoding2(torch.nn.Module):
    requires_grad = False

    def __init__(
        self,
        bin_widths=(100, 200, 500, 1000, 2000, 5000),
        window=(-100000, 100000),
        n_features=2,
        scale=1,
        parameterize_loc=False,
    ):
        super().__init__()

        locs = []
        scales = []

        for bin_width in bin_widths:
            loc = torch.arange(window[0], window[1] + 1, bin_width)
            locs.extend(loc)
            scales.extend((bin_width * scale) * np.ones(len(loc)))

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

        # b = bin, f = fragment, c = cut sites (left/right), d = embedding dimension
        out = torch.einsum("bfcd,bfc->fd", w0, 1 - alphas) + torch.einsum("bfcd,bfc->fd", w1, alphas)
        return out


class SplineBinaryFullEncoding(torch.nn.Module):
    def __init__(self, binwidths=(100, 200, 500, 1000, 2000, 5000), window=(-100000, 100000)):
        super().__init__()
        self.register_buffer("binwidths", torch.tensor(binwidths))
        self.register_buffer("binshifts", window[0] // self.binwidths)
        self.nbins = torch.tensor([(window[1] - window[0]) // binwidth + 1 for binwidth in self.binwidths])
        self.register_buffer(
            "bincumstarts", torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(self.nbins, 0)[:-1]])
        )
        self.window = window

        self.register_buffer(
            "binpositions", torch.concatenate([torch.arange(window[0], window[1] + 1, bw) for bw in binwidths])
        )
        self.register_buffer(
            "binscales",
            torch.concatenate([torch.tensor([bw] * math.ceil((window[1] - window[0] + 1) / bw)) for bw in binwidths]),
        )

    def forward(self, coordinates):
        coordinates = coordinates[..., None]
        embedding = torch.clamp(1 - torch.abs((self.binpositions - coordinates) / self.binscales), 0, 1).flatten(-2)
        return embedding


import time


class catchtime(object):
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.t = time.time() - self.t
        # print(self.name, self.t)


class MultiSplineBinaryEncoding(torch.nn.Module):
    def __init__(
        self,
        n_regions,
        binwidths=(100, 200, 500, 1000, 2000, 5000),
        window=(-100000, 100000),
        n_embedding_dimensions=100,
        weight_constructor=torch.zeros,
    ):
        super().__init__()
        self.register_buffer("binwidths", torch.tensor(binwidths))
        self.register_buffer("binshifts", window[0] // self.binwidths)
        nbins = torch.tensor([(window[1] - window[0]) // binwidth + 1 for binwidth in self.binwidths])
        self.register_buffer(
            "bincumstarts", torch.cat([torch.zeros(1, dtype=torch.int64), torch.cumsum(nbins, 0)[:-1]])
        )

        self.w = FeatureParameter(
            n_regions,
            (
                nbins.sum(),
                n_embedding_dimensions,
            ),
            constructor=torch.zeros,
        )
        self.n_embedding_dimensions = n_embedding_dimensions
        self.window = window

    def forward(self, coordinates, indptr, regions_oi):
        coords = torch.clamp(coordinates, self.window[0], self.window[1] - 1)

        with catchtime("cumidx"):
            cumidxs = (
                coords // self.binwidths[:, None, None]
                - self.binshifts[:, None, None]
                + self.bincumstarts[:, None, None]
            ).transpose(1, 0)
            alphas = (coords % self.binwidths[:, None, None] / self.binwidths[:, None, None]).transpose(1, 0)

        with catchtime("zoof"):
            outputs = []
            for ix, start, end in zip(regions_oi, indptr[:-1], indptr[1:]):
                with catchtime("subset"):
                    cumidxs_oi = cumidxs[start:end]
                    alphas_oi = alphas[start:end]
                with catchtime("select"):
                    w0 = (
                        self.w[ix]
                        .index_select(0, cumidxs_oi.flatten())
                        .reshape(cumidxs_oi.shape + (self.n_embedding_dimensions,))
                    )
                    w1 = (
                        self.w[ix]
                        .index_select(0, cumidxs_oi.flatten() + 1)
                        .reshape(cumidxs_oi.shape + (self.n_embedding_dimensions,))
                    )

                # b = bin, f = fragment, c = left/right cut site, d = embedding dimension
                with catchtime("einsum"):
                    outputs.append(
                        torch.einsum("fbcd,fbc->fd", w0, 1 - alphas_oi) + torch.einsum("fbcd,fbc->fd", w1, alphas_oi)
                    )
        return torch.cat(outputs, dim=0)

    # def forward(self, coordinates, indptr, regions_oi):
    #     coords = torch.clamp(coordinates, self.window[0], self.window[1] - 1)

    #     with catchtime("cumidx"):
    #         cumidxs = (
    #             coords // self.binwidths[:, None, None]
    #             - self.binshifts[:, None, None]
    #             + self.bincumstarts[:, None, None]
    #         ).transpose(1, 0)
    #         alphas = (coords % self.binwidths[:, None, None] / self.binwidths[:, None, None]).transpose(1, 0)

    #     output = torch.zeros((coordinates.shape[0], self.n_embedding_dimensions), device=coordinates.device)
    #     with catchtime("zoof"):
    #         for ix, start, end in zip(regions_oi, indptr[:-1], indptr[1:]):
    #             with catchtime("subset"):
    #                 cumidxs_oi = cumidxs[start:end]
    #                 alphas_oi = alphas[start:end]
    #             with catchtime("select"):
    #                 w0 = (
    #                     self.w[ix]
    #                     .index_select(0, cumidxs_oi.flatten())
    #                     .reshape(cumidxs_oi.shape + (self.n_embedding_dimensions,))
    #                 )
    #                 w1 = (
    #                     self.w[ix]
    #                     .index_select(0, cumidxs_oi.flatten() + 1)
    #                     .reshape(cumidxs_oi.shape + (self.n_embedding_dimensions,))
    #                 )

    #             # b = bin, f = fragment, c = left/right cut site, d = embedding dimension
    #             output[start:end] = torch.einsum("fbcd,fbc->fd", w0, 1 - alphas_oi) + torch.einsum(
    #                 "fbcd,fbc->fd", w1, alphas_oi
    #             )
    #     return output


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


class TophatBinaryEncoding(torch.nn.Module):
    requires_grad = False

    def __init__(
        self,
        n_frequencies,
        window=[-100000, 100000],
        n_features=2,
        scale=1,
    ):
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

        self.register_buffer("locs", torch.tensor(locs).float())
        self.register_buffer("scales", torch.tensor(scales).float())

        self.n_embedding_dimensions = len(locs) * n_features

    def forward(self, coordinates):
        coordinates = coordinates[..., None]
        normalized = ((coordinates - self.locs) / self.scales).flatten(-2)
        embedding = (normalized > 0) & (normalized <= 1)

        if self.requires_grad:
            embedding.requires_grad = True
            self.embedding = embedding
        return embedding


class DirectDistanceEncoding(torch.nn.Module):
    def __init__(self, max=800):
        super().__init__()

        self.max = max

        self.n_embedding_dimensions = 1

    def forward(self, coordinates):
        return torch.clamp(torch.abs(coordinates[:, 1] - coordinates[:, 0]), max=self.max).unsqueeze(-1) / self.max


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

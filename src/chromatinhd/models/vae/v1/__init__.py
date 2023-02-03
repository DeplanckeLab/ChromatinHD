import chromatinhd.models.positional.v16 as positional_model

import torch
import numpy as np
import math

from chromatinhd.embedding import EmbeddingTensor


def initialize_mixture(window, coordinates, genemapping, n_genes, n_components=64):
    """
    Initializes a mixture model by binning and selecting the top bins as components
    """
    bindiff = (window[1] - window[0]) / n_components / 10
    bins = torch.arange(window[0] - bindiff / 2, window[1] - bindiff / 2 + 0.1, bindiff)
    bins_mid = bins + bindiff / 2
    coordinates_binned = (
        torch.from_numpy(np.digitize(coordinates.cpu().numpy(), bins)) - 1
    )

    locs = []
    logits = []

    for gene_ix in range(n_genes):
        fragments_oi = genemapping == gene_ix
        bincounts = torch.bincount(
            coordinates_binned[fragments_oi].flatten(), minlength=len(bins)
        )
        bins_ranked = torch.argsort(bincounts, descending=True)
        bins_chosen = bins_ranked[:n_components]
        loc = bins_mid[bins_chosen]
        logit = torch.log1p(bincounts[bins_chosen])

        locs.append(loc)
        logits.append(logit)

    locs = torch.stack(locs)
    logits = torch.stack(logits)
    return locs, logits


class Decoder(torch.nn.Module):
    def __init__(
        self, n_latent, n_genes, n_output_components, n_layers=1, n_hidden_dimensions=16
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                torch.nn.Linear(
                    n_hidden_dimensions if i > 0 else n_latent, n_hidden_dimensions
                )
            )
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(n_hidden_dimensions))
            layers.append(torch.nn.Dropout(0.5))
        self.nn = torch.nn.Sequential(*layers)

        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_output_components = n_output_components

        self.weight = EmbeddingTensor(
            n_genes,
            (n_hidden_dimensions if n_layers > 0 else n_latent, n_output_components),
            sparse=True,
        )
        stdv = 1.0 / math.sqrt(self.weight.weight.size(1))
        self.weight.weight.data.uniform_(-stdv, stdv)

        self.bias = EmbeddingTensor(n_genes, (n_output_components,), sparse=True)
        self.bias.weight.data.zero_()

    def forward(self, latent, genes_oi):
        weight = self.weight(genes_oi)
        bias = self.bias(genes_oi)
        nn_output = (
            self.nn(latent).unsqueeze(1).unsqueeze(2)
        )  # broadcast across genes and across components
        print(weight.shape)
        print(nn_output.shape)
        return torch.matmul(nn_output, weight).squeeze(-2) + bias

    def parameters_dense(self):
        return [*self.nn.parameters()]

    def parameters_sparse(self):
        return [self.weight.weight, self.bias.weight]


class Mixture(torch.nn.Module):
    def __init__(
        self,
        n_components,
        n_genes,
        window,
        loc_init=None,
        scale_init=None,
        logit_init=None,
        debug=False,
    ):
        super().__init__()
        self.n_components = n_components
        self.register_buffer("a", torch.tensor(window[0]).unsqueeze(-1))
        self.register_buffer("b", torch.tensor(window[1]).unsqueeze(-1))
        self.register_buffer("ab", self.b - self.a)

        self.loc = EmbeddingTensor(n_genes, (n_components,), sparse=True)
        if loc_init is None:
            loc_init = torch.linspace(0, 1, n_components + 2)[1:-1]
        self.loc.weight.data[:, :] = torch.special.logit(loc_init / self.ab)

        if self.loc.weight.data.isnan().any():
            raise ValueError(
                "Some locs are outside of bounds: ",
                torch.where(self.loc.weight.data.isnan()),
            )

        self.register_buffer("scale_lower_bound", self.ab / n_components / 100)
        self.scale = EmbeddingTensor(n_genes, (n_components,), sparse=True)
        if scale_init is None:
            scale_init = torch.log(self.scale_lower_bound * 10)
        self.scale.weight.data[:, :] = scale_init

        self.logit = EmbeddingTensor(n_genes, (n_components,), sparse=True)
        if logit_init is None:
            logit_init = self.logit.weight.data.normal_() / math.sqrt(n_components)
        self.logit.weight.data[:, :] = logit_init

        self.debug = debug

    def log_prob(self, value, weight_change, genes_oi, local_gene_ix):
        loc = self.loc(genes_oi)
        loc = ((torch.special.expit(loc) * self.ab))[local_gene_ix]
        if self.debug:
            if loc.isnan().any():
                raise ValueError("NaNs in loc", torch.where(loc.isnan()))

        scale = (self.scale_lower_bound + torch.exp(self.scale(genes_oi)))[
            local_gene_ix
        ]

        component_distribution = torch.distributions.Normal(
            loc=loc, scale=scale, validate_args=True
        )
        logit_intercept = self.logit(genes_oi)[local_gene_ix]
        logits = logit_intercept + weight_change
        mixture_distribution = torch.distributions.Categorical(logits=logits)
        dist = torch.distributions.MixtureSameFamily(
            mixture_distribution, component_distribution, validate_args=True
        )

        return dist.log_prob(value)

    def parameters_dense(self):
        return []

    def parameters_sparse(self):
        return [self.loc.weight, self.scale.weight, self.logit.weight]


class CellGeneEmbeddingToCellEmbedding(torch.nn.Module):
    def __init__(self, n_input_features, n_genes, n_output_features=None):
        if n_output_features is None:
            n_output_features = n_input_features

        super().__init__()

        # default initialization same as a torch.nn.Linear
        self.bias1 = torch.nn.Parameter(
            torch.empty(n_output_features, requires_grad=True)
        )
        self.bias1.data.zero_()

        self.n_output_features = n_output_features

        self.weight1 = EmbeddingTensor(
            n_genes, (n_input_features, self.n_output_features), sparse=True
        )
        stdv = 1.0 / math.sqrt(self.weight1.weight.size(-1))
        self.weight1.weight.data.uniform_(-stdv, stdv)

    def forward(self, cellgene_embedding, genes_oi):
        weight1 = self.weight1(genes_oi)
        bias1 = self.bias1
        return torch.einsum("abc,bcd->ad", cellgene_embedding, weight1) + bias1

    def parameters_dense(self):
        return [self.bias1]

    def parameters_sparse(self):
        return [self.weight1.weight]


class Encoder(torch.nn.Module):
    def __init__(self, n_input_features, n_latent, n_layers=1, n_hidden_dimensions=16):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                torch.nn.Linear(
                    n_hidden_dimensions if i > 0 else n_input_features,
                    n_hidden_dimensions,
                )
            )
            layers.append(torch.nn.ReLU())
            # layers.append(torch.nn.Dropout(0.5))
        self.nn = torch.nn.Sequential(*layers)

        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_latent = n_latent

        self.linear_loc = torch.nn.Linear(
            n_hidden_dimensions if n_layers > 0 else n_input_features, n_latent
        )
        self.linear_scale = torch.nn.Linear(
            n_hidden_dimensions if n_layers > 0 else n_input_features, n_latent
        )

    def forward(self, x):
        nn_output = self.nn(x)
        loc = self.linear_loc(nn_output)
        scale = self.linear_scale(nn_output).exp()
        return loc, scale

    def parameters_dense(self):
        return self.parameters()

    def parameters_sparse(self):
        return []

import peakfreeatac.models.positional.v16 as positional_model

import torch
import numpy as np
import math
import tqdm.auto as tqdm
import torch_scatter

from peakfreeatac.embedding import EmbeddingTensor

def initialize_mixture(coordinates, genemapping, n_genes, n_components = 64):
    """
    Initializes a mixture model by binning and selecting the top bins as components
    """
    binwidth = 1/n_components/5
    bins = torch.arange(0-binwidth/2, 1-binwidth/2 + 0.1, binwidth)
    bins_mid = bins + binwidth/2
    coordinates_binned = torch.clamp(torch.from_numpy(np.digitize(coordinates.cpu().numpy(), bins)) - 1, 0, len(bins))

    bincounts = torch.bincount(coordinates_binned + genemapping * len(bins), minlength = len(bins) * n_genes)
    bincount_reshaped = bincounts.reshape((n_genes, len(bins)))
    bins_chosen = torch.argsort(bincount_reshaped, 1, descending = True)[:, :n_components]

    locs = bins_mid[bins_chosen]
    locs[locs <= 0] = 1e-4
    locs[locs >= 1] = 1-1e-4

    logits = torch.log1p(bincount_reshaped[torch.repeat_interleave(torch.arange(n_genes), n_components), bins_chosen.flatten()].reshape((n_genes, n_components)))

    return locs, logits


class Decoder(torch.nn.Module):
    def __init__(self, n_latent, n_genes, n_output_components, n_layers = 1, n_hidden_dimensions = 32):
        super().__init__()
        
        layers = []
        for i in range(n_layers):
            layers.append(torch.nn.Linear(n_hidden_dimensions if i > 0 else n_latent, n_hidden_dimensions))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(n_hidden_dimensions))
            layers.append(torch.nn.Dropout(0.2))
        self.nn = torch.nn.Sequential(*layers)
        
        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_output_components = n_output_components
        
        self.logit_weight = EmbeddingTensor(n_genes, (n_hidden_dimensions if n_layers > 0 else n_latent, n_output_components), sparse = True)
        stdv = 1. / math.sqrt(self.logit_weight.weight.size(1))
        self.logit_weight.weight.data.uniform_(-stdv, stdv)

        self.rho_weight = EmbeddingTensor(n_genes, (n_hidden_dimensions if n_layers > 0 else n_latent, ), sparse = True)
        stdv = 1. / math.sqrt(self.rho_weight.weight.size(1)) / 100
        self.rho_weight.weight.data.uniform_(-stdv, stdv)
        
    def forward(self, latent, genes_oi):
        logit_weight = self.logit_weight(genes_oi)
        rho_weight = self.rho_weight(genes_oi)
        nn_output = self.nn(latent)

        # nn_output is broadcasted across genes and across components
        logit = torch.matmul(nn_output.unsqueeze(1).unsqueeze(2), logit_weight).squeeze(-2)

        rho = torch.matmul(nn_output, rho_weight.T).squeeze(-1)
        return logit, rho
    
    def parameters_dense(self):
        return [*self.nn.parameters()]
    
    def parameters_sparse(self):
        return [self.logit_weight.weight, self.rho_weight.weight]


class BaselineDecoder(torch.nn.Module):
    def __init__(self, n_latent, n_genes, n_output_components, n_layers = 1, n_hidden_dimensions = 32):
        super().__init__()
        
        layers = []
        for i in range(n_layers):
            layers.append(torch.nn.Linear(n_hidden_dimensions if i > 0 else n_latent, n_hidden_dimensions))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(n_hidden_dimensions))
            layers.append(torch.nn.Dropout(0.2))
        self.nn = torch.nn.Sequential(*layers)
        
        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_output_components = n_output_components

        self.rho_weight = EmbeddingTensor(n_genes, (n_hidden_dimensions if n_layers > 0 else n_latent, ), sparse = True)
        stdv = 1. / math.sqrt(self.rho_weight.weight.size(1)) / 100
        self.rho_weight.weight.data.uniform_(-stdv, stdv)

    def forward(self, latent, genes_oi):
        rho_weight = self.rho_weight(genes_oi)
        nn_output = self.nn(latent)

        # nn_output is broadcasted across genes and across components
        logit = torch.zeros((latent.shape[0], len(genes_oi), self.n_output_components), device = latent.device)

        rho = torch.matmul(nn_output, rho_weight.T).squeeze(-1)
        return logit, rho
    
    def parameters_dense(self):
        return [*self.nn.parameters()]
    
    def parameters_sparse(self):
        return [self.rho_weight.weight]

class Mixture(torch.nn.Module):
    """
    Note: the log prob is calculated within [0, 1] to normalize the gradients
    
    """
    def __init__(self, n_components, n_genes, loc_init = None, scale_init = None, logit_init = None, debug = False):
        super().__init__()
        self.n_components = n_components
        
        self.loc = EmbeddingTensor(n_genes, (n_components, ), sparse = True)
        if loc_init is None:
            loc_init = torch.linspace(0, 1, n_components+2)[1:-1]
        self.loc.weight.data[:, :] = torch.special.logit(loc_init)

        if self.loc.weight.data.isnan().any():
            raise ValueError("Some locs are outside of bounds: ", torch.where(self.loc.weight.data.isnan()))

        self.register_buffer("scale_lower_bound", torch.tensor(1e-5))
        self.scale = EmbeddingTensor(n_genes, (n_components, ), sparse = True)
        if scale_init is None:
            scale_init = math.log(1/n_components / 5)
        self.scale.weight.data[:, :] = scale_init
        
        self.logit = EmbeddingTensor(n_genes, (n_components, ), sparse = True)
        if logit_init is None:
            logit_init = self.logit.weight.data.normal_() / math.sqrt(n_components)
        self.logit.weight.data[:, :] = logit_init

        self.debug = debug
        
    def log_prob(self, value, delta_logit, genes_oi, local_gene_ix):
        loc = self.loc(genes_oi)
        loc =  torch.special.expit(loc)[local_gene_ix]
        if self.debug:
            if loc.isnan().any():
                raise ValueError("NaNs in loc", torch.where(loc.isnan()))
        
        scale = self.scale_lower_bound + torch.exp(self.scale(genes_oi))[local_gene_ix]

        component_distribution = torch.distributions.Normal(
            loc = loc,
            scale = scale,
            validate_args = False
        )
        logits = self.logit(genes_oi)[local_gene_ix]

        if delta_logit is not None:
            logits = logits + delta_logit

        mixture_distribution = torch.distributions.Categorical(logits = logits)
        dist = torch.distributions.MixtureSameFamily(
            mixture_distribution,
            component_distribution,
            validate_args = False
        )
        
        return dist.log_prob(value)
    
    def parameters_dense(self):
        return []
    
    def parameters_sparse(self):
        return [self.loc.weight, self.scale.weight, self.logit.weight]

class Decoding(torch.nn.Module):
    def __init__(self, fragments, cell_latent_space, n_components = 32, decoder_n_layers = 0, baseline = False):
        super().__init__()

        n_latent_dimensions = cell_latent_space.shape[1]
        
        locs, logits = initialize_mixture(fragments.cut_coordinates, fragments.cut_local_gene_ix, fragments.n_genes, n_components = n_components)
        self.mixture = Mixture(
            n_components,
            fragments.n_genes,
            loc_init = locs,
            logit_init = logits
        )

        if not baseline:
            self.decoder = Decoder(
                n_latent_dimensions,
                fragments.n_genes,
                self.mixture.n_components,
                n_layers = decoder_n_layers
            )
        else:
            self.decoder = BaselineDecoder(
                n_latent_dimensions,
                fragments.n_genes,
                self.mixture.n_components,
                n_layers = decoder_n_layers
            )
        
        libsize = torch.bincount(fragments.mapping[:, 0], minlength = fragments.n_cells)
        rho_bias = torch.bincount(fragments.mapping[:, 1], minlength = fragments.n_genes) / fragments.n_cells / libsize.to(torch.float).mean()
        
        self.register_buffer("libsize", libsize)
        self.register_buffer("rho_bias", rho_bias)

        self.register_buffer("cell_latent_space", cell_latent_space)

        self.track = {}
        
    def forward_(
        self,
        local_cellxgene_ix,
        cut_coordinates,
        latent,
        genes_oi,
        cells_oi,
        cut_local_cellxgene_ix,
        cut_local_gene_ix,
        n_cells,
        n_genes
    ):
        if latent is None:
            latent = self.cell_latent_space[cells_oi]

        # decode
        mixture_delta, rho_change = self.decoder(latent, genes_oi)

        # fragment counts
        mixture_delta_cellxgene = mixture_delta.view(np.prod(mixture_delta.shape[:2]), mixture_delta.shape[-1])
        mixture_delta = mixture_delta_cellxgene[cut_local_cellxgene_ix]

        likelihood_mixture = self.track["likelihood_mixture"] = self.mixture.log_prob(cut_coordinates, mixture_delta, genes_oi, cut_local_gene_ix)

        # expression
        fragmentexpression = (self.rho_bias[genes_oi] * torch.exp(rho_change)) * self.libsize[cells_oi].unsqueeze(1)
        fragmentcounts_p = torch.distributions.Poisson(fragmentexpression)
        fragmentcounts = torch.reshape(torch.bincount(local_cellxgene_ix, minlength = n_cells * n_genes), (n_cells, n_genes))
        likelihood_fragmentcounts = self.track["likelihood_fragmentcounts"] = fragmentcounts_p.log_prob(fragmentcounts)

        # likelihood
        likelihood = likelihood_mixture.sum() + likelihood_fragmentcounts.sum()

        # ELBO
        elbo = -likelihood
        
        return elbo

    def forward(self, data):
        if not hasattr(data, "latent"):
            data.latent = self.cell_latent_space[data.cells_oi]
        return self.forward_(
            cut_coordinates = data.cut_coordinates,
            latent = data.latent,
            cells_oi = data.cells_oi_torch,
            genes_oi = data.genes_oi_torch,
            cut_local_cellxgene_ix = data.cut_local_cellxgene_ix,
            n_cells = data.n_cells,
            n_genes = data.n_genes,
            cut_local_gene_ix = data.cut_local_gene_ix,
            local_cellxgene_ix = data.local_cellxgene_ix,
        )

    def _get_likelihood_mixture_cell_gene(self, likelihood_mixture, cut_local_cellxgene_ix, n_cells, n_genes):
        return torch_scatter.segment_sum_coo(likelihood_mixture, cut_local_cellxgene_ix, dim_size = n_cells * n_genes).reshape((n_cells, n_genes))

    def forward_likelihood_mixture(self, data):
        self.forward(data)
        return self.track["likelihood_mixture"]

    def forward_likelihood(self, data):
        self.forward(data)
        likelihood_mixture_cell_gene = self._get_likelihood_mixture_cell_gene(
            self.track["likelihood_mixture"],
            data.cut_local_cellxgene_ix,
            data.n_cells,
            data.n_genes
        )

        likelihood = (
            self.track["likelihood_fragmentcounts"] +
            likelihood_mixture_cell_gene
        )

        return likelihood

    def parameters_dense(self):
        return [parameter for module in self._modules.values() for parameter in (module.parameters_dense() if hasattr(module, "parameters_dense") else module.parameters())]
    
    def parameters_sparse(self):
        return [parameter for module in self._modules.values() for parameter in (module.parameters_sparse() if hasattr(module, "parameters_sparse") else [])]

    def evaluate_pseudo(self, coordinates, latent = None, gene_oi = None):
        device = coordinates.device
        if not torch.is_tensor(latent):
            if latent is None:
                latent = 0.
            latent = torch.ones((1, self.n_latent), device = coordinates.device) * latent
        if gene_oi is None:
            gene_oi = 0
        genes_oi = torch.tensor([gene_oi], device = coordinates.device, dtype = torch.long)
        cells_oi = torch.ones((1, ), dtype = torch.long)
        local_cellxgene_ix = torch.tensor([], dtype = torch.long)
        cut_local_gene_ix = torch.zeros_like(coordinates).to(torch.long)
        cut_local_cellxgene_ix = torch.zeros_like(coordinates).to(torch.long)

        data = FullDict(
            local_cellxgene_ix = local_cellxgene_ix.to(device),
            cut_local_gene_ix = cut_local_gene_ix.to(device),
            cut_local_cellxgene_ix = cut_local_cellxgene_ix.to(device),
            cut_coordinates = coordinates.to(device),
            n_cells = 1,
            n_genes = 1,
            genes_oi_torch = genes_oi.to(device),
            cells_oi_torch = cells_oi.to(device),
            latent = latent.to(device)
        )
        return self.forward_likelihood_mixture(data)

class FullDict(dict):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

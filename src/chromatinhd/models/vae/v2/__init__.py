import peakfreeatac.models.positional.v16 as positional_model

import torch
import numpy as np
import math
import tqdm.auto as tqdm

from peakfreeatac.embedding import EmbeddingTensor


def initialize_mixture(window, coordinates, genemapping, n_genes, n_components = 64):
    """
    Initializes a mixture model by binning and selecting the top bins as components
    """
    bindiff = (window[1] - window[0])/n_components
    bins = torch.arange(window[0]-bindiff/2, window[1]-bindiff/2 + 0.1, bindiff)
    bins_mid = bins + bindiff/2
    coordinates_binned = torch.clamp(torch.from_numpy(np.digitize(coordinates.cpu().numpy(), bins)) - 1, 0, len(bins))

    locs = []
    logits = []

    for gene_ix in tqdm.tqdm(range(n_genes)):
        fragments_oi = genemapping == gene_ix
        bincounts = torch.bincount(coordinates_binned[fragments_oi].flatten().to(torch.int), minlength = len(bins))
        bins_ranked = torch.argsort(bincounts, descending = True)

        bins_chosen = bins_ranked[:n_components]
        print(bincounts)
        print(bins_chosen)
        loc = bins_mid[bins_chosen]
        logit = torch.log1p(bincounts[bins_chosen])
        
        locs.append(loc)
        logits.append(logit)

        print(logit)

        break
        
    locs = torch.stack(locs)
    logits = torch.stack(logits)
    return locs, logits

def initialize_mixture(window, coordinates, genemapping, n_genes, n_components = 64):
    """
    Initializes a mixture model by binning and selecting the top bins as components
    """
    binwidth = (window[1] - window[0])/n_components/5
    bins = torch.arange(window[0]-binwidth/2, window[1]-binwidth/2 + 0.1, binwidth)
    bins_mid = bins + binwidth/2
    coordinates_binned = torch.clamp(torch.from_numpy(np.digitize(coordinates.cpu().numpy(), bins)) - 1, 0, len(bins))

    genemapping_flatten = torch.repeat_interleave(genemapping, 2)
    coordinates_binned_flatten = coordinates_binned.flatten()

    bincounts = torch.bincount(coordinates_binned_flatten + genemapping_flatten * len(bins), minlength = len(bins) * n_genes)
    bincount_reshaped = bincounts.reshape((n_genes, len(bins)))
    bins_chosen = torch.argsort(bincount_reshaped, 1, descending = True)[:, :n_components]

    locs = bins_mid[bins_chosen]
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

class Mixture(torch.nn.Module):
    """
    Note: the log prob is calculated within [0, 1] to normalize the gradients
    
    """
    def __init__(self, n_components, n_genes, window, loc_init = None, scale_init = None, logit_init = None, debug = False):
        super().__init__()
        self.n_components = n_components
        self.register_buffer("a", torch.tensor(window[0]).unsqueeze(-1))
        self.register_buffer("b", torch.tensor(window[1]).unsqueeze(-1))
        self.register_buffer("ab", self.b - self.a)
        
        self.loc = EmbeddingTensor(n_genes, (n_components, ), sparse = True)
        if loc_init is None:
            loc_init = torch.linspace(0, 1, n_components+2)[1:-1]
        self.loc.weight.data[:, :] = torch.special.logit((loc_init - self.a) / self.ab)

        if self.loc.weight.data.isnan().any():
            raise ValueError("Some locs are outside of bounds: ", torch.where(self.loc.weight.data.isnan()))

        self.register_buffer("scale_lower_bound", torch.tensor(2.) / self.ab)
        self.scale = EmbeddingTensor(n_genes, (n_components, ), sparse = True)
        if scale_init is None:
            scale_init = torch.log(self.ab/n_components)
        self.scale.weight.data[:, :] = scale_init - torch.log(self.ab)
        
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
        logit_intercept = self.logit(genes_oi)[local_gene_ix]
        logits = logit_intercept + delta_logit
        mixture_distribution = torch.distributions.Categorical(logits = logits)
        dist = torch.distributions.MixtureSameFamily(
            mixture_distribution,
            component_distribution,
            validate_args = False
        )
        
        return dist.log_prob((value - self.a)/self.ab)
    
    def parameters_dense(self):
        return []
    
    def parameters_sparse(self):
        return [self.loc.weight, self.scale.weight, self.logit.weight]

class CellGeneEmbeddingToCellEmbedding(torch.nn.Module):
    def __init__(self, n_input_features, n_genes, n_output_features = None):
        if n_output_features is None:
            n_output_features = n_input_features
            
        super().__init__()
        
        # default initialization same as a torch.nn.Linear
        self.bias1 = torch.nn.Parameter(torch.empty(n_output_features, requires_grad = True))
        self.bias1.data.zero_()
        
        self.n_output_features = n_output_features

        self.weight1 = EmbeddingTensor(n_genes, (n_input_features, self.n_output_features), sparse = True)
        stdv = 1. / math.sqrt(self.weight1.weight.size(-1))
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
    def __init__(self, n_input_features, n_latent, n_layers = 2, n_hidden_dimensions = 16):
        super().__init__()
        
        layers = []
        layers.append(torch.nn.BatchNorm1d(n_input_features))
        for i in range(n_layers):
            layers.append(torch.nn.Linear(n_hidden_dimensions if i > 0 else n_input_features, n_hidden_dimensions))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(n_hidden_dimensions))
            layers.append(torch.nn.Dropout(0.2))
        self.nn = torch.nn.Sequential(*layers)
        
        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_latent = n_latent
        
        self.linear_loc = torch.nn.Linear(n_hidden_dimensions if n_layers > 0 else n_input_features, n_latent)
        self.linear_scale = torch.nn.Linear(n_hidden_dimensions if n_layers > 0 else n_input_features, n_latent)
        
    def forward(self, x):
        nn_output = self.nn(x)
        loc = self.linear_loc(nn_output)
        scale = self.linear_scale(nn_output).exp()
        return loc, scale
    
    def parameters_dense(self):
        return self.parameters()
    
    def parameters_sparse(self):
        return []

class VAE(torch.nn.Module):
    def __init__(
        self,
        fragments,
        n_fragment_embedding_dimensions = 16,
        n_latent_dimensions = 10,
        n_components = 32,
        decoder_n_layers = 1,
        encoder_n_layers = 1,
        n_frequencies = 25
    ):
        super().__init__()
        
        self.fragment_embedder = positional_model.FragmentEmbedder(
            fragments.n_genes,
            n_frequencies=n_frequencies,
            n_embedding_dimensions=n_fragment_embedding_dimensions
        )
        self.embedding_gene_pooler = positional_model.EmbeddingGenePooler()
        
        self.cellgene_embedding_to_cell_embedding = CellGeneEmbeddingToCellEmbedding(
            self.fragment_embedder.n_embedding_dimensions,
            fragments.n_genes
        )
        
        self.n_latent_dimensions = n_latent_dimensions
        self.encoder = Encoder(
            self.cellgene_embedding_to_cell_embedding.n_output_features,
            self.n_latent_dimensions,
            n_layers = encoder_n_layers,
        )
        
        locs, logits = initialize_mixture(fragments.window, fragments.coordinates, fragments.genemapping, fragments.n_genes, n_components = n_components)
        self.mixture = Mixture(
            n_components,
            fragments.n_genes,
            fragments.window,
            loc_init = locs,
            logit_init = logits
        )
        
        self.decoder = Decoder(
            n_latent_dimensions,
            fragments.n_genes,
            self.mixture.n_components,
            n_layers = decoder_n_layers
        )
        
        libsize = torch.bincount(fragments.mapping[:, 0], minlength = fragments.n_cells)
        rho_bias = torch.bincount(fragments.mapping[:, 1], minlength = fragments.n_genes) / fragments.n_cells / libsize.to(torch.float).mean()
        
        self.register_buffer("libsize", libsize)
        self.register_buffer("rho_bias", rho_bias)
        
    def forward(self, data, track_elbo = True):
        # encode
        fragment_embedding = self.fragment_embedder.forward(data.coordinates, data.genemapping)
        cell_gene_embedding = self.embedding_gene_pooler.forward(fragment_embedding, data.local_cellxgene_ix, data.n_cells, data.n_genes)
        cell_embedding = self.cellgene_embedding_to_cell_embedding.forward(cell_gene_embedding, torch.from_numpy(data.genes_oi).to(cell_gene_embedding.device))

        loc, scale = self.encoder(cell_embedding)

        # latent
        latent_q = torch.distributions.Normal(loc, scale*0.1)
        latent_p = torch.distributions.Normal(0, 1)
        latent = latent_q.rsample()

        latent_kl = (latent_p.log_prob(latent) - latent_q.log_prob(latent)).sum()

        # decode
        genes_oi = torch.from_numpy(data.genes_oi).to(latent.device)
        logit_change, rho_change = self.decoder(latent, genes_oi)
        logit_change_cellxgene = logit_change.view(np.prod(logit_change.shape[:2]), logit_change.shape[-1])
        logit_change_fragments = logit_change_cellxgene[data.local_cellxgene_ix]

        # fragment counts
        likelihood_left = self.mixture.log_prob(data.coordinates[:, 0], logit_change_fragments, genes_oi, data.local_gene_ix)
        likelihood_right = self.mixture.log_prob(data.coordinates[:, 1], logit_change_fragments, genes_oi, data.local_gene_ix)

        likelihood_loci = (likelihood_left.sum() + likelihood_right.sum())

        # expression
        fragmentexpression = (self.rho_bias[data.genes_oi] * torch.exp(rho_change)) * self.libsize[data.cells_oi].unsqueeze(1)
        fragmentcounts_p = torch.distributions.Poisson(fragmentexpression)
        fragmentcounts = torch.reshape(torch.bincount(data.local_cellxgene_ix, minlength = data.n_cells * data.n_genes), (data.n_cells, data.n_genes))
        likelihood_fragmentcounts = fragmentcounts_p.log_prob(fragmentcounts)

        # likelihood
        likelihood = likelihood_loci.sum() + likelihood_fragmentcounts.sum()

        # ELBO
        elbo = -likelihood - latent_kl
        
        return elbo

    def evaluate_latent(self, data):
        fragment_embedding = self.fragment_embedder.forward(data.coordinates, data.genemapping)

        cell_gene_embedding = self.embedding_gene_pooler.forward(fragment_embedding, data.local_cellxgene_ix, data.n_cells, data.n_genes)

        cell_embedding = self.cellgene_embedding_to_cell_embedding.forward(cell_gene_embedding, torch.from_numpy(data.genes_oi).to(cell_gene_embedding.device))

        loc, scale = self.encoder(cell_embedding)

        return loc

    def evaluate_pseudo(self, pseudocoordinates, pseudolatent, gene_oi):
        local_cellxgene_ix = torch.arange(len(pseudocoordinates)).to(pseudocoordinates.device)
        local_gene_ix = torch.zeros(len(pseudocoordinates), dtype = torch.long).to(pseudocoordinates.device)
        # decode
        genes_oi = torch.tensor([gene_oi], dtype = torch.long, device = pseudocoordinates.device)

        spline, rho_change = self.decoder(pseudolatent, genes_oi)

        # fragment counts
        spline_cellxgene = spline.view(np.prod(spline.shape[:2]), spline.shape[-1])
        spline_fragments = spline_cellxgene[local_cellxgene_ix]

        # fragment counts
        likelihood_left = self.mixture.log_prob(pseudocoordinates, spline_fragments, genes_oi, local_gene_ix)

        return likelihood_left

    def parameters_dense(self):
        return [parameter for module in self._modules.values() for parameter in (module.parameters_dense() if hasattr(module, "parameters_dense") else module.parameters())]
    
    def parameters_sparse(self):
        return [parameter for module in self._modules.values() for parameter in (module.parameters_sparse() if hasattr(module, "parameters_sparse") else [])]



class Decoding(torch.nn.Module):
    def __init__(self, fragments, cell_latent_space, n_components = 32, decoder_n_layers = 2):
        super().__init__()

        n_latent_dimensions = cell_latent_space.shape[1]
        
        locs, logits = initialize_mixture(fragments.window, fragments.coordinates, fragments.genemapping, fragments.n_genes, n_components = n_components)
        self.mixture = Mixture(
            n_components,
            fragments.n_genes,
            fragments.window,
            loc_init = locs,
            logit_init = logits
        )
        
        self.decoder = Decoder(
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
        
    def forward(self, data, latent = None, track_elbo = True):
        if latent is None:
            latent = self.cell_latent_space[data.cells_oi]

        # decode
        genes_oi = torch.from_numpy(data.genes_oi).to(latent.device)
        logit, rho_change = self.decoder(latent, genes_oi)

        # fragment counts
        logit_cellxgene = logit.view(np.prod(logit.shape[:2]), logit.shape[-1])
        logit_change = logit_cellxgene[data.local_cellxgene_ix]

        likelihood_left = self.mixture.log_prob(data.coordinates[:, 0], logit_change, genes_oi, data.local_gene_ix)
        likelihood_right = self.mixture.log_prob(data.coordinates[:, 1], logit_change, genes_oi, data.local_gene_ix)

        likelihood_loci = (likelihood_left.sum() + likelihood_right.sum())

        # expression
        fragmentexpression = (self.rho_bias[data.genes_oi] * torch.exp(rho_change)) * self.libsize[data.cells_oi].unsqueeze(1)
        fragmentcounts_p = torch.distributions.Poisson(fragmentexpression)
        fragmentcounts = torch.reshape(torch.bincount(data.local_cellxgene_ix, minlength = data.n_cells * data.n_genes), (data.n_cells, data.n_genes))
        likelihood_fragmentcounts = fragmentcounts_p.log_prob(fragmentcounts)

        # likelihood
        likelihood = likelihood_loci.sum() + likelihood_fragmentcounts.sum()

        # ELBO
        elbo = -likelihood
        
        return elbo

    def parameters_dense(self):
        return [parameter for module in self._modules.values() for parameter in (module.parameters_dense() if hasattr(module, "parameters_dense") else module.parameters())]
    
    def parameters_sparse(self):
        return [parameter for module in self._modules.values() for parameter in (module.parameters_sparse() if hasattr(module, "parameters_sparse") else [])]

    def evaluate_pseudo(self, pseudocoordinates, pseudolatent, gene_oi):
        local_cellxgene_ix = torch.arange(len(pseudocoordinates)).to(pseudocoordinates.device)
        local_gene_ix = torch.zeros(len(pseudocoordinates), dtype = torch.long).to(pseudocoordinates.device)
        # decode
        genes_oi = torch.tensor([gene_oi], dtype = torch.long, device = pseudocoordinates.device)

        spline, rho_change = self.decoder(pseudolatent, genes_oi)

        # fragment counts
        spline_cellxgene = spline.view(np.prod(spline.shape[:2]), spline.shape[-1])
        spline_fragments = spline_cellxgene[local_cellxgene_ix]

        # fragment counts
        likelihood_left = self.mixture.log_prob(pseudocoordinates, spline_fragments, genes_oi, local_gene_ix)

        return likelihood_left
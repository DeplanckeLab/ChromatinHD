import torch
import numpy as np

from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.models import HybridModel


class LogfoldPredictor(torch.nn.Module):
    def __init__(self, n_clusters: int, n_variantxgenes: int):
        super().__init__()

        self.variantxgene_cluster_effect = EmbeddingTensor(
            n_variantxgenes, (n_clusters,), sparse=True
        )
        self.variantxgene_cluster_effect.data[:] = 0.0

    def forward(self, variantxgene_ixs):
        # fc_log [cluster, variantxgene]
        fc_log = self.variantxgene_cluster_effect[variantxgene_ixs].transpose(1, 0)
        self.elbo = -torch.distributions.Normal(0.0, 0.1).log_prob(fc_log)
        return fc_log

    def get_elbo(self):
        return self.elbo

    def parameters_sparse(self):
        return [self.variantxgene_cluster_effect.weight]


class NegativeBinomial2(torch.distributions.NegativeBinomial):
    def __init__(self, mu, dispersion, eps=1e-8):
        # avoids NaNs induced in gradients when mu is very low
        dispersion = torch.clamp_max(dispersion, 20.0)
        logits = (mu + eps).log() - (1 / dispersion + eps).log()

        total_count = 1 / dispersion

        super().__init__(total_count=total_count, logits=logits)


class ExpressionPredictor(torch.nn.Module):
    elbo = None

    def __init__(
        self,
        n_genes: int,
        n_clusters: int,
        n_donors: int,
        lib: torch.Tensor,
        baseline: torch.Tensor,
    ):
        super().__init__()

        self.dispersion_log = torch.nn.Parameter(torch.zeros(n_clusters, n_genes))

        assert lib.shape == (n_donors, n_clusters)
        self.register_buffer("lib", lib)

        assert baseline.shape == (n_clusters, n_genes)
        baseline_log = torch.log(baseline.clone())
        baseline_log.requires_grad = True
        self.baseline_log = torch.nn.Parameter(baseline_log)

    def forward(
        self,
        fc_log,
        genotypes,
        expression_obs,
        variantxgene_to_gene,
        local_variant_to_local_variantxgene_selector,
        variantxgene_to_local_gene,
    ):
        self.track = {}

        # genotype [donor, variant] -> [donor, variantxgene]
        genotypes = genotypes[:, local_variant_to_local_variantxgene_selector]

        # genotype [donor, variantxgene] -> [donor, (cluster), variantxgene]
        # fc_log [cluster, variantxgene] -> [(donor), cluster, variantxgene]
        expression_delta = genotypes.unsqueeze(1) * fc_log.unsqueeze(0)

        # expression_delta[:] = 0.

        # baseline [cluster, gene] -> [(donor), cluster, variantxgene]
        # expression = [donor, cluster, variantxgene]
        expression_log = self.baseline_log[:, variantxgene_to_gene] + expression_delta
        expression = torch.exp(expression_log)

        # lib [donor, cluster] -> [donor, cluster, (variantxgene)]
        expressed = expression * self.lib.unsqueeze(-1)

        if expressed.isnan().any():
            raise ValueError("`expressed` contains NaNs")

        # dispersion [cluster, gene] -> [cluster, variantxgene]
        dispersion = torch.exp(self.dispersion_log)[:, variantxgene_to_gene]

        expression_dist = NegativeBinomial2(expressed, dispersion)

        # expression_obs [donor, cluster, gene] -> [donor, cluster, variantxgene]
        expression_obs = expression_obs[:, :, variantxgene_to_local_gene]

        # expression_obs [donor, cluster, variantxgene]
        expression_likelihood = expression_dist.log_prob(expression_obs)

        elbo = -expression_likelihood

        self.track.update(locals())

        self.elbo = elbo

        return expressed

    def get_elbo(self):
        return self.elbo


class Model(torch.nn.Module, HybridModel):
    def __init__(
        self, n_genes, n_clusters, n_variantxgenes, n_donors, lib, baseline, dummy=False
    ):
        super().__init__()
        self.n_variantxgenes = n_variantxgenes
        self.fc_log_predictor = LogfoldPredictor(n_clusters, n_variantxgenes)
        self.expression_predictor = ExpressionPredictor(
            n_genes, n_clusters, n_donors, lib, baseline
        )
        self.dummy = dummy

    @classmethod
    def create(cls, transcriptome, genotype, gene_variants_mapping, **kwargs):
        """
        Creates the model using the data, i.e. transcriptome, genotype and gene_variants_mapping
        """
        n_variantxgenes = sum(len(variants) for variants in gene_variants_mapping)

        lib = transcriptome.X.sum(-1).astype(np.float32)
        lib_torch = torch.from_numpy(lib)

        baseline = (transcriptome.X / np.expand_dims(lib + 1e-8, -1)).mean(0)
        baseline_torch = torch.from_numpy(baseline)
        return cls(
            n_genes=len(transcriptome.var),
            n_clusters=len(transcriptome.clusters_info),
            n_variantxgenes=n_variantxgenes,
            n_donors=len(transcriptome.donors_info),
            lib=lib_torch,
            baseline=baseline_torch,
            **kwargs
        )

    def forward(self, data):
        fc_log = self.fc_log_predictor(
            data.variantxgene_ixs,
        )
        if self.dummy:
            fc_log[:] = 0.0
        expression = self.expression_predictor(
            fc_log,
            data.genotypes,
            data.expression,
            data.variantxgene_to_gene,
            data.local_variant_to_local_variantxgene_selector,
            data.variantxgene_to_local_gene,
        )

        elbo = (
            self.expression_predictor.get_elbo().sum()
            + self.fc_log_predictor.get_elbo().sum()
        )

        return elbo

    def get_full_elbo(self):
        return self.expression_predictor.get_elbo()

import torch

from chromatinhd.embedding import EmbeddingTensor


class LogfoldPredictor(torch.nn.Module):
    def __init__(self, n_clusters: int, n_variantxgenes: int):
        super().__init__()

        self.variantxgene_cluster_effect = EmbeddingTensor(
            n_variantxgenes, (n_clusters,)
        )
        self.variantxgene_cluster_effect.data[:] = 0.0

    def forward(self, variantxgene_ixs):
        fc_log = self.variantxgene_cluster_effect[variantxgene_ixs]
        self.elbo = torch.distributions.Normal(0.0, 0.1).log_prob(fc_log)
        return fc_log

    def get_elbo(self):
        return self.elbo


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

    def forward(self, fc_log, variantxgene_to_gene, genotypes, expression_obs):
        self.track = {}

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

        # expression_obs [donor, cluster, variantxgene]
        expression_likelihood = expression_dist.log_prob(expression_obs)

        elbo = -expression_likelihood.sum()

        self.track.update(locals())

        self.elbo = elbo

        return

    def get_elbo(self):
        return self.elbo


class Model(torch.nn.Module):
    def __init__(self, n_genes, n_clusters, n_variantxgenes, n_donors, lib, baseline):
        self.fc_log_predictor = LogfoldPredictor(n_clusters, n_variantxgenes)
        self.expression_predictor = ExpressionPredictor(
            n_genes, n_clusters, n_donors, lib, baseline
        )

    def forward(self, data):
        fc_log = self.fc_log_predictor(
            data.variantxgene_ixs,
        )
        expression = self.expression_predictor(
            fc_log,
            data.variantxgene_to_gene,
            data.genotypes,
            data.expression,
        )

        elbo = -self.expression_predictor.get_elbo() - self.fc_log_predictor.get_elbo()

        return elbo

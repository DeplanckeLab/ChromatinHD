import torch


class NegativeBinomial2(torch.distributions.NegativeBinomial):
    """
    This is the NegativeBinomial2 parameterization within ProbOnto v2.5
    """

    def __init__(self, mu, dispersion, eps=1e-8):
        # avoids NaNs induced in gradients when mu is very low
        dispersion = torch.clamp_max(dispersion, 20.0)
        logits = (mu + eps).log() - (1 / dispersion + eps).log()

        total_count = 1 / dispersion

        super().__init__(total_count=total_count, logits=logits)


class Model(torch.nn.Module):
    def __init__(
        self,
        n_clusters: int,
        n_donors: int,
        n_variants: int,
        lib: torch.Tensor,
        baseline: torch.Tensor,
        apply_genotype_effect=True,
    ):
        super().__init__()

        self.dispersion_log = torch.nn.Parameter(torch.zeros(n_clusters))
        self.fc_log_mu = torch.nn.Parameter(torch.zeros((n_clusters, n_variants)))

        assert lib.shape == (n_donors, n_clusters)
        self.lib = lib

        assert baseline.shape == (n_clusters,)
        baseline_log = torch.log(baseline.clone())
        baseline_log.requires_grad = True
        self.baseline_log = torch.nn.Parameter(baseline_log)

        self.apply_genotype_effect = apply_genotype_effect

        self.track = {}

    def forward(self, genotype, expression_obs):
        self.track = {}

        fc_log = self.fc_log_mu

        fc_log_likelihood = torch.distributions.Normal(0.0, 0.1).log_prob(fc_log)

        if not self.apply_genotype_effect:
            fc_log = torch.zeros_like(fc_log)

        # genotype [donor, variant] -> [donor, cluster, variant]
        # fc_log [cluster, variant] -> [donor, cluster, variant]
        expression_delta = genotype.unsqueeze(1) * fc_log.unsqueeze(0)

        # baseline [cluster] -> [(donor), cluster, variant]
        expression_log = self.baseline_log.unsqueeze(-1) + expression_delta
        expression = torch.exp(expression_log)

        # lib [donor, cluster] -> [donor, cluster, variant]
        expressed = expression * self.lib.unsqueeze(-1)

        # dispersion [cluster] -> [cluster, variant]
        dispersion = torch.exp(self.dispersion_log).unsqueeze(-1)

        # expression_dist = torch.distributions.Poisson(expressed)  # much faster 🤤
        expression_dist = NegativeBinomial2(expressed, dispersion)

        # expression_obs [cluster, donor] -> [cluster, donor, variant]
        expression_likelihood = expression_dist.log_prob(expression_obs.unsqueeze(-1))

        elbo = -expression_likelihood.sum() - fc_log_likelihood.sum()

        self.track.update(locals())

        return elbo

    def get_likelihood(self):
        return self.track["expression_likelihood"]

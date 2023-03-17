import torch
import numpy as np

from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.models import HybridModel
from .effect_predictors import EffectPredictor
from . import cut_embedders
from . import cut_counters

from .loader import Data


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
        baseline_log: torch.Tensor,
        dispersion_log: torch.Tensor,
    ):
        super().__init__()

        self.register_buffer("dispersion_log", dispersion_log.clone())

        assert lib.shape == (n_donors, n_clusters)
        self.register_buffer("lib", lib)

        self.register_buffer("baseline_log", baseline_log.clone())

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
        self,
        n_genes,
        n_clusters,
        n_variantxgenes,
        n_donors,
        lib,
        variantxgene_effect,
        cluster_cut_lib,
        dispersion_log,
        baseline_log,
        window_size,
        ground_truth_variantxgene_effect=None,
        ground_truth_significant=None,
        ground_truth_prioritization=None,
        dummy=False,
        dumb=False,
    ):
        super().__init__()
        self.n_variantxgenes = n_variantxgenes

        self.variant_embedder = cut_counters.VariantEmbedder(
            window_size=window_size,
            cluster_cut_lib=cluster_cut_lib,
            dummy=dummy,
            dumb=dumb,
        )
        self.fc_log_predictor = EffectPredictor(
            self.variant_embedder.n_embedding_dimensions + 2,
            n_variantxgenes,
            variantxgene_effect=variantxgene_effect,
            n_layers=0,
        )
        self.expression_predictor = ExpressionPredictor(
            n_genes,
            n_clusters,
            n_donors,
            lib,
            baseline_log=baseline_log,
            dispersion_log=dispersion_log,
        )
        # self.dummy = dummy
        self.register_buffer(
            "ground_truth_variantxgene_effect", ground_truth_variantxgene_effect
        )

        self.register_buffer("ground_truth_significant", ground_truth_significant)
        self.register_buffer("ground_truth_prioritization", ground_truth_prioritization)

    @classmethod
    def create(
        cls,
        transcriptome,
        genotype,
        fragments,
        gene_variants_mapping,
        variantxgene_effect,
        reference_expression_predictor,
        **kwargs,
    ):
        """
        Creates the model using the data, i.e. transcriptome, genotype and gene_variants_mapping
        """
        n_variantxgenes = sum(len(variants) for variants in gene_variants_mapping)

        lib = transcriptome.X.sum(-1).astype(np.float32)
        lib_torch = torch.from_numpy(lib)

        cluster_cut_lib = torch.bincount(
            fragments.clusters, minlength=len(fragments.clusters_info)
        )

        dispersion_log = reference_expression_predictor.dispersion_log.detach().cpu()
        baseline_log = reference_expression_predictor.baseline_log.detach().cpu()
        return cls(
            n_genes=len(transcriptome.var),
            n_clusters=len(transcriptome.clusters_info),
            n_variantxgenes=n_variantxgenes,
            n_donors=len(transcriptome.donors_info),
            lib=lib_torch,
            variantxgene_effect=torch.from_numpy(variantxgene_effect.values),
            cluster_cut_lib=cluster_cut_lib,
            dispersion_log=dispersion_log,
            baseline_log=baseline_log,
            **kwargs,
        )

    def forward(self, data: Data):
        # embed variants
        variant_embedding = self.variant_embedder(
            data.relative_coordinates,
            local_clusterxvariant_indptr=data.local_clusterxvariant_indptr,
            n_variants=data.n_variants,
            n_clusters=data.n_clusters,
        )

        # variant_embedding [cluster, variant] -> [cluster, variantxgene]
        variantxgene_embedding = variant_embedding[
            :, data.local_variant_to_local_variantxgene_selector
        ]

        variantxgene_tss_distances = data.variantxgene_tss_distances.repeat(
            data.n_clusters, 1
        ).unsqueeze(-1)

        mean_expression = (
            data.expression.mean(0)[:, data.variantxgene_to_local_gene].unsqueeze(-1)
        ) / self.expression_predictor.lib.mean(0).unsqueeze(-1).unsqueeze(-1)
        # print(self.expression_predictor.lib.shape)

        variantxgene_embedding = torch.concat(
            [
                variantxgene_embedding,
                variantxgene_tss_distances / (10**5),
                mean_expression,
            ],
            -1,
        )

        fc_log, prioritization = self.fc_log_predictor(
            variantxgene_embedding,
            data.variantxgene_ixs,
        )

        ground_truth_prioritization = self.ground_truth_prioritization[
            :, data.variantxgene_ixs
        ]

        return torch.nn.MSELoss()(
            prioritization.squeeze(-1),
            ground_truth_prioritization,
        )

        # ground_truth_significant = self.ground_truth_significant[
        #     :, data.variantxgene_ixs
        # ]

        # return torch.nn.CrossEntropyLoss()(
        #     torch.exp(prioritization).squeeze(-1),
        #     ground_truth_significant,
        # )

        expression = self.expression_predictor(
            fc_log,
            data.genotypes,
            data.expression,
            data.variantxgene_to_gene,
            data.local_variant_to_local_variantxgene_selector,
            data.variantxgene_to_local_gene,
        )

        ground_truth_variantxgene_effect = self.ground_truth_variantxgene_effect[
            :, data.variantxgene_ixs
        ]

        return ((ground_truth_variantxgene_effect - fc_log) ** 2).sum()

        elbo = self.expression_predictor.get_elbo().sum()

        return elbo

    def get_full_elbo(self):
        return self.expression_predictor.get_elbo()

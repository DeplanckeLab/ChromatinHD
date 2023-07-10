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
        dummy=False,
    ):
        super().__init__()
        self.n_variantxgenes = n_variantxgenes

        if dummy:
            self.cut_embedder = cut_embedders.CutEmbedderDummy()
        else:
            # self.cut_embedder = cut_embedders.CutEmbedderPositional(
            #     window_size=window_size
            # )
            self.cut_embedder = cut_embedders.CutEmbedderBins(window_size=window_size)
        self.variant_embedder = cut_embedders.VariantEmbedder(
            cluster_cut_lib=cluster_cut_lib,
        )
        self.fc_log_predictor = EffectPredictor(
            self.cut_embedder.n_embedding_dimensions * 2 + 1,
            n_variantxgenes,
            variantxgene_effect=variantxgene_effect,
            n_layers=1,
        )
        self.expression_predictor = ExpressionPredictor(
            n_genes,
            n_clusters,
            n_donors,
            lib,
            baseline_log=baseline_log,
            dispersion_log=dispersion_log,
        )
        self.dummy = dummy
        self.register_buffer(
            "ground_truth_variantxgene_effect", ground_truth_variantxgene_effect
        )

        self.register_buffer("ground_truth_significant", ground_truth_significant)

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
        cut_embedding = self.cut_embedder(data.relative_coordinates)
        variant_embedding = self.variant_embedder(
            cut_embedding,
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

        variantxgene_embedding = torch.concat(
            [variantxgene_embedding, variantxgene_tss_distances / (10**5)], -1
        )

        fc_log, prioritization = self.fc_log_predictor(
            variantxgene_embedding,
            data.variantxgene_ixs,
        )

        # ground_truth_significant = self.ground_truth_significant[
        #     :, data.variantxgene_ixs
        # ]

        # return torch.nn.CrossEntropyLoss()(
        #     prioritization[
        #         :, data.local_variant_to_local_variantxgene_selector
        #     ].squeeze(-1),
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

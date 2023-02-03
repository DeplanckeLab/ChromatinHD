import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import chromatinhd as chd


class Scorer:
    def __init__(self, models, folds, loaders, gene_ids, cell_ids, device="cuda"):
        assert len(models) == len(folds)
        self.models = models
        self.device = device
        self.folds = folds
        self.loaders = loaders
        self.gene_ids = gene_ids
        self.cell_ids = gene_ids

    def score(
        self,
        loader_kwargs=None,
        return_prediction=False,
        transcriptome_predicted_full=None,
    ):
        """
        gene_ids: mapping of gene ix to gene id
        """

        likelihoods = {}

        if loader_kwargs is None:
            loader_kwargs = {}

        next_task_sets = []
        for fold in self.folds:
            next_task_sets.append({"tasks": fold["minibatches"]})
        next_task_sets[0]["loader_kwargs"] = loader_kwargs
        self.loaders.initialize(next_task_sets=next_task_sets)

        scores = []
        genescores = []
        for model_ix, (model, fold) in tqdm.tqdm(
            enumerate(zip(self.models, self.folds)), leave=False, total=len(self.models)
        ):
            # infer and score
            likelihood = np.zeros((len(self.cell_ids), len(self.gene_ids)))
            likelihoods[model_ix] = likelihood

            with torch.no_grad():
                # infer
                model = model.to(self.device)
                # for data in tqdm.tqdm(self.loaders):
                for data in self.loaders:
                    data = data.to(self.device)
                    likelihood_ = model.forward_likelihood(data)

                    self.loaders.submit_next()

                    likelihood[np.ix_(data.cells_oi, data.genes_oi)] = (
                        likelihood_.detach().cpu().numpy()
                    )

            # score
            for phase, (cells, genes) in fold["phases"].items():
                likelihood_total = likelihood[np.ix_(cells, genes)].sum()
                likelihood_gene = likelihood[np.ix_(cells, genes)].sum(0)
                score = {
                    "model": [model_ix],
                    "phase": [phase],
                    "likelihood": [likelihood_total],
                }
                genescore = pd.DataFrame(
                    {
                        "model": model_ix,
                        "phase": phase,
                        "likelihood": likelihood_gene,
                        "gene": self.gene_ids[genes].values,
                    }
                )

                scores.append(score)
                genescores.append(genescore)

        score_cols = ["likelihood"]
        if transcriptome_predicted_full is not None:
            score_cols.append("effect")

        genescores_across = pd.concat(genescores)
        scores = pd.concat(genescores)
        genescores = genescores_across.groupby(["phase", "gene"])[score_cols].mean()
        scores = scores.groupby(["phase"])[score_cols].mean()

        return scores, genescores

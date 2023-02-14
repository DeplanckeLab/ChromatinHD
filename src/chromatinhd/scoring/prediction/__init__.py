import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import chromatinhd as chd


class Scorer:
    def __init__(self, models, folds, loaders, outcome, gene_ids, device="cuda"):
        assert len(models) == len(folds)
        self.outcome = outcome
        self.models = models
        self.device = device
        self.folds = folds
        self.loaders = loaders
        self.gene_ids = gene_ids

    def score(
        self,
        loader_kwargs=None,
        return_prediction=False,
        transcriptome_predicted_full=None,
    ):
        """
        gene_ids: mapping of gene ix to gene id
        """

        transcriptome_predicted = {}
        n_fragments = {}

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
            # create transcriptome_predicted
            transcriptome_predicted_ = np.zeros(self.outcome.shape)
            transcriptome_predicted[model_ix] = transcriptome_predicted_

            n_fragments = pd.Series(np.zeros(self.outcome.shape[1]))
            n_fragments.index.name = "gene_ix"

            # infer and score

            with torch.no_grad():
                # infer
                model = model.to(self.device)
                # for data in tqdm.tqdm(self.loaders):
                for data in self.loaders:
                    data = data.to(self.device)
                    predicted = model(data)

                    self.loaders.submit_next()

                    # if (transcriptome_predicted[model_ix][data.cells_oi, data.genes_oi] != 0).any():
                    #     raise ValueError("Double booked cells and/or genes")

                    transcriptome_predicted_[np.ix_(data.cells_oi, data.genes_oi)] = (
                        predicted.detach().cpu().numpy()
                    )
                    # print(transcriptome_predicted[model_ix][data.cells_oi, :][:, data.genes_oi])

                # score
                for phase, (cells, genes) in fold["phases"].items():
                    outcome_phase = self.outcome.numpy()[np.ix_(cells, genes)]
                    # outcome_phase_dummy = outcome_phase.mean(0, keepdims=True).repeat(
                    #     outcome_phase.shape[0], 0
                    # )

                    transcriptome_predicted__ = transcriptome_predicted_[
                        np.ix_(cells, genes)
                    ]

                    cor_gene = chd.utils.paircor(
                        outcome_phase, transcriptome_predicted__
                    )

                    # def zscore(x, dim=0):
                    #     return (x - x.mean(dim, keepdims=True)) / (
                    #         x.std(dim, keepdims=True)
                    #     )

                    # input = transcriptome_predicted__
                    # target = outcome_phase
                    # eps = 0.1
                    # dim = 0
                    # input_normalized = (
                    #     input
                    #     - input.mean(dim, keepdims=True)
                    #     + target.mean(dim, keepdims=True)
                    # ) * (
                    #     target.std(dim, keepdims=True)
                    #     / (input.std(dim, keepdims=True) + eps)
                    # )

                    # rmse = -np.mean((input_normalized - target) ** 2, 0)
                    # mse = np.mean((input - target) ** 2, 0) * 100

                    score = {
                        "model": [model_ix],
                        "phase": [phase],
                        "cor": [cor_gene.mean()],
                        # "rmse": [rmse.mean()],
                        # "mse": [mse.mean()],
                    }
                    genescore = pd.DataFrame(
                        {
                            "model": model_ix,
                            "phase": phase,
                            "cor": cor_gene,
                            "gene": self.gene_ids[genes].values,
                            # "rmse": rmse,
                            # "mse": mse,
                        }
                    )

                    # check effect if desired
                    if transcriptome_predicted_full is not None:
                        effect = (
                            transcriptome_predicted__
                            - transcriptome_predicted_full[model_ix][
                                np.ix_(cells, genes)
                            ]
                        ).mean(0)
                        score["effect"] = effect.mean()
                        genescore["effect"] = effect

                    scores.append(score)
                    genescores.append(genescore)

        score_cols = [
            "cor",
            # "rmse",
            # "mse",
        ]
        if transcriptome_predicted_full is not None:
            score_cols.append("effect")

        genescores_across = pd.concat(genescores)
        scores = pd.concat(genescores)
        genescores = genescores_across.groupby(["phase", "gene"])[score_cols].mean()
        scores = scores.groupby(["phase"])[score_cols].mean()

        if return_prediction:
            return transcriptome_predicted, scores, genescores
        return scores, genescores

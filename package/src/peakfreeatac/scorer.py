import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import peakfreeatac as pfa


class Scorer:
    def __init__(self, models, folds, loaders, outcome, device="cuda"):
        assert len(models) == len(folds)
        self.outcome = outcome
        self.models = models
        self.device = device
        self.folds = folds
        self.loaders = loaders
        
        self.reset()

    def reset(self):
        self.transcriptome_predicted = {}
        self.n_fragments = {}
        for model_ix in range(len(self.models)):
            transcriptome_predicted = pd.DataFrame(np.zeros(self.outcome.shape))
            transcriptome_predicted.index.name = "cell_ix"
            transcriptome_predicted.columns.name = "gene_ix"

            self.transcriptome_predicted[model_ix] = transcriptome_predicted

            n_fragments = pd.Series(np.zeros(self.outcome.shape[1]))
            n_fragments.index.name = "gene_ix"

    def infer(self, fragments_oi=None):
        for model_ix, (model, fold) in enumerate(zip(self.models, self.folds)):
            loader_kwargs = {}
            if fragments_oi is not None:
                loader_kwargs["fragments_oi"] = fragments_oi

            self.loaders.initialize(fold["minibatches"])
            self.loaders.restart(**loader_kwargs)
            assert self.loaders.n_done == 0

            def cell_gene_to_cellxgene(cells_oi, genes_oi, n_genes):
                return (cells_oi[:, None] * n_genes + genes_oi).flatten()

            # infer and store
            with torch.no_grad():
                model = model.to(self.device)
                done = set()
                for data in tqdm.tqdm(self.loaders):
                    cellxgene_oi = set(cell_gene_to_cellxgene(data.cells_oi, data.genes_oi, 5000))
                    intersect = done.intersection(cellxgene_oi)
                    if len(intersect) > 0:
                        print(len(intersect))
                    done.update(cellxgene_oi)

                    data = data.to(self.device)
                    transcriptome_predicted = model(data)

                    if (self.transcriptome_predicted[model_ix].iloc[data.cells_oi, data.genes_oi].values != 0).any():
                        # print(data.cells_oi, data.genes_oi, len(data.genes_oi), len(data.cells_oi))
                        raise ValueError("Double booked cells and/or genes")

                    self.transcriptome_predicted[model_ix].iloc[data.cells_oi, data.genes_oi] = (
                        transcriptome_predicted.detach().cpu().numpy()
                    )

                    # self.n_fragments[model_ix].iloc[data.genes_oi] = 

                    self.loaders.submit_next()

    def score(self, gene_ids):
        scores = []
        genescores = []

        for model_ix, fold in zip(range(len(self.models)), self.folds):
            for phase, (cells, genes) in fold["phases"].items():
                outcome_phase = self.outcome.numpy()[cells, :][:, genes]
                outcome_phase_dummy = outcome_phase.mean(0, keepdims=True).repeat(
                    outcome_phase.shape[0], 0
                )

                transcriptome_predicted = self.transcriptome_predicted[model_ix]

                cor_gene = pfa.utils.paircor(
                    outcome_phase, transcriptome_predicted.values[cells, :][:, genes]
                )
                cos_gene = pfa.utils.paircos(
                    outcome_phase, transcriptome_predicted.values[cells, :][:, genes]
                )

                def zscore(x, dim = 0):
                    return (x - x.mean(dim, keepdims = True)) / (x.std(dim, keepdims = True))

                mse = -np.mean((zscore(transcriptome_predicted.values[cells, :][:, genes]) - zscore(outcome_phase))**2, 0)

                scores.append(
                    pd.DataFrame(
                        {
                            "model":[model_ix],
                            "phase": [phase],
                            "cor": [cor_gene.mean()],
                            "cos": [cos_gene.mean()],
                            "mse":[mse.mean()]
                        }
                    )
                )
                genescores.append(
                    pd.DataFrame(
                        {
                            "model": model_ix,
                            "phase": phase,
                            "cor": cor_gene,
                            "cos": cos_gene,
                            "gene": gene_ids[genes].values,
                            "mse":mse
                        }
                    )
                )


        score_cols = ["cor", "cos", "mse"]

        genescores_across = pd.concat(genescores)
        scores = pd.concat(genescores)
        genescores = genescores_across.groupby(["phase", "gene"])[score_cols].mean()
        scores = scores.groupby(["phase"])[score_cols].mean()

        return scores, genescores

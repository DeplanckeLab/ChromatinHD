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
        for model_ix in range(len(self.models)):
            transcriptome_predicted = pd.DataFrame(np.zeros(self.outcome.shape))
            transcriptome_predicted.index.name = "cell_ix"
            transcriptome_predicted.columns.name = "gene_ix"

            self.transcriptome_predicted[model_ix] = transcriptome_predicted

    def infer(self, fragments_oi=None):
        for model_ix, (model, fold) in enumerate(zip(self.models, self.folds)):
            loader_kwargs = {}
            if fragments_oi is not None:
                loader_kwargs["fragments_oi"] = fragments_oi

            self.loaders.initialize(fold["minibatches"])
            self.loaders.restart(**loader_kwargs)
            assert self.loaders.n_done == 0

            # infer and store
            with torch.no_grad():
                model = model.to(self.device)
                for data in tqdm.tqdm(self.loaders):
                    data = data.to(self.device)
                    transcriptome_predicted = model(data)

                    self.transcriptome_predicted[model_ix].iloc[data.cells_oi, data.genes_oi] = (
                        transcriptome_predicted.detach().cpu().numpy()
                    )

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
                cos_gene_dummy = pfa.utils.paircos(outcome_phase, outcome_phase_dummy)

                cos_diff = cos_gene - cos_gene_dummy
                cos_diff[cos_gene == 0] = 0.0

                scores.append(
                    pd.DataFrame(
                        {
                            "model":[model_ix],
                            "phase": [phase],
                            "cor": [cor_gene.mean()],
                            "cos": [cos_gene.mean()],
                            "cos_dummy": [cos_gene_dummy.mean()],
                            "cos_diff": [cos_diff.mean()],
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
                            "cos_dummy": cos_gene_dummy,
                            "cos_diff": cos_diff,
                        }
                    )
                )
        genescores = pd.concat(genescores).set_index(["model", "phase", "gene"]).groupby(["phase", "gene"]).mean()
        scores = pd.concat(scores).set_index(["model", "phase"]).groupby(["phase"]).mean()

        return scores, genescores

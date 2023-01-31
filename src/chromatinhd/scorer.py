import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import peakfreeatac as pfa


# class Scorer:
#     def __init__(self, models, folds, loaders, outcome, device="cuda"):
#         assert len(models) == len(folds)
#         self.outcome = outcome
#         self.models = models
#         self.device = device
#         self.folds = folds
#         self.loaders = loaders
        
#         self.reset() 

#     def reset(self):
#         self.transcriptome_predicted = {}
#         self.n_fragments = {}
#         for model_ix in range(len(self.models)):
#             transcriptome_predicted = pd.DataFrame(np.zeros(self.outcome.shape))
#             transcriptome_predicted.index.name = "cell_ix"
#             transcriptome_predicted.columns.name = "gene_ix"

#             self.transcriptome_predicted[model_ix] = transcriptome_predicted

#             n_fragments = pd.Series(np.zeros(self.outcome.shape[1]))
#             n_fragments.index.name = "gene_ix"

#     def infer(self, fragments_oi=None):
#         for model_ix, (model, fold) in enumerate(zip(self.models, self.folds)):
#             loader_kwargs = {}
#             if fragments_oi is not None:
#                 loader_kwargs["fragments_oi"] = fragments_oi

#             self.loaders.initialize(fold["minibatches"])
#             self.loaders.restart(**loader_kwargs)
#             assert self.loaders.n_done == 0

#             # def cell_gene_to_cellxgene(cells_oi, genes_oi, n_genes):
#             #     return (cells_oi[:, None] * n_genes + genes_oi).flatten()

#             # infer and store
#             with torch.no_grad():
#                 model = model.to(self.device)
#                 # done = set()
#                 for data in tqdm.tqdm(self.loaders):
#                     # cellxgene_oi = set(cell_gene_to_cellxgene(data.cells_oi, data.genes_oi, 5000))
#                     # intersect = done.intersection(cellxgene_oi)
#                     # if len(intersect) > 0:
#                     #     print(len(intersect))
#                     # done.update(cellxgene_oi)

#                     data = data.to(self.device)
#                     transcriptome_predicted = model(data)

#                     if (self.transcriptome_predicted[model_ix].iloc[data.cells_oi, data.genes_oi].values != 0).any():
#                         raise ValueError("Double booked cells and/or genes")

#                     self.transcriptome_predicted[model_ix].iloc[data.cells_oi, data.genes_oi] = (
#                         transcriptome_predicted.detach().cpu().numpy()
#                     )

#                     # self.n_fragments[model_ix].iloc[data.genes_oi] = 

#                     self.loaders.submit_next()

#     def score(self, gene_ids):
#         scores = []
#         genescores = []

#         for model_ix, fold in zip(range(len(self.models)), self.folds):
#             for phase, (cells, genes) in fold["phases"].items():
#                 outcome_phase = self.outcome.numpy()[cells, :][:, genes]
#                 outcome_phase_dummy = outcome_phase.mean(0, keepdims=True).repeat(
#                     outcome_phase.shape[0], 0
#                 )

#                 transcriptome_predicted = self.transcriptome_predicted[model_ix]

#                 cor_gene = pfa.utils.paircor(
#                     outcome_phase, transcriptome_predicted.values[cells, :][:, genes]
#                 )

#                 def zscore(x, dim = 0):
#                     return (x - x.mean(dim, keepdims = True)) / (x.std(dim, keepdims = True))

#                 input = transcriptome_predicted.values[cells, :][:, genes]
#                 target = outcome_phase
#                 eps = 0.1
#                 dim = 0
#                 input_normalized = (
#                     (input - input.mean(dim, keepdims = True) + target.mean(dim, keepdims = True)) * 
#                     (target.std(dim, keepdims = True) / (input.std(dim, keepdims = True) + eps))
#                 )

#                 rmse = -np.mean((input_normalized - target)**2, 0)
#                 mse = -np.mean((input - target)**2, 0)

#                 scores.append(
#                     pd.DataFrame(
#                         {
#                             "model":[model_ix],
#                             "phase": [phase],
#                             "cor": [cor_gene.mean()],
#                             "rmse":[rmse.mean()],
#                             "mse":[mse.mean()]
#                         }
#                     )
#                 )
#                 genescores.append(
#                     pd.DataFrame(
#                         {
#                             "model": model_ix,
#                             "phase": phase,
#                             "cor": cor_gene,
#                             "gene": gene_ids[genes].values,
#                             "rmse":rmse,
#                             "mse":mse
#                         }
#                     )
#                 )


#         score_cols = ["cor", "rmse", "mse"]

#         genescores_across = pd.concat(genescores)
#         scores = pd.concat(genescores)
#         genescores = genescores_across.groupby(["phase", "gene"])[score_cols].mean()
#         scores = scores.groupby(["phase"])[score_cols].mean()

#         return scores, genescores





class Scorer:
    def __init__(self, models, folds, loaders, outcome, gene_ids, device="cuda"):
        assert len(models) == len(folds)
        self.outcome = outcome
        self.models = models
        self.device = device
        self.folds = folds
        self.loaders = loaders
        self.gene_ids = gene_ids

    def score(self, loader_kwargs = None, return_prediction = False, transcriptome_predicted_full = None):
        """
        gene_ids: mapping of gene ix to gene id        
        """
        
        transcriptome_predicted = {}
        n_fragments = {}
        
        if loader_kwargs is None:
            loader_kwargs = {}
        
        next_task_sets = []
        for fold in self.folds:
            next_task_sets.append({"tasks":fold["minibatches"]})
        next_task_sets[0]["loader_kwargs"] = loader_kwargs
        self.loaders.initialize(next_task_sets = next_task_sets)
        
        scores = []
        genescores = []
        for model_ix, (model, fold) in tqdm.tqdm(enumerate(zip(self.models, self.folds)), leave = False, total = len(self.models)):
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
                for data in (self.loaders):
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
                    outcome_phase_dummy = outcome_phase.mean(0, keepdims=True).repeat(
                        outcome_phase.shape[0], 0
                    )

                    transcriptome_predicted__ = transcriptome_predicted_[np.ix_(cells, genes)]

                    cor_gene = pfa.utils.paircor(
                        outcome_phase, transcriptome_predicted__
                    )

                    def zscore(x, dim = 0):
                        return (x - x.mean(dim, keepdims = True)) / (x.std(dim, keepdims = True))

                    input = transcriptome_predicted__
                    target = outcome_phase
                    eps = 0.1
                    dim = 0
                    input_normalized = (
                        (input - input.mean(dim, keepdims = True) + target.mean(dim, keepdims = True)) * 
                        (target.std(dim, keepdims = True) / (input.std(dim, keepdims = True) + eps))
                    )

                    rmse = -np.mean((input_normalized - target)**2, 0)
                    mse = np.mean((input - target)**2, 0) * 100
                    
                    score = {
                        "model":[model_ix],
                        "phase": [phase],
                        "cor": [cor_gene.mean()],
                        "rmse":[rmse.mean()],
                        "mse":[mse.mean()]
                    }
                    genescore = pd.DataFrame({
                        "model": model_ix,
                        "phase": phase,
                        "cor": cor_gene,
                        "gene": self.gene_ids[genes].values,
                        "rmse":rmse,
                        "mse":mse
                    })
                    
                    # check effect if desired
                    if transcriptome_predicted_full is not None:
                        effect = (transcriptome_predicted__ - transcriptome_predicted_full[model_ix][np.ix_(cells, genes)]).mean(0)
                        score["effect"] = effect.mean()
                        genescore["effect"] = effect
                        
                    scores.append(score)
                    genescores.append(genescore)

        score_cols = ["cor", "rmse", "mse"]
        if transcriptome_predicted_full is not None:
            score_cols.append("effect")

        genescores_across = pd.concat(genescores)
        scores = pd.concat(genescores)
        genescores = genescores_across.groupby(["phase", "gene"])[score_cols].mean()
        scores = scores.groupby(["phase"])[score_cols].mean()

        if return_prediction:
            return transcriptome_predicted, scores, genescores
        return scores, genescores
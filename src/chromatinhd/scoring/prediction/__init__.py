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
        folds=None,
        return_prediction=False,
        transcriptome_predicted_full=None,
    ):
        """
        gene_ids: mapping of gene ix to gene id
        """

        transcriptome_predicted = {}
        n_fragments = {}

        folds = self.folds

        if loader_kwargs is None:
            loader_kwargs = {}

        next_task_sets = []
        for fold in folds:
            next_task_sets.append({"tasks": fold["minibatches"]})
        next_task_sets[0]["loader_kwargs"] = loader_kwargs
        self.loaders.initialize(next_task_sets=next_task_sets)

        scores = []
        genescores = []
        for model_ix, (model, fold) in tqdm.tqdm(
            enumerate(zip(self.models, folds)), leave=False, total=len(self.models)
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

                    transcriptome_predicted_[np.ix_(data.cells_oi, data.genes_oi)] = (
                        predicted.detach().cpu().numpy()
                    )

                # score
                for phase, (cells, genes) in fold["phases"].items():
                    outcome_phase = self.outcome.numpy()[np.ix_(cells, genes)]
                    transcriptome_predicted__ = transcriptome_predicted_[
                        np.ix_(cells, genes)
                    ]

                    cor_gene = chd.utils.paircor(
                        outcome_phase, transcriptome_predicted__
                    )

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


import xarray as xr


class Scorer2:
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
        transcriptome_predicted_full=None,
        filterer=None,
        extract_total=False,
    ):
        """
        gene_ids: mapping of gene ix to gene id
        """

        n_fragments = {}

        folds = self.folds

        if loader_kwargs is None:
            loader_kwargs = {}

        next_task_sets = []
        for fold in folds:
            next_task_sets.append({"tasks": fold["minibatches"]})
        next_task_sets[0]["loader_kwargs"] = loader_kwargs
        self.loaders.initialize(next_task_sets=next_task_sets)

        phases_dim = pd.Index(folds[0]["phases"], name="phase")
        genes_dim = self.gene_ids
        design_dim = filterer.design.index
        model_dim = pd.Index(np.arange(len(folds)), name="model")

        cors = xr.DataArray(
            0.0,
            coords=[model_dim, phases_dim, design_dim],
        )

        genecors = xr.DataArray(
            0.0,
            coords=[model_dim, phases_dim, genes_dim, design_dim],
        )

        geneffects = xr.DataArray(
            0.0,
            coords=[model_dim, phases_dim, genes_dim, design_dim],
        )

        n_fragments_lost = xr.DataArray(
            0,
            coords=[model_dim, phases_dim, genes_dim, design_dim],
        )

        for model_ix, (model, fold) in tqdm.tqdm(
            enumerate(zip(self.models, folds)), leave=False, total=len(self.models)
        ):
            assert len(genes_dim) == len(fold["genes_all"]), (
                genes_dim,
                fold["genes_all"],
            )
            gene_ix_mapper = fold["gene_ix_mapper"]

            n_design = filterer.setup_next_chunk()
            # create transcriptome_predicted
            transcriptome_predicted_ = [
                np.zeros([self.outcome.shape[0], len(genes_dim)])
                for i in range(n_design)
            ]
            n_fragments_lost_cells = [
                np.zeros([self.outcome.shape[0], len(genes_dim)])
                for i in range(n_design)
            ]

            # infer and score
            with torch.no_grad():
                # infer
                model = model.to(self.device)
                for data in tqdm.tqdm(self.loaders):
                    # for data in self.loaders:
                    data = data.to(self.device)

                    fragments_oi = filterer.filter(data)

                    for design_ix, (predicted, n_fragments_oi_mb) in enumerate(
                        model.forward_multiple(
                            data, fragments_oi, extract_total=extract_total
                        )
                    ):
                        transcriptome_predicted_[design_ix][
                            np.ix_(data.cells_oi, gene_ix_mapper[data.genes_oi])
                        ] = (predicted.detach().cpu().numpy())

                        n_fragments_lost_cells[design_ix][
                            np.ix_(data.cells_oi, gene_ix_mapper[data.genes_oi])
                        ] = (n_fragments_oi_mb.detach().cpu().numpy())

                    self.loaders.submit_next()

                # score
                for phase, (cells, genes) in fold["phases"].items():
                    phase_ix = phases_dim.tolist().index(phase)

                    outcome_phase = self.outcome.numpy()[np.ix_(cells, genes)]

                    transcriptome_predicted_full__ = transcriptome_predicted_full[
                        model_ix
                    ][np.ix_(cells, genes)]

                    genes = gene_ix_mapper[genes]

                    for design_ix in range(n_design):
                        transcriptome_predicted__ = transcriptome_predicted_[design_ix][
                            np.ix_(cells, genes)
                        ]

                        cor_gene = chd.utils.paircor(
                            outcome_phase, transcriptome_predicted__
                        )

                        cors.values[model_ix, phase_ix, design_ix] = cor_gene.mean()
                        genecors.values[model_ix, phase_ix, genes, design_ix] = cor_gene

                        n_fragments_lost.values[
                            model_ix, phase_ix, genes, design_ix
                        ] = n_fragments_lost_cells[design_ix][np.ix_(cells, genes)].sum(
                            0
                        )

                        effect = (
                            transcriptome_predicted__ - transcriptome_predicted_full__
                        ).mean(0)
                        geneffects[model_ix, phase_ix, genes, design_ix] = effect

        scores = xr.Dataset({"cor": cors})
        genescores = xr.Dataset(
            {
                "cor": genecors,
                "effect": geneffects,
                "lost": n_fragments_lost,
            }
        )
        if extract_total:
            genescores["total"] = n_fragments_lost
            del genescores["lost"]

        self.scores = scores.mean("model")
        self.genescores = genescores.mean("model")

        return scores, genescores


class Scorer3:
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
        filterer=None,
    ):
        """
        gene_ids: mapping of gene ix to gene id
        """

        n_fragments = {}

        folds = self.folds

        if loader_kwargs is None:
            loader_kwargs = {}

        next_task_sets = []
        for fold in folds:
            next_task_sets.append({"tasks": fold["minibatches"]})
        next_task_sets[0]["loader_kwargs"] = loader_kwargs
        self.loaders.initialize(next_task_sets=next_task_sets)

        phases_dim = pd.Index(folds[0]["phases"], name="phase")
        genes_dim = self.gene_ids
        design_dim = filterer.design.index
        model_dim = pd.Index(np.arange(len(folds)), name="model")

        cors = xr.DataArray(
            0.0,
            coords=[model_dim, phases_dim, design_dim],
        )

        genecors = xr.DataArray(
            0.0,
            coords=[model_dim, phases_dim, genes_dim, design_dim],
        )

        geneffects = xr.DataArray(
            0.0,
            coords=[model_dim, phases_dim, genes_dim, design_dim],
        )

        genelost = xr.DataArray(
            0,
            coords=[model_dim, phases_dim, genes_dim, design_dim],
        )
        genetotal = xr.DataArray(
            0,
            coords=[model_dim, phases_dim, genes_dim, design_dim],
        )

        for model_ix, (model, fold) in tqdm.tqdm(
            enumerate(zip(self.models, folds)), leave=False, total=len(self.models)
        ):
            assert len(genes_dim) == fold["genes_all"]
            gene_ix_mapper = fold["gene_ix_mapper"]

            n_design = filterer.setup_next_chunk()
            # create transcriptome_predicted
            transcriptome_predicted_ = [
                np.zeros(self.outcome.shape) for i in range(n_design)
            ]
            n_fragments_ = [np.zeros(self.outcome.shape) for i in range(n_design)]

            # infer and score
            with torch.no_grad():
                # infer
                model = model.to(self.device)
                for data in tqdm.tqdm(self.loaders):
                    # for data in self.loaders:
                    data = data.to(self.device)

                    fragments_oi = filterer.filter(data)

                    for design_ix, (predicted, n_lost, n_total) in enumerate(
                        model.forward_multiple(data, fragments_oi)
                    ):
                        transcriptome_predicted_[design_ix][
                            np.ix_(data.cells_oi, gene_ix_mapper[data.genes_oi])
                        ] = (predicted.detach().cpu().numpy())

                        genelost.values[
                            model_ix, gene_ix_mapper[data.genes_oi], design_ix
                        ] += (n_lost.detach().cpu().numpy())
                        genetotal.values[
                            model_ix, gene_ix_mapper[data.genes_oi], design_ix
                        ] += (n_total.detach().cpu().numpy())

                    self.loaders.submit_next()

                # score
                for phase, (cells, genes) in fold["phases"].items():
                    phase_ix = phases_dim.tolist().index(phase)

                    outcome_phase = self.outcome.numpy()[
                        np.ix_(cells, gene_ix_mapper[genes])
                    ]

                    transcriptome_predicted_full__ = transcriptome_predicted_full[
                        model_ix
                    ][np.ix_(cells, gene_ix_mapper[genes])]
                    for design_ix in range(n_design):
                        transcriptome_predicted__ = transcriptome_predicted_[design_ix][
                            np.ix_(cells, gene_ix_mapper[genes])
                        ]

                        cor_gene = chd.utils.paircor(
                            outcome_phase, transcriptome_predicted__
                        )

                        cors.values[model_ix, phase_ix, design_ix] = cor_gene.mean()
                        genecors.values[
                            model_ix, phase_ix, gene_ix_mapper[genes], design_ix
                        ] = cor_gene

                        effect = (
                            transcriptome_predicted__ - transcriptome_predicted_full__
                        ).mean(0)
                        geneffects[
                            model_ix, phase_ix, gene_ix_mapper[genes], design_ix
                        ] = effect

        scores = xr.Dataset({"cor": cors})
        genescores = xr.Dataset(
            {
                "cor": genecors,
                "effect": geneffects,
                "lost": genelost,
                "total": genetotal,
            }
        )

        return scores, genescores

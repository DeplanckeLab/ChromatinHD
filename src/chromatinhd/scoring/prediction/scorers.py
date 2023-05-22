import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import chromatinhd as chd
import dataclasses
import pickle
import h5py


def zscore(x, dim=0):
    return (x - x.mean(axis=dim, keepdims=True)) / x.std(axis=dim, keepdims=True)


def zscore_relative(x, y, dim=0):
    return (x - y.mean(axis=dim, keepdims=True)) / y.std(axis=dim, keepdims=True)


def extract_from_transcriptome_predicted_full(
    transcriptome_predicted_full, model_ix, cell_ix, gene_ix
):
    if isinstance(transcriptome_predicted_full, h5py.File):
        cell_ix_sorted, cell_ix_inverse = np.unique(cell_ix, return_index=True)
        return transcriptome_predicted_full[str(model_ix)][cell_ix_sorted][
            cell_ix_inverse
        ][:, gene_ix]
    return transcriptome_predicted_full[model_ix][np.ix_(cell_ix, gene_ix)]


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
                            - extract_from_transcriptome_predicted_full(
                                transcriptome_predicted_full, model_ix, cells, genes
                            )
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


class Scorer2(chd.flow.Flow):
    def __init__(
        self, models, folds, loaders, outcome, gene_ids, cell_ids, device="cuda"
    ):
        assert len(models) == len(folds)
        self.outcome = outcome
        self.models = models
        self.device = device
        self.folds = folds
        self.loaders = loaders
        self.gene_ids = gene_ids
        self.cell_ids = cell_ids

    def score(
        self,
        filterer,
        loader_kwargs=None,
        transcriptome_predicted_full=None,
        nothing_scoring=None,
        extract_total=False,
        extract_per_cellxgene=True,
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

        cellgeneeffects = []
        cellgenelosts = []
        cellgenedeltacors = []

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
            cell_ix_mapper = fold["cell_ix_mapper"]
            cells_dim = self.cell_ids[fold["cells_all"]]
            cells_dim.name = "cell"
            # assert cells_dim.name == "cell"

            n_design = filterer.setup_next_chunk()
            # create transcriptome_predicted
            transcriptome_predicted_ = [
                np.zeros([len(cells_dim), len(genes_dim)]) for i in range(n_design)
            ]
            transcriptome_predicted_full_ = np.zeros([len(cells_dim), len(genes_dim)])
            n_fragments_lost_cells = [
                np.zeros([len(cells_dim), len(genes_dim)]) for i in range(n_design)
            ]

            cell_oi = "GGACTAAAGGAACACA-1"

            debug = False
            if cell_oi in cells_dim:
                debug = True
                cell_ix = cells_dim.tolist().index(cell_oi)

            # infer and score
            with torch.no_grad():
                # infer
                model = model.to(self.device)
                # for data in tqdm.tqdm(self.loaders):
                for data in self.loaders:
                    data = data.to(self.device)

                    fragments_oi = filterer.filter(data)

                    for design_ix, (predicted, n_fragments_oi_mb,) in enumerate(
                        model.forward_multiple(
                            data, [*fragments_oi, None], extract_total=extract_total
                        )
                    ):
                        if design_ix == len(filterer.design):
                            transcriptome_predicted_full_[
                                np.ix_(
                                    cell_ix_mapper[data.cells_oi],
                                    gene_ix_mapper[data.genes_oi],
                                )
                            ] = (
                                predicted.detach().cpu().numpy()
                            )
                        else:
                            transcriptome_predicted_[design_ix][
                                np.ix_(
                                    cell_ix_mapper[data.cells_oi],
                                    gene_ix_mapper[data.genes_oi],
                                )
                            ] = (
                                predicted.detach().cpu().numpy()
                            )

                            n_fragments_lost_cells[design_ix][
                                np.ix_(
                                    cell_ix_mapper[data.cells_oi],
                                    gene_ix_mapper[data.genes_oi],
                                )
                            ] = (
                                n_fragments_oi_mb.detach().cpu().numpy()
                            )

                    self.loaders.submit_next()

            model = model.to("cpu")

            # score
            for phase, (cells, genes) in fold["phases"].items():
                phase_ix = phases_dim.tolist().index(phase)

                outcome_phase = self.outcome.numpy()[np.ix_(cells, genes)]

                genes = gene_ix_mapper[genes]
                cells = cell_ix_mapper[cells]

                transcriptome_predicted_full__ = transcriptome_predicted_full_[
                    np.ix_(cells, genes)
                ]
                for design_ix in range(n_design):
                    transcriptome_predicted__ = transcriptome_predicted_[design_ix][
                        np.ix_(cells, genes)
                    ]

                    # calculate correlation per gene and across genes
                    cor_gene = chd.utils.paircor(
                        outcome_phase, transcriptome_predicted__
                    )

                    cors.values[model_ix, phase_ix, design_ix] = cor_gene.mean()
                    genecors.values[model_ix, phase_ix, genes, design_ix] = cor_gene

                    # calculate n_fragments_lost
                    n_fragments_lost.values[
                        model_ix, phase_ix, genes, design_ix
                    ] = n_fragments_lost_cells[design_ix][np.ix_(cells, genes)].sum(0)

                    # calculate effect per gene and across genes
                    effect = (
                        transcriptome_predicted__ - transcriptome_predicted_full__
                    ).mean(0)
                    geneffects[model_ix, phase_ix, genes, design_ix] = effect

            if extract_per_cellxgene:
                cellgeneeffect = xr.DataArray(
                    (
                        np.stack(transcriptome_predicted_, 0)
                        - transcriptome_predicted_full_
                    ),
                    coords=[design_dim, cells_dim, genes_dim],
                )
                cellgeneeffects.append(cellgeneeffect)

                cellgenelost = xr.DataArray(
                    n_fragments_lost_cells,
                    coords=[design_dim, cells_dim, genes_dim],
                )
                cellgenelosts.append(cellgenelost)

                # calculate effect per cellxgene combination
                transcriptomes_predicted = np.stack(transcriptome_predicted_, 0)
                transcriptomes_predicted_full = transcriptome_predicted_full_[None, ...]
                transcriptomes_predicted_full_norm = zscore(
                    transcriptomes_predicted_full, 1
                )
                transcriptomes_predicted_norm = zscore_relative(
                    transcriptomes_predicted, transcriptomes_predicted_full, 1
                )

                outcomes = self.outcome.numpy()[
                    np.ix_(fold["cells_all"], fold["genes_all"])
                ][None, ...]
                outcomes_norm = zscore(outcomes, 1)

                cellgenedeltacor = xr.DataArray(
                    -np.sqrt(((transcriptomes_predicted_norm - outcomes_norm) ** 2))
                    - -np.sqrt(
                        ((transcriptomes_predicted_full_norm - outcomes_norm) ** 2)
                    ),
                    coords=[design_dim, cells_dim, genes_dim],
                )
                cellgenedeltacors.append(cellgenedeltacor)

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

        # compare with nothing_scoring, e.g. for retained and deltacor
        if nothing_scoring is not None:
            genescores["retained"] = 1 - genescores[
                "lost"
            ] / nothing_scoring.genescores["total"].sel(i=0)
            genescores["deltacor"] = (
                genescores["cor"] - nothing_scoring.genescores.sel(i=0)["cor"]
            )

        # create scoring
        scoring = Scoring(
            scores=scores,
            genescores=genescores,
            design=filterer.design,
        )

        # postprocess per cellxgene scores
        if extract_per_cellxgene:
            effects = xr.concat(
                cellgeneeffects,
                dim=model_dim,
            )  # .mean("model", skipna=True)

            losts = xr.concat(
                cellgenelosts,
                dim=model_dim,
            )  # .mean("model", skipna=True)

            deltacors = xr.concat(
                cellgenedeltacors,
                dim=model_dim,
            )  # .mean("model", skipna=True)

            # calculate deltacor_down_ratio
            # genescores["deltacor_down_ratio"] = (deltacors < -0.01).sum("cell") / (
            #     deltacors > 0.01
            # ).sum("cell")

            scoring.effects = effects
            scoring.losts = losts
            scoring.deltacors = deltacors
            scoring.cellgenedeltacors = cellgenedeltacors

        return scoring


@dataclasses.dataclass
class Scoring:
    scores: xr.Dataset
    genescores: xr.Dataset
    design: pd.DataFrame
    effects: xr.DataArray = None
    losts: xr.DataArray = None
    deltacors: xr.DataArray = None

    def save(self, scorer_folder):
        self.scores.to_netcdf(scorer_folder / "scores.nc")
        self.genescores.to_netcdf(scorer_folder / "genescores.nc")
        self.design.to_pickle(scorer_folder / "design.pkl")

    @classmethod
    def load(cls, scorer_folder):
        with xr.open_dataset(scorer_folder / "scores.nc") as scores:
            scores.load()

        with xr.open_dataset(scorer_folder / "genescores.nc") as genescores:
            genescores.load()
        return cls(
            scores=scores,
            genescores=genescores,
            design=pd.read_pickle((scorer_folder / "design.pkl").open("rb")),
        )

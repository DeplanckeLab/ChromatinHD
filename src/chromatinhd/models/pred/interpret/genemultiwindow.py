import chromatinhd as chd
import numpy as np
import pandas as pd
import xarray as xr
import pickle
import scipy.stats
import tqdm.auto as tqdm


def select_cutwindow(coordinates, window_start, window_end):
    """
    check whether coordinate 0 or coordinate 1 is within the window
    """
    return ~(
        ((coordinates[:, 0] < window_end) & (coordinates[:, 0] > window_start))
        | ((coordinates[:, 1] < window_end) & (coordinates[:, 1] > window_start))
    )


class MultiWindowCensorer:
    def __init__(self, window, window_sizes=(50, 100, 200, 500), relative_stride=0.5):
        design = [{"window": "control"}]
        for window_size in window_sizes:
            cuts = np.arange(*window, step=int(window_size * relative_stride))

            for window_start, window_end in zip(cuts, cuts + window_size):
                if window_start < window[0]:
                    continue
                if window_end > window[1]:
                    continue
                design.append(
                    {
                        "window_start": window_start,
                        "window_end": window_end,
                        "window_mid": window_start + (window_end - window_start) / 2,
                        "window_size": window_size,
                        "window": f"{window_start}-{window_end}",
                    }
                )
        design = pd.DataFrame(design).set_index("window")
        assert design.index.is_unique
        design["ix"] = np.arange(len(design))
        self.design = design

    def __len__(self):
        return len(self.design)

    def __call__(self, data):
        for window_start, window_end in zip(
            self.design["window_start"], self.design["window_end"]
        ):
            if np.isnan(window_start):
                fragments_oi = None
            else:
                fragments_oi = select_cutwindow(
                    data.fragments.coordinates, window_start, window_end
                )
            yield fragments_oi


class GeneMultiWindow(chd.flow.Flow):
    design = chd.flow.Stored("design")

    genes = chd.flow.Stored("genes", default=set)

    def score(
        self,
        fragments,
        transcriptome,
        models,
        folds,
        genes,
        censorer,
        force=False,
        device="cuda",
    ):
        force_ = force
        design = censorer.design.iloc[1:].copy()
        self.design = design

        pbar = tqdm.tqdm(genes, leave=False)

        for gene in pbar:
            pbar.set_description(gene)
            scores_file = self.get_scoring_path(gene) / "scores.pkl"

            force = force_
            if not scores_file.exists():
                force = True

            if force:
                deltacor_folds = []
                lost_folds = []
                for fold, model in zip(folds, models):
                    predicted, expected, n_fragments = model.get_prediction_censored(
                        fragments,
                        transcriptome,
                        censorer,
                        cell_ixs=np.concatenate(
                            [fold["cells_validation"], fold["cells_test"]]
                        ),
                        genes=[gene],
                        device=device,
                    )
                    predicted = predicted[..., 0]
                    expected = expected[..., 0]
                    n_fragments = n_fragments[..., 0]

                    cor = chd.utils.paircor(predicted, expected, dim=-1)
                    deltacor = cor[1:] - cor[0]

                    lost = (n_fragments[0] - n_fragments[1:]).mean(-1)

                    deltacor_folds.append(deltacor)
                    lost_folds.append(lost)

                deltacor_folds = np.stack(deltacor_folds, 0)
                lost_folds = np.stack(lost_folds, 0)

                result = xr.Dataset(
                    {
                        "deltacor": xr.DataArray(
                            deltacor_folds,
                            coords=[
                                ("model", np.arange(len(models))),
                                ("window", design.index),
                            ],
                        ),
                        "lost": xr.DataArray(
                            lost_folds,
                            coords=[
                                ("model", np.arange(len(models))),
                                ("window", design.index),
                            ],
                        ),
                    }
                )

                pickle.dump(result, scores_file.open("wb"))

                self.genes = self.genes | {gene}

    def interpolate(self, genes=None, force=False):
        force_ = force

        def fdr(p_vals):
            from scipy.stats import rankdata

            ranked_p_values = rankdata(p_vals)
            fdr = p_vals * len(p_vals) / ranked_p_values
            fdr[fdr > 1] = 1

            return fdr

        if genes is None:
            genes = self.genes

        pbar = tqdm.tqdm(genes, leave=False)

        for gene in pbar:
            pbar.set_description(gene)
            scores_file = self.get_scoring_path(gene) / "scores.pkl"

            if not scores_file.exists():
                continue

            interpolate_file = self.get_scoring_path(gene) / "interpolated.pkl"

            force = force_
            if not interpolate_file.exists():
                force = True

            if force:
                scores = pickle.load(scores_file.open("rb"))
                x = scores["deltacor"].values
                scores_statistical = []
                for i in range(x.shape[1]):
                    scores_statistical.append(
                        scipy.stats.ttest_1samp(x[:, i], 0, alternative="less").pvalue
                    )
                scores_statistical = pd.DataFrame({"pvalue": scores_statistical})
                scores_statistical["qval"] = fdr(scores_statistical["pvalue"])

                plotdata = scores.mean("model").stack().to_dataframe()
                plotdata = self.design.join(plotdata)

                plotdata["qval"] = scores_statistical["qval"].values

                window_sizes_info = pd.DataFrame(
                    {"window_size": self.design["window_size"].unique()}
                ).set_index("window_size")
                window_sizes_info["ix"] = np.arange(len(window_sizes_info))

                # interpolate
                positions_oi = np.arange(
                    self.design["window_start"].min(),
                    self.design["window_end"].max() + 1,
                )

                deltacor_interpolated = np.zeros(
                    (len(window_sizes_info), len(positions_oi))
                )
                lost_interpolated = np.zeros(
                    (len(window_sizes_info), len(positions_oi))
                )
                for window_size, window_size_info in window_sizes_info.iterrows():
                    plotdata_oi = plotdata.query("window_size == @window_size")
                    x = plotdata_oi["window_mid"].values.copy()
                    y = plotdata_oi["deltacor"].values.copy()
                    y[plotdata_oi["qval"] > 0.1] = 0.0
                    deltacor_interpolated_ = np.clip(
                        np.interp(positions_oi, x, y) / window_size * 1000,
                        -np.inf,
                        0,
                        # np.inf,
                    )
                    deltacor_interpolated[
                        window_size_info["ix"], :
                    ] = deltacor_interpolated_

                    lost_interpolated_ = (
                        np.interp(
                            positions_oi, plotdata_oi["window_mid"], plotdata_oi["lost"]
                        )
                        / window_size
                        * 1000
                    )
                    lost_interpolated[window_size_info["ix"], :] = lost_interpolated_

                deltacor = xr.DataArray(
                    deltacor_interpolated.mean(0),
                    coords=[
                        ("position", positions_oi),
                    ],
                )
                lost = xr.DataArray(
                    lost_interpolated.mean(0),
                    coords=[
                        ("position", positions_oi),
                    ],
                )

                # save
                interpolated = xr.Dataset({"deltacor": deltacor, "lost": lost})
                pickle.dump(
                    interpolated,
                    interpolate_file.open("wb"),
                )

    def get_plotdata(self, gene):
        interpolated_file = self.get_scoring_path(gene) / "interpolated.pkl"
        if not interpolated_file.exists():
            raise FileNotFoundError(f"File {interpolated_file} does not exist")

        interpolated = pickle.load(interpolated_file.open("rb"))

        plotdata = interpolated.to_dataframe()

        return plotdata

    def get_scoring_path(self, gene):
        path = self.path / f"{gene}"
        path.mkdir(parents=True, exist_ok=True)
        return path


# import xarray as xr
# import pandas as pd


# def score(fragments, transcriptome, folds, models, filterer, gene, device="cuda"):
#     phases_dim = pd.Index(folds[0]["phases"], name="phase")
#     genes_dim = pd.Index([gene], name="gene")
#     design_dim = filterer.design.index
#     model_dim = pd.Index(np.arange(len(folds)), name="model")

#     for model_ix, (model, fold) in tqdm.tqdm(
#         enumerate(zip(models, folds)), leave=False, total=len(models)
#     ):
#         cell_ix_mapper = fold["cell_ix_mapper"]
#         cells_dim = self.cell_ids[fold["cells_all"]]
#         cells_dim.name = "cell"

#         n_design = filterer.setup_next_chunk()
#         # create transcriptome_predicted
#         transcriptome_predicted_ = [
#             np.zeros([len(cells_dim), len(genes_dim)]) for i in range(n_design)
#         ]
#         transcriptome_predicted_full_ = np.zeros([len(cells_dim), len(genes_dim)])
#         n_fragments_lost_cells = [
#             np.zeros([len(cells_dim), len(genes_dim)]) for i in range(n_design)
#         ]

#         # infer and score
#         with torch.no_grad():
#             # infer
#             model = model.to(device)
#             for data in loaders:
#                 data = data.to(device)

#                 fragments_oi = filterer.filter(data)

#                 for design_ix, (predicted, n_fragments_oi_mb,) in enumerate(
#                     model.forward_multiple(
#                         data, [*fragments_oi, None], extract_total=extract_total
#                     )
#                 ):
#                     if design_ix == len(filterer.design):
#                         transcriptome_predicted_full_[
#                             np.ix_(
#                                 cell_ix_mapper[data.cells_oi],
#                                 gene_ix_mapper[data.genes_oi],
#                             )
#                         ] = (
#                             predicted.detach().cpu().numpy()
#                         )
#                     else:
#                         transcriptome_predicted_[design_ix][
#                             np.ix_(
#                                 cell_ix_mapper[data.cells_oi],
#                                 gene_ix_mapper[data.genes_oi],
#                             )
#                         ] = (
#                             predicted.detach().cpu().numpy()
#                         )

#                         n_fragments_lost_cells[design_ix][
#                             np.ix_(
#                                 cell_ix_mapper[data.cells_oi],
#                                 gene_ix_mapper[data.genes_oi],
#                             )
#                         ] = (
#                             n_fragments_oi_mb.detach().cpu().numpy()
#                         )

#                 self.loaders.submit_next()

#         model = model.to("cpu")

#         # score
#         for phase, (cells, genes) in fold["phases"].items():
#             phase_ix = phases_dim.tolist().index(phase)

#             outcome_phase = self.outcome.numpy()[np.ix_(cells, genes)]

#             genes = gene_ix_mapper[genes]
#             cells = cell_ix_mapper[cells]

#             transcriptome_predicted_full__ = transcriptome_predicted_full_[
#                 np.ix_(cells, genes)
#             ]
#             for design_ix in range(n_design):
#                 transcriptome_predicted__ = transcriptome_predicted_[design_ix][
#                     np.ix_(cells, genes)
#                 ]

#                 # calculate correlation per gene and across genes
#                 cor_gene = chd.utils.paircor(outcome_phase, transcriptome_predicted__)

#                 cors.values[model_ix, phase_ix, design_ix] = cor_gene.mean()
#                 genecors.values[model_ix, phase_ix, genes, design_ix] = cor_gene

#                 # calculate n_fragments_lost
#                 n_fragments_lost.values[
#                     model_ix, phase_ix, genes, design_ix
#                 ] = n_fragments_lost_cells[design_ix][np.ix_(cells, genes)].sum(0)

#                 # calculate effect per gene and across genes
#                 effect = (
#                     transcriptome_predicted__ - transcriptome_predicted_full__
#                 ).mean(0)
#                 geneffects[model_ix, phase_ix, genes, design_ix] = effect

#         if extract_per_cellxgene:
#             cellgeneeffect = xr.DataArray(
#                 (np.stack(transcriptome_predicted_, 0) - transcriptome_predicted_full_),
#                 coords=[design_dim, cells_dim, genes_dim],
#             )
#             cellgeneeffects.append(cellgeneeffect)

#             cellgenelost = xr.DataArray(
#                 n_fragments_lost_cells,
#                 coords=[design_dim, cells_dim, genes_dim],
#             )
#             cellgenelosts.append(cellgenelost)

#             # calculate effect per cellxgene combination
#             transcriptomes_predicted = np.stack(transcriptome_predicted_, 0)
#             transcriptomes_predicted_full = transcriptome_predicted_full_[None, ...]
#             transcriptomes_predicted_full_norm = zscore(
#                 transcriptomes_predicted_full, 1
#             )
#             transcriptomes_predicted_norm = zscore_relative(
#                 transcriptomes_predicted, transcriptomes_predicted_full, 1
#             )

#             outcomes = self.outcome.numpy()[
#                 np.ix_(fold["cells_all"], fold["genes_all"])
#             ][None, ...]
#             outcomes_norm = zscore(outcomes, 1)

#             cellgenedeltacor = xr.DataArray(
#                 -np.sqrt(((transcriptomes_predicted_norm - outcomes_norm) ** 2))
#                 - -np.sqrt(((transcriptomes_predicted_full_norm - outcomes_norm) ** 2)),
#                 coords=[design_dim, cells_dim, genes_dim],
#             )
#             cellgenedeltacors.append(cellgenedeltacor)

#     scores = xr.Dataset({"cor": cors})
#     genescores = xr.Dataset(
#         {
#             "cor": genecors,
#             "effect": geneffects,
#             "lost": n_fragments_lost,
#         }
#     )
#     if extract_total:
#         genescores["total"] = n_fragments_lost
#         del genescores["lost"]

#     # compare with nothing_scoring, e.g. for retained and deltacor
#     if nothing_scoring is not None:
#         genescores["retained"] = 1 - genescores["lost"] / nothing_scoring.genescores[
#             "total"
#         ].sel(i=0)
#         genescores["deltacor"] = (
#             genescores["cor"] - nothing_scoring.genescores.sel(i=0)["cor"]
#         )

#     # create scoring
#     scoring = Scoring(
#         scores=scores,
#         genescores=genescores,
#         design=filterer.design,
#     )

#     # postprocess per cellxgene scores
#     if extract_per_cellxgene:
#         scoring.cellgenelosts = cellgenelosts
#         scoring.cellgenedeltacors = cellgenedeltacors


# class Scorer2(chd.flow.Flow):
#     def __init__(
#         self, models, folds, loaders, outcome, gene_ids, cell_ids, device="cuda"
#     ):
#         assert len(models) == len(folds)
#         self.outcome = outcome
#         self.models = models
#         self.device = device
#         self.folds = folds
#         self.loaders = loaders
#         self.gene_ids = gene_ids
#         self.cell_ids = cell_ids

#         phases_dim = pd.Index(folds[0]["phases"], name="phase")
#         genes_dim = self.gene_ids
#         design_dim = filterer.design.index
#         model_dim = pd.Index(np.arange(len(folds)), name="model")

#         cors = xr.DataArray(
#             0.0,
#             coords=[model_dim, phases_dim, design_dim],
#         )

#         genecors = xr.DataArray(
#             0.0,
#             coords=[model_dim, phases_dim, genes_dim, design_dim],
#         )

#         geneffects = xr.DataArray(
#             0.0,
#             coords=[model_dim, phases_dim, genes_dim, design_dim],
#         )

#         cellgeneeffects = []
#         cellgenelosts = []
#         cellgenedeltacors = []

#         n_fragments_lost = xr.DataArray(
#             0,
#             coords=[model_dim, phases_dim, genes_dim, design_dim],
#         )

#     def score(
#         self,
#         filterer,
#         loader_kwargs=None,
#         nothing_scoring=None,
#         extract_total=False,
#         extract_per_cellxgene=True,
#     ):
#         """
#         gene_ids: mapping of gene ix to gene id
#         """

#         folds = self.folds

#         if loader_kwargs is None:
#             loader_kwargs = {}

#         next_task_sets = []
#         for fold in folds:
#             next_task_sets.append({"tasks": fold["minibatches"]})
#         next_task_sets[0]["loader_kwargs"] = loader_kwargs
#         self.loaders.initialize(next_task_sets=next_task_sets)

#         phases_dim = pd.Index(folds[0]["phases"], name="phase")
#         genes_dim = self.gene_ids
#         design_dim = filterer.design.index
#         model_dim = pd.Index(np.arange(len(folds)), name="model")

#         cors = xr.DataArray(
#             0.0,
#             coords=[model_dim, phases_dim, design_dim],
#         )

#         genecors = xr.DataArray(
#             0.0,
#             coords=[model_dim, phases_dim, genes_dim, design_dim],
#         )

#         geneffects = xr.DataArray(
#             0.0,
#             coords=[model_dim, phases_dim, genes_dim, design_dim],
#         )

#         cellgeneeffects = []
#         cellgenelosts = []
#         cellgenedeltacors = []

#         n_fragments_lost = xr.DataArray(
#             0,
#             coords=[model_dim, phases_dim, genes_dim, design_dim],
#         )

#         for model_ix, (model, fold) in tqdm.tqdm(
#             enumerate(zip(self.models, folds)), leave=False, total=len(self.models)
#         ):
#             assert len(genes_dim) == len(fold["genes_all"]), (
#                 genes_dim,
#                 fold["genes_all"],
#             )
#             gene_ix_mapper = fold["gene_ix_mapper"]
#             cell_ix_mapper = fold["cell_ix_mapper"]
#             cells_dim = self.cell_ids[fold["cells_all"]]
#             cells_dim.name = "cell"
#             # assert cells_dim.name == "cell"

#             n_design = filterer.setup_next_chunk()
#             # create transcriptome_predicted
#             transcriptome_predicted_ = [
#                 np.zeros([len(cells_dim), len(genes_dim)]) for i in range(n_design)
#             ]
#             transcriptome_predicted_full_ = np.zeros([len(cells_dim), len(genes_dim)])
#             n_fragments_lost_cells = [
#                 np.zeros([len(cells_dim), len(genes_dim)]) for i in range(n_design)
#             ]

#             # infer and score
#             with torch.no_grad():
#                 # infer
#                 model = model.to(self.device)
#                 # for data in tqdm.tqdm(self.loaders):
#                 for data in self.loaders:
#                     data = data.to(self.device)

#                     fragments_oi = filterer.filter(data)

#                     for design_ix, (predicted, n_fragments_oi_mb,) in enumerate(
#                         model.forward_multiple(
#                             data, [*fragments_oi, None], extract_total=extract_total
#                         )
#                     ):
#                         if design_ix == len(filterer.design):
#                             transcriptome_predicted_full_[
#                                 np.ix_(
#                                     cell_ix_mapper[data.cells_oi],
#                                     gene_ix_mapper[data.genes_oi],
#                                 )
#                             ] = (
#                                 predicted.detach().cpu().numpy()
#                             )
#                         else:
#                             transcriptome_predicted_[design_ix][
#                                 np.ix_(
#                                     cell_ix_mapper[data.cells_oi],
#                                     gene_ix_mapper[data.genes_oi],
#                                 )
#                             ] = (
#                                 predicted.detach().cpu().numpy()
#                             )

#                             n_fragments_lost_cells[design_ix][
#                                 np.ix_(
#                                     cell_ix_mapper[data.cells_oi],
#                                     gene_ix_mapper[data.genes_oi],
#                                 )
#                             ] = (
#                                 n_fragments_oi_mb.detach().cpu().numpy()
#                             )

#                     self.loaders.submit_next()

#             model = model.to("cpu")

#             # score
#             for phase, (cells, genes) in fold["phases"].items():
#                 phase_ix = phases_dim.tolist().index(phase)

#                 outcome_phase = self.outcome.numpy()[np.ix_(cells, genes)]

#                 genes = gene_ix_mapper[genes]
#                 cells = cell_ix_mapper[cells]

#                 transcriptome_predicted_full__ = transcriptome_predicted_full_[
#                     np.ix_(cells, genes)
#                 ]
#                 for design_ix in range(n_design):
#                     transcriptome_predicted__ = transcriptome_predicted_[design_ix][
#                         np.ix_(cells, genes)
#                     ]

#                     # calculate correlation per gene and across genes
#                     cor_gene = chd.utils.paircor(
#                         outcome_phase, transcriptome_predicted__
#                     )

#                     cors.values[model_ix, phase_ix, design_ix] = cor_gene.mean()
#                     genecors.values[model_ix, phase_ix, genes, design_ix] = cor_gene

#                     # calculate n_fragments_lost
#                     n_fragments_lost.values[
#                         model_ix, phase_ix, genes, design_ix
#                     ] = n_fragments_lost_cells[design_ix][np.ix_(cells, genes)].sum(0)

#                     # calculate effect per gene and across genes
#                     effect = (
#                         transcriptome_predicted__ - transcriptome_predicted_full__
#                     ).mean(0)
#                     geneffects[model_ix, phase_ix, genes, design_ix] = effect

#             if extract_per_cellxgene:
#                 cellgeneeffect = xr.DataArray(
#                     (
#                         np.stack(transcriptome_predicted_, 0)
#                         - transcriptome_predicted_full_
#                     ),
#                     coords=[design_dim, cells_dim, genes_dim],
#                 )
#                 cellgeneeffects.append(cellgeneeffect)

#                 cellgenelost = xr.DataArray(
#                     n_fragments_lost_cells,
#                     coords=[design_dim, cells_dim, genes_dim],
#                 )
#                 cellgenelosts.append(cellgenelost)

#                 # calculate effect per cellxgene combination
#                 transcriptomes_predicted = np.stack(transcriptome_predicted_, 0)
#                 transcriptomes_predicted_full = transcriptome_predicted_full_[None, ...]
#                 transcriptomes_predicted_full_norm = zscore(
#                     transcriptomes_predicted_full, 1
#                 )
#                 transcriptomes_predicted_norm = zscore_relative(
#                     transcriptomes_predicted, transcriptomes_predicted_full, 1
#                 )

#                 outcomes = self.outcome.numpy()[
#                     np.ix_(fold["cells_all"], fold["genes_all"])
#                 ][None, ...]
#                 outcomes_norm = zscore(outcomes, 1)

#                 cellgenedeltacor = xr.DataArray(
#                     -np.sqrt(((transcriptomes_predicted_norm - outcomes_norm) ** 2))
#                     - -np.sqrt(
#                         ((transcriptomes_predicted_full_norm - outcomes_norm) ** 2)
#                     ),
#                     coords=[design_dim, cells_dim, genes_dim],
#                 )
#                 cellgenedeltacors.append(cellgenedeltacor)

#         scores = xr.Dataset({"cor": cors})
#         genescores = xr.Dataset(
#             {
#                 "cor": genecors,
#                 "effect": geneffects,
#                 "lost": n_fragments_lost,
#             }
#         )
#         if extract_total:
#             genescores["total"] = n_fragments_lost
#             del genescores["lost"]

#         # compare with nothing_scoring, e.g. for retained and deltacor
#         if nothing_scoring is not None:
#             genescores["retained"] = 1 - genescores[
#                 "lost"
#             ] / nothing_scoring.genescores["total"].sel(i=0)
#             genescores["deltacor"] = (
#                 genescores["cor"] - nothing_scoring.genescores.sel(i=0)["cor"]
#             )

#         # create scoring
#         scoring = Scoring(
#             scores=scores,
#             genescores=genescores,
#             design=filterer.design,
#         )

#         # postprocess per cellxgene scores
#         if extract_per_cellxgene:
#             scoring.cellgenelosts = cellgenelosts
#             scoring.cellgenedeltacors = cellgenedeltacors

#         return scoring


# @dataclasses.dataclass
# class Scoring:
#     scores: xr.Dataset
#     genescores: xr.Dataset
#     design: pd.DataFrame
#     effects: xr.DataArray = None
#     losts: xr.DataArray = None
#     deltacors: xr.DataArray = None

#     def save(self, scorer_folder):
#         self.scores.to_netcdf(scorer_folder / "scores.nc")
#         self.genescores.to_netcdf(scorer_folder / "genescores.nc")
#         self.design.to_pickle(scorer_folder / "design.pkl")

#     @classmethod
#     def load(cls, scorer_folder):
#         with xr.open_dataset(scorer_folder / "scores.nc") as scores:
#             scores.load()

#         with xr.open_dataset(scorer_folder / "genescores.nc") as genescores:
#             genescores.load()
#         return cls(
#             scores=scores,
#             genescores=genescores,
#             design=pd.read_pickle((scorer_folder / "design.pkl").open("rb")),
#         )

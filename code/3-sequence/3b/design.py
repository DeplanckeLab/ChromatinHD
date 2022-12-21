import peakfreeatac.loaders.fragments
import peakfreeatac.loaders.fragmentmotif
import peakfreeatac.models.promotermotif.v4.actual
import peakfreeatac.models.promotermotif.v4.actual
import peakfreeatac.models.promotermotif.v4.baseline

import pickle
import numpy as np

n_cells_step = 300
n_genes_step = 1000

def get_design(dataset_name, transcriptome, motifscan, fragments, window):
    transcriptome_X_dense = transcriptome.X.dense()
    general_model_parameters = {
        "mean_gene_expression":transcriptome_X_dense.mean(0),
        "n_genes":fragments.n_genes,
    }

    general_loader_parameters = {
        "fragments":fragments,
        "motifscan":motifscan,
        "cellxgene_batch_size":n_cells_step * n_genes_step,
        "window":window
    }

    design = {}
    design["v4_baseline"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.baseline.Model,
        "model_parameters": {**general_model_parameters},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }

    baseline_location = pfa.get_git_root() / (dataset_name + "_" + "baseline_model.pkl")
    if baseline_location.exists():
        baseline_model = pickle.load(open(baseline_location, "rb"))
    else:
        baseline_model = None
    design["v4_dummy"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "dummy_motifs":True},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.Motifcounts,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 150])}
    }
    design["v4"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.Motifcounts,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 150])}
    }
    design["v4_cutoff001"] = design["v4"]
    design["v4_1k-1k"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.Motifcounts,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-1000, 1000])}
    }
    design["v4_150-0"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.Motifcounts,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 0])}
    }
    design["v4_0-150"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.Motifcounts,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([0, 150])}
    }
    design["v4_10-10"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.Motifcounts,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-10, 10])}
    }
    design["v4_1-1"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.Motifcounts,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-1, 1])}
    }
    design["v4_nn_1k-1k"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "n_layers":1},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.Motifcounts,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-1000, 1000])}
    }   
    design["v4_nn"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "n_layers":1},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.Motifcounts,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 150])}
    }
    design["v4_nn_cutoff001"] = design["v4_nn"]
    design["v4_split"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.MotifcountsSplit,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 150])}
    }
    design["v4_nn_split"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "n_layers":1},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.MotifcountsSplit,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 150])}
    }

    design["v4_nn_dummy1"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "n_layers":1, "dummy_motifs":1},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.Motifcounts,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 150])}
    }

    design["v4_lw_split"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "weight_lengths":"feature"},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.MotifcountsSplit,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 150])}
    }

    design["v4_nn_lw"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "weight_lengths":"feature", "n_layers":1},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.MotifcountsSplit,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 150])}
    }

    design["v4_nn_lw_split"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "weight_lengths":"feature", "n_layers":1},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.MotifcountsSplit,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 150])}
    }

    design["v4_nn_lw_split_mean"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "weight_lengths":"feature", "n_layers":1, "aggregation":"mean"},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.MotifcountsSplit,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 150])}
    }

    design["v4_nn2_lw_split"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "weight_lengths":"feature", "n_layers":2},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.MotifcountsSplit,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 150])}
    }

    design["v4_150-100-50-0"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.MotifcountsMultiple,
        "loader_parameters": {**general_loader_parameters, "cutwindows":np.array([-150, -100, -50, 0])}
    }

    design["v4_150-0_nn"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "n_layers":1},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.Motifcounts,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 0])}
    }

    design["v4_150-100-50-0_nn"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "n_layers":1},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.MotifcountsMultiple,
        "loader_parameters": {**general_loader_parameters, "cutwindows":np.array([-150, -100, -50, 0])}
    }

    design["v4_1k-150-0_nn"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "n_layers":1},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.MotifcountsMultiple,
        "loader_parameters": {**general_loader_parameters, "cutwindows":np.array([-1000, -150, 0])}
    }

    design["v4_prom"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.MotifcountsRelative,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 150]), "promoter_width":500}
    }

    design["v4_prom_nn"] = {
        "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
        "model_parameters": {**general_model_parameters, "baseline":baseline_model, "n_layers":1},
        "loader_cls":peakfreeatac.loaders.fragmentmotif.MotifcountsRelative,
        "loader_parameters": {**general_loader_parameters, "cutwindow":np.array([-150, 150]), "promoter_width":500}
    }

    design_titr = {}
    starts = np.arange(-200, 180, 10)
    ends = starts + 30
    for start, end in zip(starts, ends):
        design_titr["v4_titr" + str(start)] = {
            "model_cls":peakfreeatac.models.promotermotif.v4.actual.Model,
            "model_parameters": {**general_model_parameters, "baseline":baseline_model},
            "loader_cls":peakfreeatac.loaders.fragmentmotif.MotifcountsMultiple,
            "loader_parameters": {**general_loader_parameters, "cutwindows":np.array([start, end])}
        }
    design["titr"] = design_titr
    return design


import peakfreeatac as pfa
import peakfreeatac.loaders.minibatching
def get_folds_training(fragments, folds):
    for fold in folds:
        rg = np.random.RandomState(0)
        fold["minibatches_train"] = pfa.loaders.minibatching.create_bins_random(
            fold["cells_train"],
            fold["genes_train"],
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all = True,
            rg=rg,
        )
        fold["minibatches_validation"] = pfa.loaders.minibatching.create_bins_ordered(
            fold["cells_validation"],
            fold["genes_validation"],
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all = True,
            rg=rg,
        )
        fold["minibatches_validation_trace"] = fold["minibatches_validation"][:5]
    return folds

def get_folds_inference(fragments, folds):
    for fold in folds:
        cells_train = list(fold["cells_train"])
        genes_train = list(fold["genes_train"])
        cells_validation = list(fold["cells_validation"])
        genes_validation = list(fold["genes_validation"])
        
        rg = np.random.RandomState(0)

        minibatches_train = pfa.loaders.minibatching.create_bins_ordered(
            cells_train,
            genes_train,
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all = True,
            rg=rg,
        )
        minibatches_validation = pfa.loaders.minibatching.create_bins_ordered(
            cells_train + cells_validation,
            genes_validation,
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all = True,
            rg=rg,
        )
        minibatches_validation_cell = pfa.loaders.minibatching.create_bins_ordered(
            cells_validation,
            genes_train,
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all = True,
            rg=rg,
        )
        fold["minibatches_train"] = minibatches_train
        fold["minibatches_validation_cell"] = minibatches_validation_cell
        fold["minibatches_validation"] = minibatches_validation
        fold["minibatches"] = minibatches_validation_cell + minibatches_validation
        
        fold["phases"] = {
            # "train":[cells_train, genes_train],
            "validation_cell":[cells_validation, genes_train],
            "validation":[cells_train + cells_validation, genes_validation],
            "validation_full":[cells_validation, genes_validation]
        }
    return folds



def get_folds_test(fragments, folds):
    for fold in folds:
        genes_train = list(fold["genes_train"])
        cells_test = list(fold["cells_test"])
        genes_test = list(fold["genes_test"])
        
        rg = np.random.RandomState(0)

        minibatches_test = pfa.loaders.minibatching.create_bins_ordered(
            cells_test,
            genes_test,
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all = True,
            rg=rg,
        )
        minibatches_test_cells = pfa.loaders.minibatching.create_bins_ordered(
            cells_test,
            genes_train,
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all = True,
            rg=rg,
        )
        fold["minibatches_test_cells"] = minibatches_test_cells
        fold["minibatches_test"] = minibatches_test
        fold["minibatches"] = minibatches_test_cells + minibatches_test
        
        fold["phases"] = {
            "test_cells":[cells_test, genes_train],
            "test":[cells_test, genes_test]
        }
    return folds
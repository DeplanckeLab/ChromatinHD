import peakfreeatac.loaders.fragments
import peakfreeatac.loaders.fragmentmotif
import peakfreeatac.models.positional.v14

import pickle
import numpy as np

n_cells_step = 100
n_genes_step = 1000

def get_design(dataset_name, transcriptome, fragments, window):
    transcriptome_X_dense = transcriptome.X.dense()
    general_model_parameters = {
        "mean_gene_expression":transcriptome_X_dense.mean(0),
        "n_genes":fragments.n_genes,
    }

    general_loader_parameters = {
        "fragments":fragments,
        "cellxgene_batch_size":n_cells_step * n_genes_step,
        "window":window
    }

    design = {}
    design["v14"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    return design


import peakfreeatac as pfa
import peakfreeatac.loaders.minibatching
def get_folds_training(fragments, folds):
    for fold in folds:
        rg = np.random.RandomState(0)
        fold["minibatches_train"] = pfa.loaders.minibatching.create_bins_random(
            fold["cells_train"],
            list(fold["genes_train"]) + list(fold["genes_validation"]),
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all = True,
            rg=rg,
        )
        fold["minibatches_validation"] = pfa.loaders.minibatching.create_bins_ordered(
            fold["cells_validation"],
            list(fold["genes_train"]) + list(fold["genes_validation"]),
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
            "validation":[cells_train + cells_validation, genes_validation]
        }
    return folds
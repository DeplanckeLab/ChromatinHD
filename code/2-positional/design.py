import peakfreeatac.loaders.fragments
import peakfreeatac.loaders.fragmentmotif
import peakfreeatac.models
import peakfreeatac.models.positional.counter
import peakfreeatac.models.positional.v1
import peakfreeatac.models.positional.v14
import peakfreeatac.models.positional.v15

import pickle
import numpy as np

n_cells_step = 2000
# n_genes_step = 1000
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
    design["counter"] = {
        "model_cls":peakfreeatac.models.positional.counter.Model,
        "model_parameters": {**general_model_parameters},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["counter_binary"] = {
        "model_cls":peakfreeatac.models.positional.counter.Model,
        "model_parameters": {**general_model_parameters, "reduce":"mean"},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v1"] = {
        "model_cls":peakfreeatac.models.positional.v1.Model,
        "model_parameters": {**general_model_parameters, "window":window},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_dummy"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "dummy":True},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_5freq"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "n_frequencies":5},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_20freq"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "n_frequencies":20},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_3freq"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "n_frequencies":3},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_50freq"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "n_frequencies":50},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_50freq_sum"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "n_frequencies":50, "reduce":"sum"},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_50freq_linear"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "n_frequencies":50, "nonlinear":False},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_50freq_sigmoid"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "n_frequencies":50, "nonlinear":"sigmoid"},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_50freq_sum_sigmoid"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "n_frequencies":50, "nonlinear":"sigmoid", "reduce":"sum"},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_50freq_sum_elu"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "n_frequencies":50, "nonlinear":"elu", "reduce":"sum"},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_50freq_sum_sigmoid_initdefault"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "n_frequencies":20, "nonlinear":"sigmoid", "reduce":"sum", "embedding_to_expression_initialization":"default"},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_50freq_sum"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "n_frequencies":50, "reduce":"sum"},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_50freq_sum_1emb_sigmoid"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "n_frequencies":50, "nonlinear":"sigmoid", "reduce":"sum", "n_embedding_dimensions":1},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v14_50freq_sum_sigmoid_drop05"] = {
        "model_cls":peakfreeatac.models.positional.v14.Model,
        "model_parameters": {**general_model_parameters, "n_frequencies":50, "nonlinear":"sigmoid", "reduce":"sum", "dropout_rate":0.1},
        "loader_cls":peakfreeatac.loaders.fragments.Fragments,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v15"] = {
        "model_cls":peakfreeatac.models.positional.v15.Model,
        "model_parameters": {**general_model_parameters},
        "loader_cls":peakfreeatac.loaders.fragments.FragmentsCounting,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v15_noselfatt"] = {
        "model_cls":peakfreeatac.models.positional.v15.Model,
        "model_parameters": {**general_model_parameters, "selfatt":False},
        "loader_cls":peakfreeatac.loaders.fragments.FragmentsCounting,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}}
    }
    design["v15_att3"] = {
        "model_cls":peakfreeatac.models.positional.v15.Model,
        "model_parameters": {**general_model_parameters},
        "loader_cls":peakfreeatac.loaders.fragments.FragmentsCounting,
        "loader_parameters": {**{k:general_loader_parameters[k] for k in ["fragments", "cellxgene_batch_size", "window"]}, "n":(2, 3)}
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
        fold["minibatches_validation_trace"] = fold["minibatches_validation"][:8]
    return folds

def get_folds_inference(fragments, folds):
    for fold in folds:
        cells_train = list(fold["cells_train"])[:200]
        genes_train = list(fold["genes_train"])
        cells_validation = list(fold["cells_validation"])
        # genes_validation = list(fold["genes_validation"])
        
        rg = np.random.RandomState(0)

        minibatches = pfa.loaders.minibatching.create_bins_ordered(
            cells_train + cells_validation,
            genes_train,
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all = True,
            rg=rg,
        )
        fold["minibatches"] = minibatches
        
        fold["phases"] = {
            "train":[cells_train, genes_train],
            "validation":[cells_validation, genes_train]
        }
    return folds

def get_folds_test(fragments, folds):
    for fold in folds:
        cells_test = list(fold["cells_test"])
        genes_test = list(fold["genes_test"])
        
        rg = np.random.RandomState(0)

        minibatches = pfa.loaders.minibatching.create_bins_ordered(
            cells_test,
            genes_test,
            n_cells_step=n_cells_step,
            n_genes_step=n_genes_step,
            n_genes_total=fragments.n_genes,
            use_all = True,
            rg=rg,
        )
        fold["minibatches"] = minibatches
        
        fold["phases"] = {
            "test":[cells_test, genes_test]
        }
    return folds
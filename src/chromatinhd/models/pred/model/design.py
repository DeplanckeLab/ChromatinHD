n_cells_step = 200
n_regions_step = 500

import chromatinhd.models.pred
import chromatinhd.models.pred.model.better


def get_design(transcriptome, fragments):
    general_model_parameters = {
        "transcriptome": transcriptome,
        "fragments": fragments,
    }

    general_loader_parameters = {
        "fragments": fragments,
        "cellxregion_batch_size": n_cells_step * n_regions_step,
    }

    design = {}
    design["counter"] = {
        "model_cls": chromatinhd.models.pred.model.additive.Model,
        "model_parameters": {"dummy": True},
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": general_loader_parameters,
    }
    design["v20"] = {
        "model_cls": chromatinhd.models.pred.model.additive.Model,
        "model_parameters": {},
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": general_loader_parameters,
    }
    design["v22"] = {
        "model_cls": chromatinhd.models.pred.model.nonadditive.Model,
        "model_parameters": {},
        "loader_cls": chromatinhd.loaders.fragments.Fragments,
        "loader_parameters": general_loader_parameters,
    }

    design["v30"] = {
        "model_cls": chromatinhd.models.pred.model.better.Model,
        "model_params": dict(
            n_embedding_dimensions=100,
            n_layers_fragment_embedder=5,
            residual_fragment_embedder=False,
            n_layers_embedding2expression=5,
            residual_embedding2expression=False,
            dropout_rate_fragment_embedder=0.0,
            dropout_rate_embedding2expression=0.0,
        ),
        "train_params": dict(
            weight_decay=1e-1,
        ),
    }
    return design

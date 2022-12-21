import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import peakfreeatac as pfa
import peakfreeatac.data
import peakfreeatac.loaders
import peakfreeatac.loaders.fragmentmotif
import peakfreeatac.loaders.minibatching
import peakfreeatac.scorer

import pickle

device = "cuda:0"

folder_root = pfa.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
# dataset_name = "lymphoma+pbmc10k"
# dataset_name = "pbmc10k+lymphoma"
folder_data_preproc = folder_data / dataset_name

transcriptome = peakfreeatac.data.Transcriptome(
    folder_data_preproc / "transcriptome"
)

# fragments
# promoter_name, window = "1k1k", np.array([-1000, 1000])
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = peakfreeatac.data.Fragments(
    folder_data_preproc / "fragments" / promoter_name
)

# motifscan
motifscan = pfa.data.Motifscan(pfa.get_output() / "motifscans" / dataset_name / promoter_name)
# motifscan = pfa.data.Motifscan(pfa.get_output() / "motifscans" / dataset_name / promoter_name / "cutoff_001")

# create design to run
from design import get_design, get_folds_inference

class Prediction(pfa.flow.Flow):
    pass

# folds & minibatching
folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
folds = get_folds_inference(fragments, folds)

# design
design = get_design(dataset_name, transcriptome, motifscan, fragments, window = window)
design = {k:design[k] for k in [
    "v4",
    "v4_dummy",
    "v4_nn",
    # "v4_1k-1k",
    # "v4_10-10",
    # "v4_150-0",
    # "v4_0-150",
    # "v4_nn_dummy1",
    # "v4_nn_1k-1k",
    # "v4_split",
    # "v4_nn_split",
    # "v4_lw_split",
    # "v4_nn_lw",
    # "v4_nn_lw_split",
    # "v4_nn_lw_split_mean",
    # "v4_nn2_lw_split",
    # "v4_150-0_nn",
    # "v4_150-100-50-0_nn",
    # "v4_150-100-50-0",
    # "v4_1k-150-0_nn",
    # "v4_cutoff001",
    # "v4_nn_cutoff001",
]}
fold_slice = slice(0, 10)

# loss
def paircor(x, y, dim = 0, eps = 1e-6):
    divisor = (y.std(dim) * x.std(dim)) + eps
    cor = ((x - x.mean(dim, keepdims = True)) * (y - y.mean(dim, keepdims = True))).mean(dim) / divisor
    return cor
loss = lambda x, y: -paircor(x, y).mean() * 100

for prediction_name, design_row in design.items():
    print(prediction_name)
    prediction = Prediction(pfa.get_output() / "prediction_sequence_v5" / dataset_name / promoter_name / prediction_name)

    # loaders
    if "loaders" in globals():
        globals()["loaders"].terminate()
        del globals()["loaders"]
        import gc
        gc.collect()

    loaders = pfa.loaders.LoaderPool(
        design_row["loader_cls"],
        design_row["loader_parameters"],
        n_workers = 10,
        shuffle_on_iter = False
    )

    # load all models
    models = [pickle.load(open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "rb")) for fold_ix, fold in enumerate(folds[fold_slice])]

    # score
    outcome = transcriptome.X.dense()
    scorer = pfa.scorer.Scorer(models, folds[:len(models)], outcome = outcome, loaders = loaders, device = device)
    scorer.infer()

    scores_dir = (prediction.path / "scoring" / "overall")
    scores_dir.mkdir(parents = True, exist_ok = True)
    scores, genescores = scorer.score(fragments.var.index)

    scores.to_pickle(scores_dir / "scores.pkl")
    genescores.to_pickle(scores_dir / "genescores.pkl")


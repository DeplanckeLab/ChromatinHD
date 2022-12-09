import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import peakfreeatac as pfa
import peakfreeatac.fragments
import peakfreeatac.transcriptome
import peakfreeatac.loaders.fragmentmotif
import peakfreeatac.loaders.minibatching
import peakfreeatac.scorer

import pickle

device = "cuda:1"

folder_root = pfa.get_output()
folder_data = folder_root / "data"

# transcriptome
dataset_name = "lymphoma"
# dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

transcriptome = peakfreeatac.transcriptome.Transcriptome(
    folder_data_preproc / "transcriptome"
)

# fragments
# promoter_name, window = "1k1k", np.array([-1000, 1000])
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = peakfreeatac.fragments.Fragments(
    folder_data_preproc / "fragments" / promoter_name
)

# create design to run
from design import get_design, get_folds_inference

class Prediction(pfa.flow.Flow):
    pass

# folds & minibatching
folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
folds = get_folds_inference(fragments, folds)

# design
from design import get_design, get_folds_training
design = get_design(dataset_name, transcriptome, fragments, window = window)
design = {k:design[k] for k in [
    "v14"
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
    prediction = Prediction(pfa.get_output() / "prediction_sequence" / dataset_name / promoter_name / prediction_name)

    # loaders
    if "loaders" in globals():
        loaders.terminate()
        del loaders
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


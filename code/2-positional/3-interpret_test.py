import pandas as pd
import numpy as np

import peakfreeatac as pfa
import peakfreeatac.loaders.minibatching
import peakfreeatac.scorer

import pickle

device = "cuda:0"

folder_root = pfa.get_output()
folder_data = folder_root / "data"

for dataset_name_train, dataset_name in [
    # ["pbmc10k", "pbmc3k-pbmc10k"],
    # ["pbmc10k", "lymphoma-pbmc10k"],
    ["pbmc10k", "pbmc10k_gran-pbmc10k"],
]:
    # transcriptome
    folder_data_preproc = folder_data / dataset_name

    transcriptome = pfa.data.Transcriptome(
        folder_data_preproc / "transcriptome"
    )

    # fragments
    # promoter_name, window = "1k1k", np.array([-1000, 1000])
    promoter_name, window = "10k10k", np.array([-10000, 10000])
    # promoter_name, window = "20kpromoter", np.array([-10000, 0])
    promoters = pd.read_csv(
        folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
    )
    window_width = window[1] - window[0]

    fragments = pfa.data.Fragments(
        folder_data_preproc / "fragments" / promoter_name
    )

    # create design to run
    from design import get_design, get_folds_test

    class Prediction(pfa.flow.Flow):
        pass

    # folds & minibatching
    folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
    folds = get_folds_test(fragments, folds)

    # design
    from design import get_design
    design = get_design(dataset_name, transcriptome, fragments, window = window)
    design = {k:design[k] for k in [
        "counter",
        "counter_binary",
        # "v14",
        # "v14_dummy",
        # "v14_5freq",
        # "v14_3freq",
        # "v14_20freq",
        # "v14_50freq",
        # "v14_50freq_sum",
        # "v14_50freq_linear",
        # "v14_50freq_sigmoid",
        # "v14_50freq_sum_sigmoid",
        "v14_50freq_sum_sigmoid_initdefault",
        # "v14_50freq_sum_1emb_sigmoid",
        # "v15",
        # "v15_noselfatt",
        # "v15_att3",
    ]}
    fold_slice = slice(0, 10)

    for prediction_name, design_row in design.items():
        print(prediction_name)
        prediction_train = Prediction(pfa.get_output() / "prediction_positional" / dataset_name_train / promoter_name / prediction_name)
        prediction = Prediction(pfa.get_output() / "prediction_positional" / dataset_name / promoter_name / prediction_name)

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
        models = [pickle.load(open(prediction_train.path / ("model_" + str(fold_ix) + ".pkl"), "rb")) for fold_ix, fold in enumerate(folds[fold_slice])]

        # score
        outcome = transcriptome.X.dense()
        scorer = pfa.scorer.Scorer(models, folds[:len(models)], outcome = outcome, loaders = loaders, device = device, gene_ids = fragments.var.index)
        transcriptome_predicted_full, scores, genescores = scorer.score(return_prediction=True)

        scores_dir = (prediction.path / "scoring" / "overall")
        scores_dir.mkdir(parents = True, exist_ok = True)

        scores.to_pickle(scores_dir / "scores.pkl")
        genescores.to_pickle(scores_dir / "genescores.pkl")
        pickle.dump(transcriptome_predicted_full, (scores_dir / "transcriptome_predicted_full.pkl").open("wb"))


import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import peakfreeatac as pfa
import peakfreeatac.data
import peakfreeatac.loaders.fragmentmotif
import peakfreeatac.loaders.minibatching

import pickle

device = "cuda:0"

folder_root = pfa.get_output()
folder_data = folder_root / "data"

class Prediction(pfa.flow.Flow):
    pass

for dataset_name in [
    # "lymphoma",
    "pbmc10k_gran",
    # "pbmc10k",
    # "e18brain"
]:
    print(f"{dataset_name=}")
    # transcriptome
    folder_data_preproc = folder_data / dataset_name

    transcriptome = peakfreeatac.data.Transcriptome(
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

    fragments = peakfreeatac.data.Fragments(
        folder_data_preproc / "fragments" / promoter_name
    )

    # create design to run
    from design import get_design, get_folds_training
    design = get_design(dataset_name, transcriptome, fragments, window = window)
    design = {k:design[k] for k in [
        "counter",
        "counter_binary",
        # "v1",
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
        # "v14_50freq_sum_elu",
        "v14_50freq_sum_sigmoid_initdefault",
        # "v14_50freq_sum_sigmoid_drop05",
        # "v14_50freq_sum_1emb_sigmoid",
        # "v15",
        # "v15_noselfatt",
        # "v15_att3",
    ]}
    # fold_slice = slice(0, 1)
    fold_slice = slice(0, 5)
    # fold_slice = slice(1, 5)

    # folds & minibatching
    folds = pickle.load((fragments.path / "folds.pkl").open("rb"))
    folds = get_folds_training(fragments, folds)

    # loss
    def paircor(x, y, dim = 0, eps = 0.1):
        divisor = (y.std(dim) * x.std(dim)) + eps
        cor = ((x - x.mean(dim, keepdims = True)) * (y - y.mean(dim, keepdims = True))).mean(dim) / divisor
        return cor
    loss = lambda x, y: -paircor(x, y).mean() * 100

    # mseloss = torch.nn.MSELoss()
    # loss = lambda x, y:mseloss(x, y) * 100

    for prediction_name, design_row in design.items():
        print(f"{dataset_name=} {promoter_name=} {prediction_name=}")
        prediction = Prediction(pfa.get_output() / "prediction_positional" / dataset_name / promoter_name / prediction_name)

        # loaders
        print("collecting...")
        if "loaders" in globals():
            globals()["loaders"].terminate()
            del globals()["loaders"]
            import gc
            gc.collect()
        if "loaders_validation" in globals():
            globals()["loaders_validation"].terminate()
            del globals()["loaders_validation"]
            import gc
            gc.collect()
        print("collected")
        loaders = pfa.loaders.LoaderPool(
            design_row["loader_cls"],
            design_row["loader_parameters"],
            n_workers = 10
        )
        print("haha!")
        loaders_validation = pfa.loaders.LoaderPool(
            design_row["loader_cls"],
            design_row["loader_parameters"],
            n_workers = 5
        )
        loaders_validation.shuffle_on_iter = False

        models = []
        for fold_ix, fold in [(fold_ix, fold) for fold_ix, fold in enumerate(folds)][fold_slice]:
            # model
            model = design_row["model_cls"](**design_row["model_parameters"], loader = loaders.loaders[0])

            # optimizer
            params = model.get_parameters()

            # optimization
            optimize_every_step = 1
            # lr = 1e-2
            # lr = 1e-3
            lr = 1e-3
            # lr = 5e-4
            # optim = torch.optim.SGD(params, lr=lr * 1000)
            # optim = torch.optim.Adam(params, lr=lr, weight_decay=lr/2)
            optim = torch.optim.AdamW(params, lr=lr, weight_decay=lr/2)
            # optim = torch.optim.SGD(params, lr=lr*500, momentum=0.9)
            # optim = torch.optim.SparseAdam(params, lr=lr)
            n_epochs = 100
            checkpoint_every_epoch = 2

            # initialize loaders
            loaders.initialize(fold["minibatches_train"])
            loaders_validation.initialize(fold["minibatches_validation_trace"])

            # train
            import peakfreeatac.train
            outcome = transcriptome.X.dense()
            trainer = pfa.train.Trainer(
                model,
                loaders,
                loaders_validation,
                outcome,
                loss,
                optim,
                checkpoint_every_epoch = checkpoint_every_epoch,
                optimize_every_step = optimize_every_step,
                n_epochs = n_epochs,
                device = device
            )
            trainer.train()

            model = model.to("cpu")
            pickle.dump(model, open(prediction.path / ("model_" + str(fold_ix) + ".pkl"), "wb"))

            torch.cuda.empty_cache()
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            plotdata_validation = pd.DataFrame(trainer.trace.validation_steps).groupby("checkpoint").mean().reset_index()
            plotdata_train = pd.DataFrame(trainer.trace.train_steps).groupby("checkpoint").mean().reset_index()
            ax.plot(plotdata_validation["checkpoint"], plotdata_validation["loss"], label = "test")
            # ax.plot(plotdata_train["checkpoint"], plotdata_train["loss"], label = "train")
            ax.legend()
            fig.savefig(str(prediction.path / ("trace_" + str(fold_ix) + ".png")))
            plt.close()

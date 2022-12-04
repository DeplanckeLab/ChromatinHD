import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import peakfreeatac as pfa
import peakfreeatac.fragments
import peakfreeatac.transcriptome
import peakfreeatac.loaders.fragmentmotif
import peakfreeatac.loaders.minibatching

import pickle

folder_root = pfa.get_output()
folder_data = folder_root / "data"

# transcriptome
# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
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

# motifscan
motifscan_folder = pfa.get_output() / "motifscans" / dataset_name / promoter_name
motifscores = pickle.load(open(motifscan_folder / "motifscores.pkl", "rb"))

# model
parameters = {}
# from peakfreeatac.models.promotermotif.v1 import FragmentEmbeddingToExpression; model_name = "v1"
# from peakfreeatac.models.promotermotif.v2 import FragmentEmbeddingToExpression; model_name = "v2"
from peakfreeatac.models.promotermotif.v3 import FragmentEmbeddingToExpression; model_name = "v3"
# from peakfreeatac.models.promotermotif.v3 import FragmentEmbeddingToExpression; model_name = "v3_noweighting"; parameters = {"weight":False}

class Prediction(pfa.flow.Flow):
    pass
prediction = Prediction(pfa.get_output() / "prediction_sequence" / dataset_name / promoter_name / model_name)

mean_gene_expression = transcriptome.X.dense().mean(0)
n_components = motifscores.shape[1]
cutwindow = np.array([-150, 150])

model = FragmentEmbeddingToExpression(
    fragments.n_genes, mean_gene_expression, n_components, **parameters
)

transcriptome_X_dense = transcriptome.X.dense()

# optimizer
params = model.get_parameters()

# optimization
optimize_every_step = 10
lr = 1e-2 / optimize_every_step
optim = torch.optim.Adam(params, lr=lr)

# loss
cos = torch.nn.CosineSimilarity(dim = 0)
loss = lambda x_1, x_2: -cos(x_1, x_2).mean()

# trace
trace = []

prev_mse_train = None
prev_mse_test = None

# load folds
folds = pickle.load((fragments.path / "folds.pkl").open("rb"))

# loaders
n_cells_step = 1000
n_genes_step = 300
loaders = pfa.loaders.LoaderPool(
    peakfreeatac.loaders.fragmentmotif.Motifcounts,
    {"fragments":fragments, "motifscores":motifscores, "cellxgene_batch_size":n_cells_step * n_genes_step, "window":window, "cutwindow":cutwindow},
    n_workers = 20
)
loaders_validation = pfa.loaders.LoaderPool(
    peakfreeatac.loaders.fragmentmotif.Motifcounts,
    {"fragments":fragments, "motifscores":motifscores, "cellxgene_batch_size":n_cells_step * n_genes_step, "window":window, "cutwindow":cutwindow},
    n_workers = 5
)

models = []
for fold in folds:
    rg = np.random.RandomState(0)
    minibatches_train = pfa.loaders.minibatching.create_bins_random(
        fold["cells_train"],
        fold["genes_train"],
        n_cells_step=n_cells_step,
        n_genes_step=n_genes_step,
        n_genes_total=fragments.n_genes,
        rg=rg,
    )
    minibatches_validation = pfa.loaders.minibatching.create_bins_ordered(
        fold["cells_validation"],
        fold["genes_validation"],
        n_cells_step=n_cells_step,
        n_genes_step=n_genes_step,
        n_genes_total=fragments.n_genes,
        rg=rg,
    )
    minibatches_validation_trace = minibatches_validation[:5]

    loaders.initialize(minibatches_train)
    loaders_validation.initialize(minibatches_validation_trace)

    # train
    import peakfreeatac.train
    trainer = pfa.train.Trainer(
        model,
        loaders,
        loaders_validation,
        transcriptome_X_dense,
        loss,
        optim,
        checkpoint_every_step = 30,
        optimize_every_step = optimize_every_step,
        n_epochs = 5,
        device = "cuda"
    )
    trainer.train()

    model = model.to("cpu")
    models.append(model)

# postprocessing
pickle.dump(models, open(prediction.path / "models.pkl", "wb"))

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plotdata_validation = pd.DataFrame(trainer.trace.validation_steps).groupby("checkpoint").mean().reset_index()
plotdata_train = pd.DataFrame(trainer.trace.train_steps).groupby("checkpoint").mean().reset_index()
ax.plot(plotdata_validation["checkpoint"], plotdata_validation["loss"])
ax.plot(plotdata_train["checkpoint"], plotdata_train["loss"])
fig.savefig(prediction.path / "trace.png")

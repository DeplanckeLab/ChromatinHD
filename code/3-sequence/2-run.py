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
# promoter_name, window = "4k2k", (2000, 4000)
promoter_name, window = "10k10k", np.array([-10000, 10000])
promoters = pd.read_csv(
    folder_data_preproc / ("promoters_" + promoter_name + ".csv"), index_col=0
)
window_width = window[1] - window[0]

fragments = peakfreeatac.fragments.Fragments(
    folder_data_preproc / "fragments" / promoter_name
)

# motifscores
motifscores = pickle.load(open(folder_data_preproc / "motifscores.pkl", "rb"))

# model
# from peakfreeatac.models.promotermotif.v1 import FragmentEmbeddingToExpression
from peakfreeatac.models.promotermotif.v2 import FragmentEmbeddingToExpression

mean_gene_expression = transcriptome.X.dense().mean(0)
n_components = motifscores.shape[1]
cutwindow = np.array([-150, 150])

model = FragmentEmbeddingToExpression(
    fragments.n_genes, mean_gene_expression, n_components
)

n_epochs = 50
device = "cuda"

transcriptome_X = transcriptome.X.to(device)
transcriptome_X_dense = transcriptome_X.dense()

# optimizer
params = model.get_parameters()

# lr = 1.0
trace_every_step = 30
optimize_every_step = 1
lr = 1e-4 / optimize_every_step
# optim = torch.optim.SGD(params, lr=lr, momentum=0.3)
optim = torch.optim.Adam(params, lr=lr)
loss = torch.nn.MSELoss()

# trace
trace = []

prev_mse_train = None
prev_mse_test = None

# minibatching
cells_all = np.arange(fragments.n_cells)
genes_all = np.arange(fragments.n_genes)

n_cells_step = 1000
n_genes_step = 300

cells_train = cells_all[: int(len(cells_all) * 4 / 5)]
genes_train = genes_all[: int(len(genes_all) * 4 / 5)]

cells_validation = cells_all[[cell for cell in cells_all if cell not in cells_train]]
genes_validation = genes_train
# genes_validation = genes_all[[gene for gene in genes_all if gene not in genes_train]]
# print(genes_validation)

rg = np.random.RandomState(2)
bins_train = pfa.loaders.minibatching.create_bins_random(
    cells_train,
    genes_train,
    n_cells_step=n_cells_step,
    n_genes_step=n_genes_step,
    n_genes_total=fragments.n_genes,
    rg=rg,
)
bins_validation = pfa.loaders.minibatching.create_bins_ordered(
    cells_validation,
    genes_validation,
    n_cells_step=n_cells_step,
    n_genes_step=n_genes_step,
    n_genes_total=fragments.n_genes,
    rg=rg,
)
bins_validation_trace = bins_validation[:5]

model = model.to(device)

# loaders
loaderpool = pfa.loaders.LoaderPool(
    peakfreeatac.loaders.fragmentmotif.Motifcounts,
    (fragments, motifscores, n_cells_step * n_genes_step, window, cutwindow),
    n_workers = 10
)
loaderpool.initialize(bins_train)
loaderpool_validation = pfa.loaders.LoaderPool(
    peakfreeatac.loaders.fragmentmotif.Motifcounts,
    (fragments, motifscores, n_cells_step * n_genes_step, window, cutwindow),
    n_workers = 5
)
loaderpool_validation.initialize(bins_validation_trace)

# train
step_ix = 0
trace_validation = []

for epoch in tqdm.tqdm(range(n_epochs)):
    # train
    for data_train in loaderpool:
        if (step_ix % trace_every_step) == 0:
            with torch.no_grad():
                print("tracing")
                mse_validation = []
                mse_validation_dummy = []
                for data_validation in tqdm.tqdm(loaderpool_validation, leave = False):
                    motifcounts = torch.from_numpy(data_validation.motifcounts).to(torch.float).to(device)
                    local_cellxgene_ix = torch.from_numpy(data_validation.local_cellxgene_ix).to(device)

                    transcriptome_subset = transcriptome_X_dense[data_validation.cells_oi, :][:, data_validation.genes_oi].to(device)

                    transcriptome_predicted = model(motifcounts, local_cellxgene_ix, data_validation.n_cells, data_validation.n_genes, data_validation.genes_oi)

                    mse = loss(transcriptome_predicted, transcriptome_subset)

                    mse_validation.append(mse.item())
                    mse_validation_dummy.append(((transcriptome_subset - transcriptome_subset.mean(0, keepdim = True)) ** 2).mean().item())

                    loaderpool_validation.submit_next()
                    
                loaderpool_validation.reset()
                mse_validation = np.mean(mse_validation)
                mse_validation_dummy = np.mean(mse_validation_dummy)
                
                print(mse_validation - mse_validation_dummy)
                
                trace_validation.append({
                    "epoch":epoch,
                    "step":step_ix,
                    "mse":mse_validation,
                    "mse_dummy":mse_validation_dummy
                })
        torch.set_grad_enabled(True)
    
        motifcounts = torch.from_numpy(data_train.motifcounts).to(torch.float).to(device)
        local_cellxgene_ix = torch.from_numpy(data_train.local_cellxgene_ix).to(device)
        
        transcriptome_subset = transcriptome_X_dense[data_train.cells_oi, :][:, data_train.genes_oi].to(device)
        
        transcriptome_predicted = model(motifcounts, local_cellxgene_ix, data_train.n_cells, data_train.n_genes, data_train.genes_oi)
        
        mse = loss(transcriptome_predicted, transcriptome_subset)

        mse.backward()
        
        if (step_ix % optimize_every_step) == 0:
            optim.step()
            optim.zero_grad()
        
        step_ix += 1

        loaderpool.submit_next()

    # reshuffle the order of the bins
    loaderpool.reset()
    # bins_train = [bins_train[i] for i in np.random.choice(len(bins_train), len(bins_train), replace = False)]

if isinstance(trace_validation, list):
    trace_validation = pd.DataFrame(list(trace_validation))

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plotdata = trace_validation.groupby("step").mean().reset_index()
ax.plot(plotdata["step"], plotdata["mse"], zorder = 6, color = "orange")
ax.plot(plotdata["step"], plotdata["mse_dummy"], zorder = 6, color = "red")
fig.savefig("trace.png")

pickle.dump(model, open("model.pkl", "wb"))
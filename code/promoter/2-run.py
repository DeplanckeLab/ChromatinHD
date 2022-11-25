import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_style('ticks')

import pickle

import itertools

import torch

import tqdm.auto as tqdm

import peakfreeatac as pfa
import peakfreeatac.fragments
import peakfreeatac.transcriptome


folder_root = pfa.get_output()
folder_data = folder_root / "data"

# dataset_name = "lymphoma"
dataset_name = "pbmc10k"
# dataset_name = "e18brain"
folder_data_preproc = folder_data / dataset_name

promoter_name = "10k10k"

transcriptome = peakfreeatac.transcriptome.Transcriptome(folder_data_preproc / "transcriptome")
fragments = peakfreeatac.fragments.Fragments(folder_data_preproc / "fragments" / promoter_name)


mean_gene_expression = transcriptome.X.dense().mean(0)


folds = pickle.load(open(fragments.path / "folds.pkl", "rb"))


# from peakfreeatac.models.promoter.v6 import FragmentsToExpression; model_name = "v6"
from peakfreeatac.models.promoter.v5 import FragmentsToExpression; model_name = "v5"
from peakfreeatac.models.promoter.v7 import FragmentsToExpression; model_name = "v7"
from peakfreeatac.models.promoter.v8 import FragmentsToExpression; model_name = "v8"
from peakfreeatac.models.promoter.v10 import FragmentsToExpression; model_name = "v10"
# from peakfreeatac.models.promoter.v3 import FragmentsToExpression; model_name = "v3"


class Prediction(pfa.flow.Flow):
    pass
prediction = Prediction(pfa.get_output() / "prediction_promoter" / dataset_name / promoter_name / model_name)



models = []
for i in range(len(folds)):
    model = FragmentsToExpression(
        fragments.n_genes,
        mean_gene_expression
    )
    models.append(model)
    
    
n_steps = 1000
trace_epoch_every = 10

lr = 1.0

# choose which models to infer
# model_ixs = [0, 1, 2, 3, 4]
model_ixs = [0]

transcriptome_X = transcriptome.X.to("cuda")
transcriptome_X_dense = transcriptome_X.dense()

coordinates = fragments.coordinates.to("cuda")
mapping = fragments.mapping.to("cuda")


for fold, model in zip([folds[i] for i in model_ixs], [models[i] for i in model_ixs]):
    params = model.get_parameters()

    optim = torch.optim.SGD(
        params,
        lr = lr
    )
    loss = torch.nn.MSELoss(reduction = "mean")
    
    splits_training = [split.to("cuda") for split in fold if split.phase == "train"]
    splits_test = [split.to("cuda") for split in fold if split.phase == "validation"]
    model = model.to("cuda").train(True)

    trace = []

    prev_mse_train = None
    prev_mse_test = None
    for epoch in tqdm.tqdm(range(n_steps)):
        # trace
        if (epoch % trace_epoch_every) == 0:
            # mse
            mse_test = []
            mse_train = []
            for split in itertools.chain(splits_training, splits_test):
                with torch.no_grad():
                    expression_predicted = model(
                        coordinates[split.fragments_selected],
                        split.fragment_cellxgene_ix,
                        mapping[split.fragments_selected, 1],
                        split.cell_n,
                        split.gene_n,
                        split.gene_ix
                    )
                    
                    transcriptome_subset = transcriptome_X_dense[split.cell_ix, split.gene_ix]
                    mse = loss(expression_predicted, transcriptome_subset)
                    
                    if split.phase == "train":
                        mse_train.append(mse.detach().cpu().item())
                    else:
                        mse_test.append(mse.detach().cpu().item())
            mse_train = np.mean(mse_train)
            mse_test = np.mean(mse_test)
            
            # train mse
            text = f"{epoch} {mse_train:.6f}"

            if prev_mse_train is not None:
                text += f" Δ{prev_mse_train-mse_train:.1e}"

            prev_mse_train = mse_train
            
            # mse test
            text += f" {mse_test:.6f}"
            
            if prev_mse_test is not None:
                text += f" Δ{prev_mse_test-mse_test:.1e}"
                
            prev_mse_test = mse_test
            
            print(text)
            
            trace.append({
                "mse_train":mse_train,
                "mse_test":mse_test,
                "epoch":epoch
            })

        # train
        for split in splits_training:
            expression_predicted = model(
                coordinates[split.fragments_selected],
                split.fragment_cellxgene_ix,
                mapping[split.fragments_selected, 1],
                split.cell_n,
                split.gene_n,
                split.gene_ix
            )

            # transcriptome_subset = transcriptome_X.dense_subset(split.cell_ix)[:, split.gene_ix]
            transcriptome_subset = transcriptome_X_dense[split.cell_ix, split.gene_ix]

            mse = loss(expression_predicted, transcriptome_subset)

            mse.backward()
            optim.step()
            optim.zero_grad()

        # reshuffle the order of the splits
        splits_training = [splits_training[i] for i in np.random.choice(len(splits_training), len(splits_training), replace = False)]
            
            
            
if isinstance(trace, list):
    trace = pd.DataFrame(list(trace))
    
fig, ax = plt.subplots()
plotdata = trace.groupby("epoch").mean().reset_index()
ax.plot(plotdata["epoch"], plotdata["mse_train"], zorder = 6, color = "red")
ax.plot(plotdata["epoch"], plotdata["mse_test"], zorder = 6, color = "orange")
fig.savefig("trace.png")


# move splits back to cpu
# otherwise if you try to load them back in they might want to immediately go to gpu
models = [model.to("cpu") for model in models]

pfa.save(models, open(prediction.path / "models.pkl", "wb"))
import tqdm.auto as tqdm
import torch
import numpy as np
from chromatinhd.train import Trace


def paircor(x, y, dim=0, eps=0.1):
    divisor = (y.std(dim) * x.std(dim)) + eps
    cor = ((x - x.mean(dim, keepdims=True)) * (y - y.mean(dim, keepdims=True))).mean(
        dim
    ) / divisor
    return cor


def filter_minibatch_sets(minibatch_sets, improved):
    new_minibatch_sets = []
    for minibatch_set in minibatch_sets:
        tasks = [
            minibatch.filter_genes(improved) for minibatch in minibatch_set["tasks"]
        ]
        tasks = [minibatch for minibatch in tasks if len(minibatch.genes_oi) > 0]
        new_minibatch_sets.append({"tasks": tasks})
    return new_minibatch_sets


class Trainer:
    def __init__(
        self,
        model,
        loaders,
        loaders_validation,
        outcome,
        loss,
        optim,
        device="cuda",
        n_epochs=30,
        checkpoint_every_epoch=1,
        optimize_every_step=10,
    ):
        self.model = model
        self.loaders = loaders
        self.loaders_validation = loaders_validation

        outcome = outcome / outcome.std(0, keepdims=True)
        self.outcome = outcome
        self.trace = Trace()

        self.loss = loss
        self.optim = optim

        self.step_ix = 0
        self.epoch = 0
        self.n_epochs = n_epochs

        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.optimize_every_step = optimize_every_step

        self.device = device

    def train(self, minibatches_train_sets, minibatches_validation):
        self.model = self.model.to(self.device)
        self.outcome = self.outcome.to(self.device)

        continue_training = True

        prev_gene_loss = None
        improved = None

        self.loaders.initialize(next_task_sets=minibatches_train_sets)
        self.loaders_validation.initialize(minibatches_validation)

        n_steps_total = self.n_epochs * len(self.loaders)
        self.pbar = tqdm.tqdm(total=n_steps_total)

        while (self.epoch < self.n_epochs) and (continue_training):
            # checkpoint if necessary
            if (self.epoch % self.checkpoint_every_epoch) == 0:
                with torch.no_grad():
                    gene_loss = np.zeros(self.outcome.shape[1])
                    for data_validation in self.loaders_validation:
                        data_validation = data_validation.to(self.device)

                        transcriptome_subset = self.outcome[
                            data_validation.cells_oi, :
                        ][:, data_validation.genes_oi].to(self.device)

                        transcriptome_predicted = self.model(data_validation)

                        loss = self.loss(transcriptome_predicted, transcriptome_subset)
                        self.trace.append(
                            loss.item(), self.epoch, self.step_ix, "validation"
                        )

                        gene_loss[data_validation.genes_oi] = (
                            gene_loss[data_validation.genes_oi]
                            + paircor(transcriptome_predicted, transcriptome_subset)
                            .detach()
                            .cpu()
                            .numpy()
                        )

                        self.loaders_validation.submit_next()
                print(f"{'•'} {self.epoch}/{self.n_epochs} {'step':>15}")
                self.trace.checkpoint()

                # compare with previous loss per gene
                if prev_gene_loss is not None:
                    improvement = gene_loss - prev_gene_loss

                    if improved is not None:
                        improved = improved & (improvement > 0)
                    else:
                        improved = improvement > 0
                    print(f"{improved.mean():.1%}")

                    # stop training once less than 1% of genes are still being optimized
                    if improved.mean() < 0.01:
                        break

                    minibatches_train_sets = filter_minibatch_sets(
                        minibatches_train_sets, improved
                    )
                    self.loaders.initialize(next_task_sets=minibatches_train_sets)

                prev_gene_loss = gene_loss.copy()

            # train
            for data_train in self.loaders:
                data_train = data_train.to(self.device)

                # get subset of transcriptomics data
                transcriptome_subset = self.outcome[data_train.cells_oi, :][
                    :, data_train.genes_oi
                ].to(self.device)

                transcriptome_predicted = self.model(data_train)

                loss = self.loss(transcriptome_predicted, transcriptome_subset)
                loss.backward()

                # check if optimization
                if (self.step_ix % self.optimize_every_step) == 0:
                    self.optim.step()
                    self.optim.zero_grad()

                self.step_ix += 1
                self.pbar.update()

                self.trace.append(loss.item(), self.epoch, self.step_ix, "train")

                self.loaders.submit_next()
            self.epoch += 1

        self.model = self.model.to("cpu")
        self.outcome = self.outcome.to("cpu")

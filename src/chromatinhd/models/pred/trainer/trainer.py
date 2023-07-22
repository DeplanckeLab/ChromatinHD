import tqdm.auto as tqdm
import torch
import numpy as np
from chromatinhd.train import Trace

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


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
        loaders_train,
        loaders_validation,
        minibatcher_train,
        minibatcher_validation,
        optim,
        device="cuda",
        n_epochs=30,
        checkpoint_every_epoch=1,
        optimize_every_step=10,
    ):
        self.model = model
        self.loaders_train = loaders_train
        self.loaders_validation = loaders_validation

        self.trace = Trace()

        self.optim = optim

        self.step_ix = 0
        self.epoch = 0
        self.n_epochs = n_epochs

        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.optimize_every_step = optimize_every_step

        self.minibatcher_train = minibatcher_train
        self.minibatcher_validation = minibatcher_validation

        self.device = device

    def train(self):
        self.model = self.model.to(self.device)

        continue_training = True

        prev_gene_loss = None
        improved = None

        self.loaders_train.initialize(self.minibatcher_train)
        self.loaders_validation.initialize(self.minibatcher_validation)

        n_steps_total = self.n_epochs * len(self.loaders_train)
        pbar = tqdm.tqdm(total=n_steps_total, leave=False)

        while (self.epoch < self.n_epochs) and (continue_training):
            # checkpoint if necessary
            if (self.epoch % self.checkpoint_every_epoch) == 0:
                with torch.no_grad():
                    gene_loss = np.zeros(self.minibatcher_train.n_genes)
                    for data_validation in self.loaders_validation:
                        data_validation = data_validation.to(self.device)

                        gene_loss_mb = (
                            self.model.forward_gene_loss(data_validation)
                            .cpu()
                            .detach()
                            .numpy()
                        )

                        gene_loss[data_validation.minibatch.genes_oi] = (
                            gene_loss[data_validation.minibatch.genes_oi] + gene_loss_mb
                        )

                self.trace.append(
                    gene_loss.mean().item(),
                    self.epoch,
                    self.step_ix,
                    "validation",
                )
                logger.info(f"{'•'} {self.epoch}/{self.n_epochs} {'step':>15}")
                self.trace.checkpoint(logger=logger)

                # compare with previous loss per gene
                if prev_gene_loss is not None:
                    improvement = gene_loss - prev_gene_loss

                    if improved is not None:
                        improved = improved & (improvement < 0)
                    else:
                        improved = improvement < 0
                    logger.info(f"{improved.mean():.1%}")

                    # stop training once less than 1% of genes are still being optimized
                    if improved.mean() < 0.001:
                        break

                    self.minibatcher_train.genes = np.arange(
                        self.minibatcher_train.n_genes
                    )[improved]

                prev_gene_loss = gene_loss.copy()

            # train
            for data_train in self.loaders_train:
                data_train = data_train.to(self.device)

                loss = self.model.forward_loss(data_train)
                loss.backward()

                # check if optimization
                if (self.step_ix % self.optimize_every_step) == 0:
                    self.optim.step()
                    self.optim.zero_grad()

                self.step_ix += 1
                pbar.update()

                self.trace.append(loss.item(), self.epoch, self.step_ix, "train")
            self.epoch += 1

        pbar.update(n_steps_total)
        pbar.close()

        self.model = self.model.to("cpu")

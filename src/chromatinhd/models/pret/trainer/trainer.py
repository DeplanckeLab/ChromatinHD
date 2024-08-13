import logging

import numpy as np
import torch
import tqdm.auto as tqdm

from chromatinhd import get_default_device
from chromatinhd.train import Trace

logger = logging.getLogger(__name__)


def paircor(x, y, dim=0, eps=0.1):
    divisor = (y.std(dim) * x.std(dim)) + eps
    cor = ((x - x.mean(dim, keepdims=True)) * (y - y.mean(dim, keepdims=True))).mean(dim) / divisor
    return cor


def filter_minibatch_sets(minibatch_sets, improved):
    new_minibatch_sets = []
    for minibatch_set in minibatch_sets:
        tasks = [minibatch.filter_regions(improved) for minibatch in minibatch_set["tasks"]]
        tasks = [minibatch for minibatch in tasks if len(minibatch.regions_oi) > 0]
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
        device=None,
        n_epochs=30,
        checkpoint_every_epoch=1,
        optimize_every_step=10,
        pbar=True,
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

        self.device = device if device is not None else get_default_device()

        self.pbar = pbar

    def train(self):
        self.model = self.model.to(self.device)

        continue_training = True

        prev_region_loss = None
        improved = None

        self.loaders_train.initialize(self.minibatcher_train)
        self.loaders_validation.initialize(self.minibatcher_validation)

        n_steps_total = self.n_epochs * len(self.loaders_train)
        pbar = tqdm.tqdm(total=n_steps_total, leave=False) if self.pbar else None

        while (self.epoch < self.n_epochs) and (continue_training):
            # checkpoint if necessary
            if (self.epoch % self.checkpoint_every_epoch) == 0:
                self.model = self.model.eval()
                with torch.no_grad():
                    region_loss = np.zeros(self.minibatcher_train.n_regions)
                    for data_validation in self.loaders_validation:
                        data_validation = data_validation.to(self.device)

                        region_loss_mb = self.model.forward_region_loss(data_validation).cpu().detach().numpy()

                        region_loss[data_validation.minibatch.regions_oi] = (
                            region_loss[data_validation.minibatch.regions_oi] + region_loss_mb
                        )

                self.trace.append(
                    region_loss.mean().item(),
                    self.epoch,
                    self.step_ix,
                    "validation",
                )
                logger.info(f"{'•'} {self.epoch}/{self.n_epochs} {'step':>15}")
                self.trace.checkpoint(logger=logger)

                # compare with previous loss per region
                if prev_region_loss is not None:
                    improvement = region_loss - prev_region_loss

                    if improved is not None:
                        improved = improved & (improvement < 0)
                    else:
                        improved = improvement < 0
                    logger.info(f"{improved.mean():.1%}")

                    # stop training once less than 1% of regions are still being optimized
                    if improved.mean() < 0.01:
                        break

                    self.minibatcher_train.regions = np.arange(self.minibatcher_train.n_regions)[improved]

                prev_region_loss = region_loss.copy()

                if pbar is not None:
                    if prev_region_loss is None:
                        pbar.set_description(f"epoch {self.epoch}/{self.n_epochs}")
                    else:
                        pbar.set_description(
                            f"epoch {self.epoch}/{self.n_epochs} validation loss {region_loss.mean():.2f}"
                        )
            self.model = self.model.train()

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
                if pbar is not None:
                    pbar.update()

                self.trace.append(loss.item(), self.epoch, self.step_ix, "train")
            self.epoch += 1

        if pbar is not None:
            pbar.update(n_steps_total)
            pbar.close()

        self.model = self.model.to("cpu")

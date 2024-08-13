import logging

import numpy as np
import torch
import tqdm.auto as tqdm

from chromatinhd import get_default_device
from chromatinhd.train import Trace

logger = logging.getLogger(__name__)
logger.handlers = []
logger.propagate = False


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
        lr_scheduler=None,
        lr_scheduler_params=None,
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

        self.scheduler = None
        if lr_scheduler == "linear":
            if lr_scheduler_params is None:
                lr_scheduler_params = {}
            self.scheduler = torch.optim.lr_scheduler.LinearLR(optim, **lr_scheduler_params)

    def train(self):
        self.model = self.model.to(self.device)

        continue_training = True

        prev_region_loss = None
        improved = None

        self.loaders_train.initialize(self.minibatcher_train)
        self.loaders_validation.initialize(self.minibatcher_validation)

        pbar = tqdm.tqdm(total=self.n_epochs, leave=False) if self.pbar else None

        n_regions = len(self.minibatcher_train.regions)
        region_mapping = np.zeros(self.minibatcher_train.regions.max() + 1, dtype=np.int64)
        region_mapping[self.minibatcher_train.regions] = np.arange(len(self.minibatcher_train.regions))
        region_mapping_inv = self.minibatcher_train.regions

        while (self.epoch < self.n_epochs) and (continue_training):
            # checkpoint if necessary
            if (self.epoch % self.checkpoint_every_epoch) == 0:
                self.model = self.model.eval()
                with torch.no_grad():
                    region_loss = np.zeros(n_regions)
                    for data_validation in self.loaders_validation:
                        data_validation = data_validation.to(self.device)

                        region_loss_mb = self.model.forward_region_loss(data_validation).cpu().detach().numpy()

                        region_loss[region_mapping[data_validation.minibatch.regions_oi]] = (
                            region_loss[region_mapping[data_validation.minibatch.regions_oi]] + region_loss_mb
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

                    print(f"{improved.mean():.1%}")

                    # stop training once less than 1% of regions are still being optimized
                    if improved.mean() == 0.00:
                        break

                    self.minibatcher_train.regions = region_mapping_inv[improved]

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

                self.trace.append(loss.item(), self.epoch, self.step_ix, "train")
            self.epoch += 1
            if pbar is not None:
                pbar.update()

            if self.scheduler is not None:
                self.scheduler.step()

        if pbar is not None:
            pbar.update(self.n_epochs)
            pbar.close()

        self.model = self.model.to("cpu")


import time


class catchtime(object):
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.t = time.time() - self.t
        # print(self.name, self.t)


class TrainerPerFeature:
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
        lr_scheduler=None,
        lr_scheduler_params=None,
        warmup_epochs=0,
        early_stopping=True,
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

        self.scheduler = None
        if lr_scheduler == "linear":
            if lr_scheduler_params is None:
                lr_scheduler_params = {}
            self.scheduler = torch.optim.lr_scheduler.LinearLR(optim, **lr_scheduler_params)

        self.warmup_epochs = warmup_epochs
        self.early_stopping = early_stopping

    def train(self):
        self.model = self.model.to(self.device)

        continue_training = True

        self.loaders_train.initialize(self.minibatcher_train)
        self.loaders_validation.initialize(self.minibatcher_validation)

        pbar = tqdm.tqdm(total=self.n_epochs, leave=False) if self.pbar else None

        n_regions = self.minibatcher_train.regions.max() + 1
        prev_region_loss = None
        improved = np.ones(n_regions, dtype=bool)

        while (self.epoch < self.n_epochs) and (continue_training):
            with catchtime("validation"):
                # checkpoint if necessary
                if (self.epoch % self.checkpoint_every_epoch) == 0:
                    self.model = self.model.eval()
                    with torch.no_grad():
                        region_loss = np.zeros(n_regions)
                        for data_validation in self.loaders_validation:
                            data_validation = data_validation.to(self.device)

                            region_loss_mb = self.model.forward_region_loss(data_validation).cpu().detach().numpy()

                            region_loss[data_validation.minibatch.regions_oi] = region_loss_mb

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
                        if self.early_stopping and (improved.mean() == 0.00):
                            break

                        if self.early_stopping and (self.epoch >= self.warmup_epochs):
                            self.minibatcher_train.regions = np.where(improved)[0]
                            self.loaders_train.start()
                            # self.minibatcher_validation.regions = np.where(improved)[0]

                    prev_region_loss = region_loss.copy()

                    if pbar is not None:
                        if prev_region_loss is None:
                            pbar.set_description(f"epoch {self.epoch}/{self.n_epochs}")
                        elif improved is None:
                            pbar.set_description(
                                f"epoch {self.epoch}/{self.n_epochs} validation loss {region_loss.mean():.2f}"
                            )
                        else:
                            pbar.set_description(
                                f"epoch {self.epoch}/{self.n_epochs} validation loss {region_loss.mean():.2f} regions {improved.mean():.1%}"
                            )
            self.model = self.model.train()

            # train
            for data_train in self.loaders_train:
                data_train = data_train.to(self.device)

                loss = self.model.forward_backward(data_train)

                # check if optimization
                if (self.step_ix % self.optimize_every_step) == 0:
                    self.optim.step(data_train.minibatch.regions_oi)
                    self.optim.zero_grad(data_train.minibatch.regions_oi)

                self.step_ix += 1

                self.trace.append(loss.item(), self.epoch, self.step_ix, "train")
            self.epoch += 1
            if pbar is not None:
                pbar.update()

        if pbar is not None:
            pbar.update(self.n_epochs)
            pbar.close()

        self.model = self.model.to("cpu")


# class TrainerPerFeature:
#     def __init__(
#         self,
#         model,
#         loaders_train,
#         loaders_validation,
#         minibatcher_train,
#         minibatcher_validation,
#         optim,
#         device=None,
#         n_epochs=30,
#         checkpoint_every_epoch=1,
#         optimize_every_step=10,
#         pbar=True,
#         lr_scheduler=None,
#         lr_scheduler_params=None,
#         warmup_epochs=0,
#         early_stopping=True,
#     ):
#         self.model = model
#         self.loaders_train = loaders_train
#         self.loaders_validation = loaders_validation

#         self.trace = Trace()

#         self.optim = optim

#         self.step_ix = 0
#         self.epoch = 0
#         self.n_epochs = n_epochs

#         self.checkpoint_every_epoch = checkpoint_every_epoch
#         self.optimize_every_step = optimize_every_step

#         self.minibatcher_train = minibatcher_train
#         self.minibatcher_validation = minibatcher_validation

#         self.device = device if device is not None else get_default_device()

#         self.pbar = pbar

#         self.scheduler = None
#         if lr_scheduler == "linear":
#             if lr_scheduler_params is None:
#                 lr_scheduler_params = {}
#             self.scheduler = torch.optim.lr_scheduler.LinearLR(optim, **lr_scheduler_params)

#         self.warmup_epochs = warmup_epochs
#         self.early_stopping = early_stopping

#     def train(self):
#         self.model = self.model.to(self.device)

#         continue_training = True

#         self.loaders_train.initialize(self.minibatcher_train)
#         self.loaders_validation.initialize(self.minibatcher_validation)

#         pbar = tqdm.tqdm(total=self.n_epochs, leave=False) if self.pbar else None

#         n_regions = self.minibatcher_train.regions.max() + 1
#         prev_region_loss = None
#         improved = np.ones(n_regions, dtype=bool)

#         datas_train = []
#         for data_train in self.loaders_train:
#             datas_train.append(data_train)
#         datas_validation = []
#         for data_validation in self.loaders_validation:
#             datas_validation.append(data_validation)

#         while (self.epoch < self.n_epochs) and (continue_training):
#             with catchtime("validation"):
#                 # checkpoint if necessary
#                 if (self.epoch % self.checkpoint_every_epoch) == 0:
#                     self.model = self.model.eval()
#                     with torch.no_grad():
#                         region_loss = np.zeros(n_regions)
#                         for data_validation in datas_validation:
#                             data_validation = data_validation.to(self.device)

#                             region_loss_mb = self.model.forward_region_loss(data_validation).cpu().detach().numpy()

#                             region_loss[data_validation.minibatch.regions_oi] = region_loss_mb

#                     self.trace.append(
#                         region_loss.mean().item(),
#                         self.epoch,
#                         self.step_ix,
#                         "validation",
#                     )
#                     logger.info(f"{'•'} {self.epoch}/{self.n_epochs} {'step':>15}")
#                     self.trace.checkpoint(logger=logger)

#                     # compare with previous loss per region
#                     if prev_region_loss is not None:
#                         improvement = region_loss - prev_region_loss

#                         if improved is not None:
#                             improved = improved & (improvement < 0)
#                         else:
#                             improved = improvement < 0
#                         logger.info(f"{improved.mean():.1%}")

#                         # stop training once less than 1% of regions are still being optimized
#                         if self.early_stopping and (improved.mean() == 0.00):
#                             break

#                         if self.early_stopping and (self.epoch >= self.warmup_epochs):
#                             self.minibatcher_train.regions = np.where(improved)[0]
#                             # self.minibatcher_validation.regions = np.where(improved)[0]

#                     prev_region_loss = region_loss.copy()

#                     if pbar is not None:
#                         if prev_region_loss is None:
#                             pbar.set_description(f"epoch {self.epoch}/{self.n_epochs}")
#                         elif improved is None:
#                             pbar.set_description(
#                                 f"epoch {self.epoch}/{self.n_epochs} validation loss {region_loss.mean():.2f}"
#                             )
#                         else:
#                             pbar.set_description(
#                                 f"epoch {self.epoch}/{self.n_epochs} validation loss {region_loss.mean():.2f} regions {improved.mean():.1%}"
#                             )
#             self.model = self.model.train()

#             # train
#             for i, data_train in enumerate(datas_train):
#                 if not improved[i]:
#                     continue
#                 data_train = data_train.to(self.device)

#                 loss = self.model.forward_backward(data_train)

#                 # check if optimization
#                 if (self.step_ix % self.optimize_every_step) == 0:
#                     self.optim.step(data_train.minibatch.regions_oi)
#                     self.optim.zero_grad(data_train.minibatch.regions_oi)

#                 self.step_ix += 1

#                 self.trace.append(loss.item(), self.epoch, self.step_ix, "train")
#             self.epoch += 1
#             if pbar is not None:
#                 pbar.update()

#         if pbar is not None:
#             pbar.update(self.n_epochs)
#             pbar.close()

#         self.model = self.model.to("cpu")


class SharedTrainer:
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
        lr_scheduler=None,
        lr_scheduler_params=None,
        early_stopping=True,
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

        self.scheduler = None
        self.lr_scheduler = lr_scheduler
        if lr_scheduler == "linear":
            if lr_scheduler_params is None:
                lr_scheduler_params = {}
            self.scheduler = torch.optim.lr_scheduler.LinearLR(optim, **lr_scheduler_params)
        elif lr_scheduler == "plateau":
            if lr_scheduler_params is None:
                lr_scheduler_params = {}
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, **lr_scheduler_params)

        self.early_stopping = early_stopping

    def train(self):
        self.model = self.model.to(self.device)

        continue_training = True

        self.loaders_train.initialize(self.minibatcher_train)
        self.loaders_validation.initialize(self.minibatcher_validation)

        n_steps_total = self.n_epochs * len(self.loaders_train)
        pbar = tqdm.tqdm(total=n_steps_total, leave=False) if self.pbar else None

        prev_loss = None

        data_train = next(iter(self.loaders_train)).to(self.device)
        data_validation = next(iter(self.loaders_validation)).to(self.device)

        while (self.epoch < self.n_epochs) and (continue_training):
            # checkpoint if necessary
            if (self.epoch % self.checkpoint_every_epoch) == 0:
                self.model = self.model.eval()
                with torch.no_grad():
                    # loss = 0.0
                    # for data_validation in self.loaders_validation:
                    #     data_validation = data_validation.to(self.device)

                    #     loss += self.model.forward_loss(data_validation).cpu().detach().item()
                    loss = self.model.forward_loss(data_validation).cpu().detach().item()

                self.trace.append(
                    loss,
                    self.epoch,
                    self.step_ix,
                    "validation",
                )
                logger.info(f"{'•'} {self.epoch}/{self.n_epochs} {'step':>15}")
                self.trace.checkpoint(logger=logger)

                if pbar is not None:
                    pbar.set_description(f"epoch {self.epoch}/{self.n_epochs} validation loss {loss:.2f}")

                if self.early_stopping and (prev_loss is not None):
                    if prev_loss < loss:
                        break

                prev_loss = loss

                if self.scheduler is not None:
                    if self.lr_scheduler == "plateau":
                        self.scheduler.step(loss)
                    else:
                        self.scheduler.step()

            self.model = self.model.train()

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

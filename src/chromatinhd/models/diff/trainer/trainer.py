import tqdm.auto as tqdm
import torch
from chromatinhd.train import Trace
from chromatinhd import get_default_device
import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.propagate = False


class Trainer:
    def __init__(
        self,
        model,
        loaders_train,
        loaders_validation,
        minibatcher_train,
        minibatcher_validation,
        optim,
        hooks_checkpoint=None,
        hooks_checkpoint2=None,
        device=None,
        n_epochs=30,
        checkpoint_every_epoch=1,
        optimize_every_step=1,
        early_stopping=True,
        early_stopping_epochs=1,
        gamma=1.0,
        do_validation=True,
    ):
        self.model = model
        self.loaders_train = loaders_train
        self.loaders_validation = loaders_validation

        self.trace = Trace()

        self.optim = optim

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1, gamma=gamma)

        self.step_ix = 0
        self.epoch = 0
        self.n_epochs = n_epochs

        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.optimize_every_step = optimize_every_step

        self.device = device

        self.minibatcher_train = minibatcher_train
        self.minibatcher_validation = minibatcher_validation

        self.hooks_checkpoint = hooks_checkpoint if hooks_checkpoint is not None else []
        self.hooks_checkpoint2 = hooks_checkpoint2 if hooks_checkpoint2 is not None else []

        self.early_stopping = early_stopping
        self.do_validation = do_validation

    def train(self):
        if self.device is None:
            self.device = get_default_device()

        self.model = self.model.to(self.device)

        self.loaders_train.initialize(self.minibatcher_train)
        self.loaders_validation.initialize(self.minibatcher_validation)

        continue_training = True

        n_steps_total = self.n_epochs * len(self.loaders_train)
        pbar = tqdm.tqdm(total=n_steps_total, leave=False)

        prev_validation_loss = None

        while (self.epoch < self.n_epochs) and (continue_training):
            pbar.set_description(f"epoch {self.epoch}")

            # checkpoint if necessary
            if (self.epoch % self.checkpoint_every_epoch) == 0:
                for hook in self.hooks_checkpoint:
                    hook.start()

                if self.do_validation:
                    with torch.no_grad():
                        losses = []
                        for data_validation in self.loaders_validation:
                            data_validation = data_validation.to(self.device)

                            loss = -self.model.forward_likelihood(data_validation)

                            self.trace.append(loss.sum().item(), self.epoch, self.step_ix, "validation")

                            for hook in self.hooks_checkpoint:
                                hook.run_individual(self.model, data_validation)
                            losses.append(loss.sum().item())
                        if (prev_validation_loss is not None) and (self.early_stopping):
                            if sum(losses) >= prev_validation_loss:
                                continue_training = False
                                logger.info("early stopping")
                        prev_validation_loss = sum(losses)

                logger.info(f"{'•'} {self.epoch}/{self.n_epochs} {'step':>15}")
                self.trace.checkpoint(logger)

                for hook in self.hooks_checkpoint:
                    hook.finish()

                for hook in self.hooks_checkpoint2:
                    hook.run(self.model)

            # train
            for data_train in self.loaders_train:
                # actual training
                data_train = data_train.to(self.device)

                loss = self.model.forward(data_train).sum()
                loss.backward()

                # check if optimization
                if (self.step_ix % self.optimize_every_step) == 0:
                    self.optim.step()
                    self.optim.zero_grad()

                self.step_ix += 1
                pbar.update()

                self.trace.append(loss.item(), self.epoch, self.step_ix, "train")
            self.epoch += 1
            self.scheduler.step()

        pbar.update(n_steps_total)
        pbar.close()

        self.model = self.model.to("cpu")


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

        self.prev_region_loss = None
        n_regions = self.minibatcher_train.regions.max() + 1
        self.improved = np.ones(n_regions, dtype=bool)

        while (self.epoch < self.n_epochs) and (continue_training):
            continue_training = self.checkpoint(pbar)
            if not continue_training:
                break

            self.model = self.model.train()

            # train
            for data_train in self.loaders_train:
                data_train = data_train.to(self.device)

                loss = self.model.forward(data_train).sum()
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

        if pbar is not None:
            pbar.update(self.n_epochs)
            pbar.close()

        self.model = self.model.to("cpu")

    def checkpoint(self, pbar):
        # checkpoint if necessary
        if (self.epoch % self.checkpoint_every_epoch) == 0:
            self.model = self.model.eval()
            with torch.no_grad():
                region_loss = np.zeros_like(self.improved, dtype=float)
                for data_validation in self.loaders_validation:
                    data_validation = data_validation.to(self.device)

                    region_loss_mb = self.model.forward_region_loss(data_validation).cpu().detach().numpy()

                    region_loss[data_validation.minibatch.regions_oi] += region_loss_mb

            self.trace.append(
                region_loss.mean().item(),
                self.epoch,
                self.step_ix,
                "validation",
            )
            logger.info(f"{'•'} {self.epoch}/{self.n_epochs} {'step':>15}")
            self.trace.checkpoint(logger=logger)

            # compare with previous loss per region
            if self.prev_region_loss is not None:
                improvement = region_loss - self.prev_region_loss

                if self.improved is not None:
                    self.improved = self.improved & (improvement < 0)
                else:
                    self.improved = improvement < 0
                logger.info(f"{self.improved.mean():.1%}")

                # stop training once less than 1% of regions are still being optimized
                if self.early_stopping and (self.improved.mean() == 0.00):
                    return False

                if self.early_stopping and (self.epoch >= self.warmup_epochs):
                    self.minibatcher_train.regions = np.where(self.improved)[0]
                    self.loaders_train.start()

            self.prev_region_loss = region_loss.copy()

            if pbar is not None:
                if self.prev_region_loss is None:
                    pbar.set_description(f"epoch {self.epoch}/{self.n_epochs}")
                elif self.improved is None:
                    pbar.set_description(f"epoch {self.epoch}/{self.n_epochs} validation loss {region_loss.mean():.2f}")
                else:
                    pbar.set_description(
                        f"epoch {self.epoch}/{self.n_epochs} validation loss {region_loss.mean():.2f} regions {self.improved.mean():.1%}"
                    )
        return True

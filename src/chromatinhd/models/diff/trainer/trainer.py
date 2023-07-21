import tqdm.auto as tqdm
import torch
import numpy as np
from chromatinhd.train import Trace

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


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
        device="cuda",
        n_epochs=30,
        checkpoint_every_epoch=1,
        optimize_every_step=1,
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

        self.device = device

        self.minibatcher_train = minibatcher_train
        self.minibatcher_validation = minibatcher_validation

        self.hooks_checkpoint = hooks_checkpoint if hooks_checkpoint is not None else []
        self.hooks_checkpoint2 = (
            hooks_checkpoint2 if hooks_checkpoint2 is not None else []
        )

    def train(self):
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        self.model = self.model.to(self.device)

        self.loaders_train.initialize(self.minibatcher_train)
        self.loaders_validation.initialize(self.minibatcher_validation)

        continue_training = True

        n_steps_total = self.n_epochs * len(self.loaders_train)
        pbar = tqdm.tqdm(total=n_steps_total, leave=False)

        while (self.epoch < self.n_epochs) and (continue_training):
            pbar.set_description(f"epoch {self.epoch}")

            # checkpoint if necessary
            if (self.epoch % self.checkpoint_every_epoch) == 0:
                for hook in self.hooks_checkpoint:
                    hook.start()

                with torch.no_grad():
                    for data_validation in self.loaders_validation:
                        data_validation = data_validation.to(self.device)

                        loss = self.model(data_validation).sum()

                        self.trace.append(
                            loss.item(), self.epoch, self.step_ix, "validation"
                        )

                        for hook in self.hooks_checkpoint:
                            hook.run_individual(self.model, data_validation)

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

        pbar.update(n_steps_total)
        pbar.close()

        self.model = self.model.to("cpu")

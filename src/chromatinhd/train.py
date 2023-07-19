import tqdm.auto as tqdm
import torch
import pandas as pd
from chromatinhd.utils.ansi import colorcodes


class Trace:
    def __init__(self, n_significant_digits=3):
        self.train_steps = []
        self.validation_steps = []
        self.n_current_train_steps = 0
        self.n_current_validation_steps = 0
        self.n_last_train_steps = None
        self.n_last_validation_steps = None
        self.current_checkpoint = 0
        self.last_validation_diff = []

    def append(self, loss, epoch, step, phase="train"):
        if phase == "train":
            self.train_steps.append(
                {
                    "loss": loss,
                    "epoch": epoch,
                    "step": step,
                    "checkpoint": self.current_checkpoint,
                }
            )
            self.n_current_train_steps += 1
        else:
            self.validation_steps.append(
                {
                    "loss": loss,
                    "epoch": epoch,
                    "step": step,
                    "checkpoint": self.current_checkpoint,
                }
            )
            self.n_current_validation_steps += 1

    def checkpoint(self, logger=print):
        if (
            (self.n_last_train_steps is not None)
            and (self.n_last_train_steps > 0)
            and (self.n_current_train_steps > 0)
        ):
            last_train_steps = pd.DataFrame(
                self.train_steps[
                    -(self.n_current_train_steps + self.n_last_train_steps) : -(
                        self.n_current_train_steps
                    )
                ]
            )
            current_train_steps = pd.DataFrame(
                self.train_steps[-(self.n_current_train_steps) :]
            )

            current_loss = current_train_steps["loss"].mean()
            diff_loss = (
                current_train_steps["loss"].mean() - last_train_steps["loss"].mean()
            )
            perc_diff_loss = diff_loss / current_loss

            logger.info(
                f"{'train':>10} {current_loss:+.2f} Δ{colorcodes.color_sign(diff_loss, '{:+.3f}')} {perc_diff_loss:+.2%}"
            )
        self.n_last_train_steps = self.n_current_train_steps
        self.n_current_train_steps = 0

        if len(self.validation_steps) > 0:
            current_validation_steps = pd.DataFrame(
                self.validation_steps[-(self.n_current_validation_steps) :]
            )
            current_loss = current_validation_steps["loss"].mean()
            if (self.n_last_validation_steps is not None) and (
                self.n_last_validation_steps > 0
            ):
                if self.n_current_validation_steps == 0:
                    raise ValueError(
                        "No validation steps were run since last checkpoint"
                    )
                assert len(self.validation_steps) >= (
                    self.n_current_validation_steps + self.n_last_validation_steps
                )

                last_validation_steps = pd.DataFrame(
                    self.validation_steps[
                        -(
                            self.n_current_validation_steps
                            + self.n_last_validation_steps
                        ) : -(self.n_current_validation_steps)
                    ]
                )

                diff_loss = (
                    current_validation_steps["loss"].mean()
                    - last_validation_steps["loss"].mean()
                )
                self.last_validation_diff.append(diff_loss)
                perc_diff_loss = diff_loss / current_loss

                logger.info(
                    f"{'validation':>10} {current_loss:+.2f} Δ{colorcodes.color_sign(diff_loss, '{:+.3f}')} {perc_diff_loss:+.2%}"
                )
            else:
                logger.info(f"{'validation':>10} {current_loss:+.2f}")
            self.n_last_validation_steps = self.n_current_validation_steps
            self.n_current_validation_steps = 0

        self.current_checkpoint += 1

    def plot(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        plotdata_validation = (
            pd.DataFrame(self.validation_steps)
            .groupby("checkpoint")
            .mean()
            .reset_index()
        )
        plotdata_train = (
            pd.DataFrame(self.train_steps).groupby("checkpoint").mean().reset_index()
        )
        ax.plot(
            plotdata_validation["checkpoint"],
            plotdata_validation["loss"],
            label="validation",
        )
        ax.plot(plotdata_train["checkpoint"], plotdata_train["loss"], label="train")
        ax.legend()


class Trainer:
    def __init__(
        self,
        model,
        loaders,
        loaders_validation,
        optim,
        hooks_checkpoint=None,
        hooks_checkpoint2=None,
        device="cuda",
        n_epochs=30,
        checkpoint_every_epoch=1,
        optimize_every_step=1,
    ):
        n_steps_total = n_epochs * len(loaders)
        self.pbar = tqdm.tqdm(total=n_steps_total)
        self.model = model
        self.loaders = loaders
        self.loaders_validation = loaders_validation

        self.trace = Trace()

        self.optim = optim

        self.step_ix = 0
        self.epoch = 0
        self.n_epochs = n_epochs

        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.optimize_every_step = optimize_every_step

        self.device = device

        self.hooks_checkpoint = hooks_checkpoint if hooks_checkpoint is not None else []
        self.hooks_checkpoint2 = (
            hooks_checkpoint2 if hooks_checkpoint2 is not None else []
        )

    def train(self):
        import gc

        gc.collect()
        torch.cuda.empty_cache()

        self.model = self.model.to(self.device)

        continue_training = True

        while (self.epoch < self.n_epochs) and (continue_training):
            self.pbar.set_description(f"epoch {self.epoch}")

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

                        self.loaders_validation.submit_next()
                print(f"{'•'} {self.epoch}/{self.n_epochs} {'step':>15}")
                self.trace.checkpoint()

                for hook in self.hooks_checkpoint:
                    hook.finish()

                for hook in self.hooks_checkpoint2:
                    hook.run(self.model)

            # train
            for data_train in self.loaders:
                # actual training
                data_train = data_train.to(self.device)

                loss = self.model.forward(data_train).sum()
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

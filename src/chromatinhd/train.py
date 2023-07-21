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

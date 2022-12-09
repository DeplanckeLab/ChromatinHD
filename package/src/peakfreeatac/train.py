import tqdm.auto as tqdm
import torch
import pandas as pd
import subprocess

class Colorcodes(object):
    """
    Provides ANSI terminal color codes which are gathered via the ``tput``
    utility. That way, they are portable. If there occurs any error with
    ``tput``, all codes are initialized as an empty string.
    The provides fields are listed below.
    Control:
    - bold
    - reset
    Colors:
    - blue
    - green
    - orange
    - red
    :license: MIT
    """
    def __init__(self):
        try:
            self.bold = subprocess.check_output("tput bold".split()).decode()
            self.reset = subprocess.check_output("tput sgr0".split()).decode()

            self.blue = subprocess.check_output("tput setaf 4".split()).decode()
            self.green = subprocess.check_output("tput setaf 2".split()).decode()
            self.orange = subprocess.check_output("tput setaf 3".split()).decode()
            self.red = subprocess.check_output("tput setaf 1".split()).decode()
        except subprocess.CalledProcessError as e:
            self.bold = ""
            self.reset = ""

            self.blue = ""
            self.green = ""
            self.orange = ""
            self.red = ""

    def color_sign(self, x, format):
        return (self.red + format.format(x) + self.reset) if x >= 0 else (self.green + format.format(x) + self.reset)

_c = Colorcodes()

class Trace():
    def __init__(self, n_significant_digits = 3):
        self.train_steps = []
        self.validation_steps = []
        self.n_current_train_steps = 0
        self.n_current_validation_steps = 0
        self.n_last_train_steps = None
        self.n_last_validation_steps = None
        self.current_checkpoint = 0
        self.last_validation_diff = []
        
    def append(self, loss, epoch, step, phase = "train"):
        if phase == "train":
            self.train_steps.append({"loss":loss, "epoch":epoch, "step":step, "checkpoint":self.current_checkpoint})
            self.n_current_train_steps += 1
        else:
            self.validation_steps.append({"loss":loss, "epoch":epoch, "step":step, "checkpoint":self.current_checkpoint})
            self.n_current_validation_steps += 1
        
    def checkpoint(self):
        if (self.n_last_train_steps is not None) and (self.n_last_train_steps > 0):
            if self.n_current_train_steps == 0:
                raise ValueError("No training steps were run since last checkpoint")
            last_train_steps = pd.DataFrame(self.train_steps[-(self.n_current_train_steps + self.n_last_train_steps):-(self.n_current_train_steps)])
            current_train_steps = pd.DataFrame(self.train_steps[-(self.n_current_train_steps):])
            
            current_loss = current_train_steps["loss"].mean()
            diff_loss = current_train_steps["loss"].mean() - last_train_steps["loss"].mean()
            perc_diff_loss = diff_loss / current_loss
            
            print(f"{'train':>10} {current_loss:+.2f} Δ{_c.color_sign(diff_loss, '{:+.3f}')} {perc_diff_loss:+.2%}")
        self.n_last_train_steps = self.n_current_train_steps
        self.n_current_train_steps = 0
            
        
        current_validation_steps = pd.DataFrame(self.validation_steps[-(self.n_current_validation_steps):])
        current_loss = current_validation_steps["loss"].mean()
        if (self.n_last_validation_steps is not None) and (self.n_last_validation_steps > 0):
            if self.n_current_validation_steps == 0:
                raise ValueError("No validation steps were run since last checkpoint")
            assert len(self.validation_steps) >= (self.n_current_validation_steps + self.n_last_validation_steps)

            last_validation_steps = pd.DataFrame(self.validation_steps[-(self.n_current_validation_steps + self.n_last_validation_steps):-(self.n_current_validation_steps)])
            
            diff_loss = current_validation_steps["loss"].mean() - last_validation_steps["loss"].mean()
            self.last_validation_diff.append(diff_loss)
            perc_diff_loss = diff_loss / current_loss
            
            print(f"{'validation':>10} {current_loss:+.2f} Δ{_c.color_sign(diff_loss, '{:+.3f}')} {perc_diff_loss:+.2%}")
        else:
            print(f"{'validation':>10} {current_loss:+.2f}")
        self.n_last_validation_steps = self.n_current_validation_steps
        self.n_current_validation_steps = 0
        
        self.current_checkpoint += 1


class Trainer():
    def __init__(
        self,
        model,
        loaders,
        loaders_validation,
        outcome,
        loss,
        optim,
        device = "cuda",
        n_epochs = 30,
        checkpoint_every_step = 30,
        optimize_every_step = 10
    ):
        n_steps_total = n_epochs * len(loaders)
        self.pbar = tqdm.tqdm(total = n_steps_total)
        self.model = model
        self.loaders = loaders
        self.loaders_validation = loaders_validation
        self.outcome = outcome
        self.trace = Trace()
        
        self.loss = loss
        self.optim = optim
        
        self.step_ix = 0
        self.epoch = 0
        self.n_epochs = n_epochs
        
        self.checkpoint_every_step = checkpoint_every_step
        self.optimize_every_step = optimize_every_step
        
        self.device = device
        
    def train(self):
        self.model = self.model.to(self.device)
        self.outcome = self.outcome.to(self.device)

        continue_training = True
        
        while (self.epoch < self.n_epochs) and (continue_training):
            # train
            for data_train in self.loaders:
                # checkpoint if necessary
                if (self.step_ix % self.checkpoint_every_step) == 0:
                    with torch.no_grad():
                        for data_validation in self.loaders_validation:
                            data_validation = data_validation.to(self.device)

                            transcriptome_subset = self.outcome[data_validation.cells_oi, :][:, data_validation.genes_oi].to(self.device)

                            transcriptome_predicted = self.model(data_validation)

                            loss = self.loss(transcriptome_predicted, transcriptome_subset)
                            self.trace.append(loss.item(), self.epoch, self.step_ix, "validation")

                            self.loaders_validation.submit_next()
                    print(f"{'•'} {self.epoch}/{self.n_epochs} {'step':>15}")
                    self.trace.checkpoint()

                    if (self.epoch > 1) and (self.trace.last_validation_diff[-1] > 0):
                        continue_training = False
                        break
                    
                # actual training
                torch.set_grad_enabled(True)

                data_train = data_train.to(self.device)

                transcriptome_subset = self.outcome[data_train.cells_oi, :][:, data_train.genes_oi].to(self.device)

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
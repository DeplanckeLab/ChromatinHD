from chromatinhd.train import Trace
import tqdm.auto as tqdm
import torch


class Trainer:
    def __init__(
        self,
        model,
        loaders,
        loaders_validation,
        optim,
        hooks_checkpoint=None,
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

        self.hooks_checkpoint = hooks_checkpoint if hooks_checkpoint is not None else []

        self.checkpoint_every_epoch = checkpoint_every_epoch
        self.optimize_every_step = optimize_every_step

        self.device = device

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

                for hook in self.hooks_checkpoint:
                    hook.finish()

                print(f"{'•'} {self.epoch}/{self.n_epochs} {'step':>15}")
                self.trace.checkpoint()

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

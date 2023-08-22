import copy
import threading
import numpy as np
from typing import List


class ThreadWithResult(threading.Thread):
    result = None

    def __init__(
        self,
        group=None,
        target=None,
        name=None,
        loader=None,
        args=(),
        kwargs={},
        *,
        daemon=None,
    ):
        assert target is not None
        self.loader = loader

        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


class LoaderPool:
    loaders_available: list
    restarted: bool = False

    def __init__(
        self,
        loader_cls,
        loader_kwargs=None,
        n_workers=3,
        loader=None,
    ):
        self.running = []

        if loader_kwargs is None:
            loader_kwargs = {}
        self.loader_cls = loader_cls
        self.loader_kwargs = loader_kwargs

        self.n_workers = n_workers

        if loader is not None:
            self.loaders = [loader.copy() for i in range(n_workers)]
        else:
            self.loaders = [loader_cls(**loader_kwargs) for i in range(n_workers)]
        for loader in self.loaders:
            loader.running = False
        self.wait = []

    def initialize(self, tasker, *args, **kwargs):
        self.tasker = tasker

        self.args = args
        self.kwargs = kwargs

        self.restart(*args, **kwargs)

    def submit_next(self):
        try:
            task = next(self.iter)
        except StopIteration:
            return
        self.submit(task, *self.args, **self.kwargs)

    def submit(self, *args, **kwargs):
        if self.loaders_available is None:
            raise ValueError("Pool was not initialized")
        if len(self.loaders_available) == 0:
            raise ValueError("No loaders available")

        loader = self.loaders_available.pop(0)
        if loader.running:
            raise ValueError
        self.loaders_available = self.loaders_available + [loader]
        loader.running = True
        thread = ThreadWithResult(target=loader.load, loader=loader, args=args, kwargs=kwargs)
        self.running.append(thread)
        thread.start()

    def pull(self):
        thread = self.running.pop(0)
        import time

        start = time.time()
        thread.join()
        wait = time.time() - start
        self.wait.append(wait)
        # print(f"Waited {wait} seconds for loading")
        thread.loader.running = False
        self.submit_next()
        return thread.result

    def __len__(self):
        return self.tasker.__len__()

    def __iter__(self):
        if not self.restarted:
            self.restart()
        self.restarted = False
        return self

    def __next__(self):
        if len(self.running) == 0:
            raise StopIteration
        return self.pull()

    def restart(self):
        # join all still running threads
        for thread in self.running:
            thread.join()

        for loader in self.loaders:
            loader.running = False

        self.loaders_available = copy.copy(self.loaders)

        self.running = []

        self.wait = []

        self.iter = iter(self.tasker)

        self.restarted = True

        for i in range(min(len(self.tasker), self.n_workers - len(self.running) - 1)):
            self.submit_next()

    def terminate(self):
        for thread in self.running:
            thread.join()
        self.running = []
        self.loaders = None

    def __del__(self):
        self.terminate()


def benchmark(loaders, minibatcher, n=100):
    """
    Benchmarks a pool of loaders
    """

    loaders.initialize(minibatcher)
    waits = []
    import time
    import tqdm.auto as tqdm

    start = time.time()
    for i, data in zip(range(n), tqdm.tqdm(loaders)):
        waits.append(time.time() - start)
        start = time.time()
    print(sum(waits))

    import matplotlib.pyplot as plt

    plt.plot(waits)

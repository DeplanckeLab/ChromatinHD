import copy
import threading
import numpy as np
from typing import List
import time


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


class LoaderPool:
    loaders_running: list
    loaders_available: list
    counter: bool = False

    def __init__(
        self,
        loader_cls,
        loader_kwargs=None,
        n_workers=3,
        loader=None,
    ):
        self.loaders_running = []

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

        self.start(*args, **kwargs)

    def start(self):
        # join all still running threads
        for thread in self.loaders_running:
            thread.join()

        for loader in self.loaders:
            loader.running = False

        self.loaders_available = copy.copy(self.loaders)
        self.loaders_running = []

        self.wait = []

        self.tasker_iter = iter(self.tasker)

        for i in range(min(len(self.tasker), self.n_workers - 1)):
            self.submit_next()

    def __iter__(self):
        self.counter = 0
        return self

    def __len__(self):
        return len(self.tasker)

    def __next__(self):
        self.counter += 1
        if self.counter > len(self.tasker):
            raise StopIteration
        result = self.pull()
        self.submit_next()
        return result

    def submit_next(self):
        try:
            task = next(self.tasker_iter)
        except StopIteration:
            self.tasker_iter = iter(self.tasker)
            task = next(self.tasker_iter)
        self.submit(task, *self.args, **self.kwargs)

    def submit(self, *args, **kwargs):
        if self.loaders_available is None:
            raise ValueError("Pool was not initialized")
        if len(self.loaders_available) == 0:
            raise ValueError("No loaders available")

        loader = self.loaders_available.pop(0)
        if loader.running:
            raise ValueError
        loader.running = True
        thread = ThreadWithResult(target=loader.load, loader=loader, args=args, kwargs=kwargs)
        self.loaders_running.append(thread)
        thread.start()

    def pull(self):
        thread = self.loaders_running.pop(0)

        start = time.time()
        thread.join()
        wait = time.time() - start
        self.wait.append(wait)
        thread.loader.running = False
        result = thread.result
        self.loaders_available.append(thread.loader)
        return result

    def terminate(self):
        for thread in self.loaders_running:
            thread.join()
        self.loaders_running = []
        self.loaders = None

    def __del__(self):
        self.terminate()

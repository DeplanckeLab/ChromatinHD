import copy
import threading
import numpy as np
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
        daemon=None
    ):
        assert target is not None
        self.loader = loader

        def function():
            self.result = target(*args, **kwargs)

        super().__init__(group=group, target=function, name=name, daemon=daemon)


class LoaderPool:
    tasks: list
    next_task_sets: list
    loaders_available: list
    n_todo: list[int]
    shuffle_on_iter = False

    def __init__(
        self,
        loader_cls,
        loader_kwargs=None,
        n_workers=3,
        shuffle_on_iter=False,
        loader=None,
    ):
        self.running = []

        self.shuffle_on_iter = shuffle_on_iter

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

    def initialize(self, tasks=None, next_task_sets=None, *args, **kwargs):
        if next_task_sets is None:
            next_task_sets = [{"tasks": tasks}]
        self.next_task_sets = copy.copy(next_task_sets)

        self.args = args
        self.kwargs = kwargs

        self.n_todo = []

        self.restart(*args, **kwargs)

    def restart_loaders(self, loader_kwargs=None):
        loader_cls = self.loader_cls
        if loader_kwargs is None:
            loader_kwargs = {}
        loader_kwargs = {**self.loader_kwargs, **loader_kwargs}
        self.loaders = [loader_cls(**loader_kwargs) for i in range(self.n_workers)]

        for loader in self.loaders:
            loader.running = False

        self.loaders_available = copy.copy(self.loaders)

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
        thread = ThreadWithResult(
            target=loader.load, loader=loader, args=args, kwargs=kwargs
        )
        self.running.append(thread)
        thread.start()

    def pull(self):
        thread = self.running.pop(0)
        # import time
        # start = time.time()
        thread.join()
        # end = time.time() - start
        # print(end)
        thread.loader.running = False
        self.n_done += 1
        return thread.result

    def __len__(self):
        return self.tasks.__len__()

    def __iter__(self):
        assert len(self.n_todo) > 0
        self.n_done = 0
        return self

    def __next__(self):
        if len(self.n_todo) == 0:
            raise StopIteration
        if self.n_done == self.n_todo[0]:
            self.n_todo.pop(0)
            raise StopIteration
        if len(self.running) == 0:
            if len(self.todo) > 0:
                raise ValueError(
                    "Iteration stopped too early, make sure to call submit_next in each iteration"
                )
        return self.pull()

    def submit_next(self):
        if len(self.todo) == 0:
            self.start_next()

        task = self.todo.pop(0)
        self.submit(task, *self.args, **self.kwargs)

    def start_next(self):
        if len(self.next_task_sets) > 0:
            # if some new tasks are next, recreate the tasks
            next_task_set = self.next_task_sets.pop(0)
            self.tasks = copy.copy(next_task_set["tasks"])
            if ("loader_kwargs" in next_task_set) and (
                next_task_set["loader_kwargs"] is not None
            ):
                self.restart_loaders(loader_kwargs=next_task_set["loader_kwargs"])

        self.n_todo.append(len(self.tasks))

        if self.shuffle_on_iter:
            self.todo = [
                self.tasks[i]
                for i in self.rg.choice(len(self.tasks), len(self.tasks), replace=False)
            ]
        else:
            self.todo = copy.copy(self.tasks)

    def restart(self):
        # join all still running threads
        for thread in self.running:
            thread.join()

        if self.shuffle_on_iter:
            self.rg = np.random.RandomState(0)

        for loader in self.loaders:
            loader.running = False

        self.loaders_available = copy.copy(self.loaders)

        self.n_done = 0
        self.running = []
        self.todo = []
        self.n_todo = []

        self.start_next()

        for i in range(min(len(self.tasks), self.n_workers - len(self.running) - 1)):
            self.submit_next()

    def terminate(self):
        for thread in self.running:
            thread.join()
        self.running = []
        self.loaders = None

    def __del__(self):
        self.terminate()


class LoaderPool2:
    loaders_available: list
    n_todo: list[int]
    shuffle_on_iter = False

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
        thread = ThreadWithResult(
            target=loader.load, loader=loader, args=args, kwargs=kwargs
        )
        self.running.append(thread)
        thread.start()

    def pull(self):
        thread = self.running.pop(0)
        import time

        start = time.time()
        thread.join()
        self.wait.append(time.time() - start)
        thread.loader.running = False
        self.submit_next()
        return thread.result

    def __len__(self):
        return self.tasker.__len__()

    def __iter__(self):
        self.restart()
        return self

    def __next__(self):
        if len(self.running) == 0:
            # if len(self.wait) > 0:
            #     print("Average wait: ", np.mean(self.wait))
            #     print("Cumulative wait: ", np.sum(self.wait))
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

        for i in range(min(len(self.tasker), self.n_workers - len(self.running) - 1)):
            self.submit_next()

    def terminate(self):
        for thread in self.running:
            thread.join()
        self.running = []
        self.loaders = None

    def __del__(self):
        self.terminate()

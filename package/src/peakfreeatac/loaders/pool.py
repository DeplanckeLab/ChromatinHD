import copy
import threading
import numpy as np
import time

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, loader = None, args=(), kwargs={}, *, daemon=None):
        self.loader = loader
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)

class LoaderPool():
    tasks = None
    loaders_available = None
    shuffle_on_iter = False
    def __init__(self, loader_cls, loader_kwargs = None, n_workers = 3, shuffle_on_iter = False):
        self.running = []

        self.shuffle_on_iter = shuffle_on_iter

        if loader_kwargs is None:
            loader_kwargs = {}

        self.n_workers = n_workers
        self.loaders = [loader_cls(**loader_kwargs) for i in range(n_workers)]
        for loader in self.loaders:
            loader.running = False

    def initialize(self, tasks, *args, **kwargs):
        assert len(tasks) > 0
        self.tasks = tasks
        self.restart(*args, **kwargs)

    def submit(self, *args, **kwargs):
        if self.loaders_available is None:
            raise ValueError("Pool was not initialized")
        if len(self.loaders_available) == 0:
            raise ValueError("No loaders available")
        
        loader = self.loaders_available.pop(0)
        loader.running = True
        thread = ThreadWithResult(target = loader.load, loader = loader, args = args, kwargs = kwargs)
        self.running.append(thread)
        thread.start()

    def pull(self):
        thread = self.running.pop(0)
        thread.join()
        self.loaders_available.append(thread.loader)
        self.n_done += 1
        return thread.result

    def __len__(self):
        return self.tasks.__len__()

    def __iter__(self):
        self.n_done = 0
        return self

    def __next__(self):
        if self.n_done == len(self.tasks):
            raise StopIteration
        if len(self.running) == 0:
            if len(self.todo) > 0:
                raise ValueError("Iteration stopped too early, make sure to call submit_next in each iteration")
        return self.pull()
    
    def submit_next(self):
        if len(self.todo) == 0:
            if self.shuffle_on_iter:
                self.todo = [self.tasks[i] for i in self.rg.choice(len(self.tasks), len(self.tasks), replace = False)]
            else:
                self.todo = copy.copy(self.tasks)

        task = self.todo.pop(0)
        self.todo.append(task)
        self.submit(task, *self.args, **self.kwargs)

    def restart(self, *args, **kwargs):
        # join all still running threads
        for thread in self.running:
            thread.join()

        if self.shuffle_on_iter:
            self.rg = np.random.RandomState(0)

        self.loaders_available = copy.copy(self.loaders)

        self.n_done = 0
        self.running = []
        self.todo = []

        self.args = args
        self.kwargs = kwargs

        for i in range(self.n_workers):
            self.submit_next()
            
    def terminate(self):
        for thread in self.running:
            thread.join()
        self.running = []
        self.loaders = None
        
    def __del__(self):
        self.terminate()
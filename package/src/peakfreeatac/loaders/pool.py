import copy
import threading
import copy

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)

class LoaderPool():
    tasks = None
    def __init__(self, loader_cls, loader_kwargs = None, n_workers = 3):
        self.running = []

        if loader_kwargs is None:
            loader_kwargs = {}

        self.n_workers = n_workers
        self.loaders = [loader_cls(**loader_kwargs) for i in range(n_workers)]

    def initialize(self, tasks, *args, **kwargs):
        assert len(tasks) > 0
        self.tasks = tasks
        self.restart(*args, **kwargs)

    def submit(self, *args, **kwargs):
        loader = self.loaders[self.current_submit]
        thread = ThreadWithResult(target = loader.load, args = args, kwargs = kwargs)
        self.running.append(thread)
        thread.start()
        self.current_submit = (self.current_submit + 1) % (self.n_workers)

    def pull(self):
        thread = self.running.pop(0)
        thread.join()
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
            self.todo = copy.copy(self.tasks)

        task = self.todo.pop(0)
        self.todo.append(task)
        self.submit(task, *self.args, **self.kwargs)

    def restart(self, *args, **kwargs):
        self.todo = copy.copy(self.tasks)
        self.current_submit = 0
        self.n_done = 0
        self.running = []

        self.args = args
        self.kwargs = kwargs

        for i in range(self.n_workers):
            self.submit_next()
            
    def terminate(self):
        self.loaders = None
        
    def __del__(self):
        self.terminate()
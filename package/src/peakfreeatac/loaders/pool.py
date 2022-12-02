import copy
import threading
import copy

class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = target(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)

class LoaderPool():
    def __init__(self, loader_cls, loader_args = (), n_workers = 3):
        self.q = []

        self.n_workers = n_workers
        self.loaders = [loader_cls(*loader_args) for i in range(n_workers)]
        self.current_submit = 0

    def initialize(self, data):
        self.data = data
        self.reset()

    def submit(self, kwargs):
        loader = self.loaders[self.current_submit]
        thread = ThreadWithResult(target = loader.load, kwargs = kwargs)
        self.q.append(thread)
        thread.start()
        self.current_submit = (self.current_submit + 1) % (self.n_workers)

    def pull(self):
        thread = self.q.pop(0)
        thread.join()
        return thread.result

    def __len__(self):
        return self.q.__len__() + self.current_order.__len__()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.q) == 0:
            if len(self.current_order) > 0:
                raise ValueError("Iteration stopped too early, make sure to call submit_next in each iteration")
            raise StopIteration
        return self.pull()
    
    def submit_next(self):
        if len(self.current_order) > 0:
            self.submit(self.current_order.pop(0))

    def reset(self):
        self.current_order = copy.copy(self.data)

        for i in range(self.n_workers):
            self.submit(self.current_order.pop(0))
            
    def terminate(self):
        self.loaders = None
        
    def __del__(self):
        self.terminate()
import collections
import time


class catchtime(object):
    def __init__(self, dict, name):
        self.name = name
        self.dict = dict

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.t = time.time() - self.t
        self.dict[self.name] += self.t


class timer(object):
    def __init__(self):
        self.times = collections.defaultdict(float)

    def catch(self, name):
        return catchtime(self.times, name)

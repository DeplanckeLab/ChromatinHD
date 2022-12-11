import pathlib
import torch
import pickle
import numpy as np
import gzip

class Flow():
    path:pathlib.Path
    default_name = None

    def __init__(self, path = None, folder = None, name = None):
        if path is None:
            assert folder is not None
            if name is None:
                assert self.default_name is not None
                name = self.default_name

            path = folder / name

        self.path = path
        if not path.exists():
            path.mkdir(parents = True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path})"


class Stored():
    _x = None
    def __init__(self, name):
        self.name = name
        self._x = None

    def get_path(self, folder):
        return (folder / (self.name + ".pkl"))

    def __get__(self, obj, type=None):
        if obj is not None:
            if self._x is None:
                self._x = pickle.load(self.get_path(obj.path).open("rb"))
            return self._x

    def __set__(self, obj, value):
        pickle.dump(value, self.get_path(obj.path).open("wb"))
        self._mapping = value

class StoredTorchInt64(Stored):
    def __get__(self, obj, type=None):
        if obj is not None:
            if self._x is None:
                self._x = pickle.load(self.get_path(obj.path).open("rb"))
                if not self._x.dtype is torch.int64:
                    self._x = self._x.to(torch.int64)
                if not self._x.is_contiguous():
                    self._x = self._x.contiguous()
            return self._x

    def __set__(self, obj, value):
        value = value.to(torch.int64).contiguous()
        pickle.dump(value, self.get_path(obj.path).open("wb"))
        self._mapping = value

class StoredNumpyInt64(Stored):
    def __get__(self, obj, type=None):
        if obj is not None:
            if self._x is None:
                self._x = pickle.load(self.get_path(obj.path).open("rb"))
                if not self._x.dtype is np.int64:
                    self._x = self._x.astype(np.int64)
                if not self._x.flags['C_CONTIGUOUS']:
                    self._x = np.ascontiguousarray(self._x)
            return self._x

    def __set__(self, obj, value):
        value = np.ascontiguousarray(value.astype(np.int64))
        pickle.dump(value, self.get_path(obj.path).open("wb"))
        self._mapping = value

class CompressedNumpy(Stored):
    dtype = np.float64

    def __get__(self, obj, type=None):
        if obj is not None:
            if self._x is None:
                self._x = pickle.load(gzip.GzipFile(self.get_path(obj.path), "rb", compresslevel=3))
                if not self._x.dtype is self.dtype:
                    self._x = self._x.astype(self.dtype)
                if not self._x.flags['C_CONTIGUOUS']:
                    self._x = np.ascontiguousarray(self._x)
            return self._x

    def __set__(self, obj, value):
        value = np.ascontiguousarray(value.astype(self.dtype))
        pickle.dump(value, gzip.GzipFile(self.get_path(obj.path), "wb", compresslevel=3))
        self._mapping = value

class CompressedNumpyFloat64(CompressedNumpy):
    dtype = np.float64

class CompressedNumpyInt64(CompressedNumpy):
    dtype = np.int64

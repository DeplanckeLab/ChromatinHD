import pathlib
import torch
import pickle
import numpy as np
import gzip
import copy


class Flow:
    path: pathlib.Path
    default_name = None

    def __init__(self, path=None, folder=None, name=None):
        if path is None:
            assert folder is not None
            if name is None:
                assert self.default_name is not None
                name = self.default_name

            path = folder / name

        self.path = path
        if not path.exists():
            path.mkdir(parents=True)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path})"


class Stored:
    def __init__(self, name):
        self.name = name

    def get_path(self, folder):
        return folder / (self.name + ".pkl")

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + self.name
            if not hasattr(obj, name):
                setattr(obj, name, pickle.load(self.get_path(obj.path).open("rb")))
            return getattr(obj, name)

    def __set__(self, obj, value):
        name = "_" + self.name
        pickle.dump(value, self.get_path(obj.path).open("wb"))
        setattr(obj, name, value)


class StoredTorchInt64(Stored):
    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + self.name
            if not hasattr(obj, name):
                x = pickle.load(self.get_path(obj.path).open("rb"))
                if not x.dtype is torch.int64:
                    x = x.to(torch.int64)
                if not x.is_contiguous():
                    x = x.contiguous()
                setattr(obj, name, x)
            return getattr(obj, name)

    def __set__(self, obj, value):
        value = value.to(torch.int64).contiguous()
        name = "_" + self.name
        pickle.dump(value, self.get_path(obj.path).open("wb"))
        setattr(obj, name, value)


class StoredTorchInt32(Stored):
    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + self.name
            if not hasattr(obj, name):
                x = pickle.load(self.get_path(obj.path).open("rb"))
                if not x.dtype is torch.int32:
                    x = x.to(torch.int32)
                if not x.is_contiguous():
                    x = x.contiguous()
                setattr(obj, name, x)
            return getattr(obj, name)

    def __set__(self, obj, value):
        value = value.to(torch.int32).contiguous()
        name = "_" + self.name
        pickle.dump(value, self.get_path(obj.path).open("wb"))
        setattr(obj, name, value)


class StoredNumpyInt64(Stored):
    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + self.name
            if not hasattr(obj, name):
                x = pickle.load(self.get_path(obj.path).open("rb"))
                if not x.dtype is np.int64:
                    x = x.astype(np.int64)
                if not x.flags["C_CONTIGUOUS"]:
                    x = np.ascontiguousarray(x)
                setattr(obj, name, x)
            return getattr(obj, name)

    def __set__(self, obj, value):
        value = np.ascontiguousarray(value.astype(np.int64))
        name = "_" + self.name
        pickle.dump(value, self.get_path(obj.path).open("wb"))
        setattr(obj, name, value)


class CompressedNumpy(Stored):
    dtype = np.float64

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + self.name
            if not hasattr(obj, name):
                x = pickle.load(
                    gzip.GzipFile(self.get_path(obj.path), "rb", compresslevel=3)
                )
                if not x.dtype is self.dtype:
                    x = x.astype(self.dtype)
                if not x.flags["C_CONTIGUOUS"]:
                    x = np.ascontiguousarray(x)
                setattr(obj, name, x)
            return getattr(obj, name)

    def __set__(self, obj, value):
        value = np.ascontiguousarray(value.astype(self.dtype))
        name = "_" + self.name
        pickle.dump(
            value, gzip.GzipFile(self.get_path(obj.path), "wb", compresslevel=3)
        )
        setattr(obj, name, value)


class CompressedNumpyFloat64(CompressedNumpy):
    dtype = np.float64


class CompressedNumpyInt64(CompressedNumpy):
    dtype = np.int64


class TSV(Stored):
    def get_path(self, folder):
        return folder / (self.name + ".tsv")

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + self.name
            if not hasattr(obj, name):
                import pandas as pd

                x = pd.read_table(self.get_path(obj.path), index_col=0)
                setattr(obj, name, x)
            return getattr(obj, name)

    def __set__(self, obj, value):
        name = "_" + self.name
        value.to_csv(self.get_path(obj.path), sep="\t")
        setattr(obj, name, value)

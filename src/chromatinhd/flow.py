import pathlib
import torch
import pickle
import numpy as np
import gzip
import copy
import json
import importlib
import shutil


class Flow:
    path: pathlib.Path
    default_name = None

    def __init__(self, path=None, folder=None, name=None, reset=False):
        if path is None:
            if folder is None:
                raise ValueError("Either path or folder must be specified")
            if name is None:
                if self.default_name is None:
                    raise ValueError(
                        "Cannot create Flow without name, and no default_name specified"
                    )
                name = self.default_name

            path = folder / name

        self.path = path
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        if reset:
            self.reset()

        if not self._get_info_path().exists():
            self._store_info()

    @classmethod
    def create(cls, path, **kwargs):
        path = pathlib.Path(path)
        self = cls(path=path)

        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def _store_info(self):
        info = {"module": self.__class__.__module__, "class": self.__class__.__name__}

        json.dump(info, self._get_info_path().open("w"))

    def _get_info_path(self):
        return self.path / ".flow"

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.path}")'

    @classmethod
    def from_path(cls, path):
        info = json.load(open(path / ".flow"))

        # load class
        module = importlib.import_module(info["module"])
        cls = getattr(module, info["class"])

        return cls(path=path)

    def reset(self):
        shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)


class Linked:
    def __init__(self, name):
        self.name = name

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + self.name
            if not hasattr(obj, name):
                path = obj.path / self.name

                if not path.exists():
                    raise FileNotFoundError(f"File {path} does not exist")
                if not path.is_symlink():
                    raise FileNotFoundError(f"File {path} is not a symlink")

                value = Flow.from_path(path.resolve())
                setattr(obj, name, value)

            return getattr(obj, name)

    def __set__(self, obj, value):
        # symlink to value.path
        path = obj.path / self.name
        name = "_" + self.name
        if path.exists():
            if not path.is_symlink():
                raise FileExistsError(f"File {path} already exists")
            else:
                path.unlink()
        path.symlink_to(value.path.resolve())
        setattr(obj, name, value)


class Stored:
    def __init__(self, name, default=None):
        self.name = name
        self.default = default

    def get_path(self, folder):
        return folder / (self.name + ".pkl")

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + self.name
            if not hasattr(obj, name):
                path = self.get_path(obj.path)
                if not path.exists():
                    if self.default is None:
                        raise FileNotFoundError(f"File {path} does not exist")
                    else:
                        value = self.default()
                        pickle.dump(value, path.open("wb"))
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
    def __init__(self, name, columns=None):
        super().__init__(name)
        self.columns = columns

    def get_path(self, folder):
        return folder / (self.name + ".tsv")

    def __get__(self, obj=None, type=None):
        if obj is not None:
            name = "_" + self.name
            if not hasattr(obj, name):
                import pandas as pd

                x = pd.read_table(self.get_path(obj.path), index_col=0)
                setattr(obj, name, x)
            return getattr(obj, name)

    def __set__(self, obj, value, folder=None):
        name = "_" + self.name
        if folder is None:
            folder = obj.path
        value.to_csv(self.get_path(folder), sep="\t")
        setattr(obj, name, value)


class StoredDict:
    def __init__(self, name, cls):
        self.name = name
        self.cls = cls

    def get_path(self, folder):
        return folder / self.name

    def __get__(self, obj, type=None):
        return StoredDictInstance(self.name, self.get_path(obj.path), self.cls, obj)


class StoredDictInstance:
    def __init__(self, name, path, cls, obj):
        self.dict = {}
        self.cls = cls
        self.obj = obj
        self.name = name
        self.path = path
        if not self.path.exists():
            self.path.mkdir(parents=True)
        for file in self.path.iterdir():
            if file.is_dir():
                raise ValueError(f"Folder {file} in {self.obj.path} is not allowed")
            # key is file name without extension
            key = file.name.split(".")[0]
            self.dict[key] = self.cls(key)

    def __getitem__(self, key):
        return self.dict[key].__get__(self)

    def __setitem__(self, key, value):
        if key not in self.dict:
            self.dict[key] = self.cls(key)
        self.dict[key].__set__(self, value)

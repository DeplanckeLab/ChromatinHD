import pathlib
import torch
import pickle
import numpy as np
import gzip
import json
import importlib
import shutil
from typing import Union

PathLike = Union[str, pathlib.Path]


class Obj:
    name = None

    def __init__(self, name=None):
        self.name = name


def is_obj(x):
    return isinstance(x, Obj)


class Flowable(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

        # compile objects
        for attr_id, attr in cls.__dict__.items():
            if is_obj(attr):
                assert isinstance(attr_id, str)
                attr.name = attr_id


class Flow(metaclass=Flowable):
    """
    A folder on disk that can contain other folders or objects
    """

    path: pathlib.Path
    default_name = None

    def __init__(self, path=None, folder=None, name=None, reset=False):
        if isinstance(path, str):
            path = pathlib.Path(path)
        if path is None:
            if folder is None:
                # make temporary
                try:
                    import pathlibfs
                except ImportError:
                    raise ImportError("To create a temporary flow, install pathlibfs")
                from uuid import uuid4

                name = str(uuid4())
                folder = pathlibfs.Path("memory://")
            elif name is None:
                if self.default_name is None:
                    raise ValueError("Cannot create Flow without name, and no default_name specified")
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
        if isinstance(path, str):
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
        """
        Load a previously created flow from disk
        """
        info = json.load(open(path / ".flow"))

        # load class
        module = importlib.import_module(info["module"])
        cls = getattr(module, info["class"])

        return cls(path=path)

    def reset(self):
        """
        Remove all files in this flow
        """
        shutil.rmtree(self.path)
        self.path.mkdir(parents=True, exist_ok=True)

    def __getstate__(self):
        raise TypeError("This class cannot be pickled.")

    def get(self, k):
        return self.__class__.__dict__[k]


class Linked(Obj):
    """
    A link to another flow on disk
    """

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
        if not str(value.path).startswith("memory"):
            print(value.path.name)
            path.symlink_to(value.path.resolve())
        setattr(obj, name, value)


class Stored(Obj):
    """
    A python object that is stored on disk using pickle
    """

    def __init__(self, default=None, name=None):
        self.default = default
        self.name = name

    def get_path(self, folder):
        return folder / (self.name + ".pkl")

    def __get__(self, obj, type=None):
        if obj is not None:
            if self.name is None:
                print(obj)
                raise ValueError(obj)
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

    def exists(self, obj):
        return self.get_path(obj.path).exists()


class StoredDataFrame(Stored):
    """
    A pandas dataframe stored on disk
    """

    def __init__(self, index_name=None, name=None):
        super().__init__(name=name)
        self.index_name = index_name

    def __set__(self, obj, value):
        if self.index_name is not None:
            value.index.name = self.index_name
        super().__set__(obj, value)


class StoredTensor(Stored):
    def __init__(self, dtype=None, name=None):
        super().__init__(name=name)
        self.dtype = dtype

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + self.name
            if not hasattr(obj, name):
                x = pickle.load(self.get_path(obj.path).open("rb"))
                if not torch.is_tensor(x):
                    raise ValueError(f"File {self.get_path(obj.path)} is not a tensor")
                elif x.dtype is not self.dtype:
                    x = x.to(self.dtype)
                if not x.is_contiguous():
                    x = x.contiguous()
                setattr(obj, name, x)
            return getattr(obj, name)

    def __set__(self, obj, value):
        if not torch.is_tensor(value):
            raise ValueError("Value is not a tensor")
        elif self.dtype is not None:
            value = value.to(self.dtype).contiguous()
        name = "_" + self.name
        pickle.dump(value, self.get_path(obj.path).open("wb"))
        setattr(obj, name, value)


class StoredNumpyInt64(Stored):
    """
    A numpy int64 tensor stored on disk
    """

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + self.name
            if not hasattr(obj, name):
                x = pickle.load(self.get_path(obj.path).open("rb"))
                if x.dtype is not np.int64:
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
    """
    A compressed numpy array stored on disk
    """

    dtype = np.float64

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + self.name
            if not hasattr(obj, name):
                x = pickle.load(gzip.GzipFile(self.get_path(obj.path), "rb", compresslevel=3))
                if x.dtype is not self.dtype:
                    x = x.astype(self.dtype)
                if not x.flags["C_CONTIGUOUS"]:
                    x = np.ascontiguousarray(x)
                setattr(obj, name, x)
            return getattr(obj, name)

    def __set__(self, obj, value):
        value = np.ascontiguousarray(value.astype(self.dtype))
        name = "_" + self.name
        pickle.dump(value, gzip.GzipFile(self.get_path(obj.path), "wb", compresslevel=3))
        setattr(obj, name, value)


class CompressedNumpyFloat64(CompressedNumpy):
    dtype = np.float64


class CompressedNumpyInt64(CompressedNumpy):
    dtype = np.int64


class TSV(Stored):
    """
    A pandas object stored on disk in tsv format
    """

    def __init__(self, columns=None, index_name=None, name=None):
        super().__init__(name=name)
        self.columns = columns
        self.index_name = index_name

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
        if self.index_name is not None:
            value.index.name = self.index_name
        value.to_csv(self.get_path(folder).open("w"), sep="\t")
        setattr(obj, name, value)


class StoredDict(Obj):
    def __init__(self, cls, name=None):
        super().__init__(name=name)
        self.cls = cls

    def get_path(self, folder):
        return folder / self.name

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + self.name
            if not hasattr(obj, name):
                x = StoredDictInstance(self.name, self.get_path(obj.path), self.cls, obj)
                setattr(obj, name, x)
            return getattr(obj, name)


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
            self.dict[key] = self.cls(name=key)

    def __getitem__(self, key):
        return self.dict[key].__get__(self)

    def __setitem__(self, key, value):
        if key not in self.dict:
            self.dict[key] = self.cls(name=key)
        self.dict[key].__set__(self, value)

    def items(self):
        for k in self.dict:
            yield k, self[k]

    def keys(self):
        return self.dict.keys()

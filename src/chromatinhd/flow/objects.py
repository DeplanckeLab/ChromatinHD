import pickle
import torch
import numpy as np
import gzip
import json
import importlib
import shutil
from typing import Union
import pandas as pd
import os


def format_size(size: int) -> str:
    for unit in ("Bb", "Kb", "Mb", "Gb", "Tb"):
        if size < 1024:
            break
        size /= 1024
    return f"{size:.1f}{unit}"


def get_size(start_path="."):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size


class Obj:
    name = None

    def __init__(self, name=None):
        self.name = name

    def _repr_html_(self, obj=None):
        if obj is not None:
            return f"<b>{self.name}</b>"
        return f"<b>{self.name}</b>"


class Instance:
    def __init__(self, name=None, path=None, obj=None):
        obj._obj_instances[name] = self

    def _repr_html_(self):
        return f"<b>{self.name}</b>"


def isinstance2(__obj: object, __class):
    return all([repr(y) in repr(__obj.__class__.__mro__) for y in __class.__mro__])


def is_obj(x):
    return isinstance2(x, Obj)


def is_instance(x):
    return isinstance2(x, Instance)


class Linked(Obj):
    """
    A link to another flow on disk
    """

    def get_path(self, obj):
        return obj.path / self.name

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + str(self.name)
            if not hasattr(obj, name):
                path = self.get_path(obj)

                if not path.exists():
                    raise FileNotFoundError(f"File {path} does not exist")
                if not path.is_symlink():
                    raise FileNotFoundError(f"File {path} is not a symlink")

                from .flow import Flow

                value = Flow.from_path(path.resolve())
                setattr(obj, name, value)

            return getattr(obj, name)

    def __set__(self, obj, value):
        # symlink to value.path
        path = self.get_path(obj)
        name = "_" + str(self.name)
        if path.exists():
            if not path.is_symlink():
                raise FileExistsError(f"File {path} already exists")
            else:
                path.unlink()
        if (not str(obj.path).startswith("memory")) and (not str(value.path).startswith("memory")):
            path.symlink_to(value.path.resolve())
        setattr(obj, name, value)

    def exists(self, obj):
        return self.get_path(obj).exists() or ("_" + str(self.name) in obj.__dict__)

    def _repr_html_(self, obj=None):
        if obj is not None:
            return f"<span class='iconify' data-icon='mdi-link'></span> <b>{self.name}</b>"
        return f"<span class='iconify' data-icon='mdi-link'></span> <b>{self.name}</b>"


class Stored(Obj):
    """
    A python object that is stored on disk using pickle
    """

    def __init__(self, default=None, name=None):
        self.default = default
        self.name = name

    def get_path(self, folder):
        return folder / (str(self.name) + ".pkl")

    def __get__(self, obj, type=None):
        if obj is not None:
            if self.name is None:
                raise ValueError(obj)
            name = "_" + str(self.name)
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
        name = "_" + str(self.name)
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

    def _repr_html_(self, obj=None):
        return f"<span class='iconify' data-icon='mdi-table'></span> <b>{self.name}</b>"


class StoredTensor(Stored):
    def __init__(self, dtype=None, name=None):
        super().__init__(name=name)
        self.dtype = dtype

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + str(self.name)
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
        name = "_" + str(self.name)
        pickle.dump(value, self.get_path(obj.path).open("wb"))
        setattr(obj, name, value)


class StoredNumpyInt64(Stored):
    """
    A numpy int64 tensor stored on disk
    """

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + str(self.name)
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
        name = "_" + str(self.name)
        pickle.dump(value, self.get_path(obj.path).open("wb"))
        setattr(obj, name, value)


class CompressedNumpy(Stored):
    """
    A compressed numpy array stored on disk
    """

    dtype = None

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + str(self.name)
            if not hasattr(obj, name):
                x = pickle.load(gzip.GzipFile(fileobj=self.get_path(obj.path).open("rb"), mode="r", compresslevel=3))
                if self.dtype is not None:
                    if x.dtype is not self.dtype:
                        x = x.astype(self.dtype)
                if not x.flags["C_CONTIGUOUS"]:
                    x = np.ascontiguousarray(x)
                setattr(obj, name, x)
            return getattr(obj, name)

    def __set__(self, obj, value):
        if self.dtype is not None:
            value = np.ascontiguousarray(value.astype(self.dtype))
        name = "_" + str(self.name)
        pickle.dump(value, gzip.GzipFile(fileobj=self.get_path(obj.path).open("wb"), mode="w", compresslevel=3))
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
            name = "_" + str(self.name)
            if not hasattr(obj, name):
                x = pd.read_table(self.get_path(obj.path), index_col=0)
                setattr(obj, name, x)
            return getattr(obj, name)

    def __set__(self, obj, value, folder=None):
        name = "_" + str(self.name)
        if folder is None:
            folder = obj.path
        if self.index_name is not None:
            value.index.name = self.index_name
        value.to_csv(self.get_path(folder).open("w"), sep="\t")
        setattr(obj, name, value)

    def _repr_html_(self, obj=None):
        return f"<span class='iconify' data-icon='mdi-table'></span> <b>{self.name}</b>"


class StoredDict(Obj):
    def __init__(self, cls, name=None, kwargs=None):
        super().__init__(name=name)
        if kwargs is None:
            kwargs = {}
        self.kwargs = kwargs
        self.cls = cls

    def get_path(self, folder):
        return folder / self.name

    def __get__(self, obj, type=None):
        if obj is not None:
            name = "_" + str(self.name)
            if not hasattr(obj, name):
                x = StoredDictInstance(self.name, self.get_path(obj.path), self.cls, obj, self.kwargs)
                setattr(obj, name, x)
            return getattr(obj, name)

    def _repr_html_(self, obj=None):
        instance = self.__get__(obj)
        return instance._repr_html_()

    def exists(self, obj):
        return True


class StoredDictInstance(Instance):
    def __init__(self, name, path, cls, obj, kwargs):
        super().__init__(name=name, path=path, obj=obj)
        self.dict = {}
        self.cls = cls
        self.obj = obj
        self.name = name
        self.path = path
        self.kwargs = kwargs
        if not self.path.exists():
            self.path.mkdir(parents=True)
        for file in self.path.iterdir():
            # key is file name without extension
            key = file.name.split(".")[0]
            self.dict[key] = self.cls(name=key, **self.kwargs)

    def __getitem__(self, key):
        return self.dict[key].__get__(self)

    def __setitem__(self, key, value):
        if key not in self.dict:
            self.dict[key] = self.cls(name=key, **self.kwargs)
        self.dict[key].__set__(self, value)

    def __contains__(self, key):
        return key in self.dict

    def __len__(self):
        return len(self.dict)

    def items(self):
        for k in self.dict:
            yield k, self[k]

    def keys(self):
        return self.dict.keys()

    def exists(self):
        return True

    def _repr_html_(self):
        # return f"<span class='iconify' data-icon='mdi-layers'></span> <b>{self.name}</b> ({', '.join([getattr(self.obj, k)._repr_html_() for k in self.keys()])})"
        items = []
        for i, k in zip(range(3), self.keys()):
            items.append(self.dict[k]._repr_html_(self))
        if len(self.keys()) > 3:
            items.append("...")
        return f"<span class='iconify' data-icon='mdi-layers'></span> <b>{self.name}</b> ({', '.join(items)})"


class DataArray(Obj):
    def __init__(self, name=None):
        super().__init__(name=name)

    def get_path(self, folder):
        return folder / (self.name + ".zarr")

    def __get__(self, obj, type=None):
        if obj is not None:
            if self.name is None:
                raise ValueError(obj)
            name = "_" + str(self.name)
            if not hasattr(obj, name):
                path = self.get_path(obj.path)
                if not path.exists():
                    raise FileNotFoundError(f"File {path} does not exist")
                import xarray as xr

                setattr(obj, name, xr.open_zarr(self.get_path(obj.path))[self.name])
            return getattr(obj, name)

    def __set__(self, obj, value):
        import xarray as xr

        name = "_" + str(self.name)
        value.to_zarr(self.get_path(obj.path), mode="w")
        setattr(obj, name, value)

    def exists(self, obj):
        return self.get_path(obj.path).exists()

    def _repr_html_(self, obj):
        instance = self.__get__(obj)
        shape = "[" + ",".join(str(x) for x in instance.shape) + "]"
        size = format_size(get_size(self.get_path(obj.path)))
        return f"<span class='iconify' data-icon='mdi-axis-arrow-info'></span> <b>{self.name}</b> {shape}, {size}"


class Dataset(Obj):
    def __init__(self, name=None):
        super().__init__(name=name)

    def get_path(self, folder):
        return folder / (self.name + ".zarr")

    def __get__(self, obj, type=None):
        if obj is not None:
            if self.name is None:
                raise ValueError(obj)
            name = "_" + str(self.name)
            if not hasattr(obj, name):
                path = self.get_path(obj.path)
                if not path.exists():
                    raise FileNotFoundError(f"File {path} does not exist")
                import xarray as xr

                setattr(obj, name, xr.open_zarr(self.get_path(obj.path)))
            return getattr(obj, name)

    def __set__(self, obj, value):
        import xarray as xr

        name = "_" + str(self.name)
        value.to_zarr(self.get_path(obj.path), mode="w")
        setattr(obj, name, value)

    def exists(self, obj):
        return self.get_path(obj.path).exists()

    def _repr_html_(self, obj):
        self.__get__(obj)
        size = format_size(get_size(self.get_path(obj.path)))
        return f"<span class='iconify' data-icon='mdi-axis-arrow-info'></span> <b>{self.name}</b> {size}"

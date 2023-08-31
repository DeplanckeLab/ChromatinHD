import copy
import os

import numpy as np
import tensorstore as ts
import shutil

from .objects import Obj

from .objects import format_size, get_size

default_spec_create = {
    "driver": "zarr",
    "kvstore": {
        "driver": "file",
    },
    "metadata": {},
}

default_spec_write = {
    "driver": "zarr",
    "kvstore": {
        "driver": "file",
    },
}


default_spec_read = {
    "driver": "zarr",
    "kvstore": {
        "driver": "file",
    },
    "open": True,
}


def deep_update(mapping, *updating_mappings):
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if k in updated_mapping and isinstance(updated_mapping[k], dict) and isinstance(v, dict):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


class Tensorstore(Obj):
    """
    A python object that is stored on disk using pickle
    """

    def __init__(
        self,
        spec_create: dict = default_spec_create,
        spec_write: dict = default_spec_write,
        spec_read: dict = default_spec_read,
        dtype=None,
        name=None,
        compression="blosc",
        chunks=None,
        shape=(0,),
    ):
        self.name = name
        self.spec_create = copy.deepcopy(spec_create)
        self.spec_write = copy.deepcopy(spec_write)
        self.spec_read = copy.deepcopy(spec_read)

        if dtype is not None:
            self.spec_create["metadata"]["dtype"] = dtype

        if compression is not None:
            if compression == "blosc":
                self.spec_create["metadata"]["compressor"] = {
                    "id": "blosc",
                    "clevel": 3,
                    "cname": "zstd",
                    "shuffle": 2,
                }
        else:
            self.spec_create["metadata"]["compressor"] = None

        if chunks is not None:
            self.spec_create["metadata"]["chunks"] = chunks

        if shape is not None:
            self.spec_create["metadata"]["shape"] = shape

        if "dtype" not in self.spec_create["metadata"]:
            raise ValueError("dtype must be specified")

    def get_path(self, folder):
        return folder / (self.name + ".zarr")

    def __get__(self, obj, type=None):
        if obj is not None:
            if self.name is None:
                raise ValueError(obj)
            name = "_" + self.name
            if not hasattr(obj, name):
                instance = TensorstoreInstance(
                    self.name, self.get_path(obj.path), self.spec_create, self.spec_write, self.spec_read
                )
                setattr(obj, name, instance)
            return getattr(obj, name)

    def __set__(self, obj, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("Must be an numpy ndarray, not " + str(type(value)))
        path = self.get_path(obj.path)
        if path.exists():
            shutil.rmtree(path)
        self.__get__(obj)[:] = value

    def exists(self, obj):
        return self.__get__(obj).exists()

    def _repr_html_(self, obj):
        instance = self.__get__(obj)
        return instance._repr_html_()


class TensorstoreInstance:
    def __init__(self, name, path, spec_create, spec_write, spec_read):
        self.name = name
        self.path = path
        self.spec_create = deep_update(spec_create, {"kvstore": {"path": str(path)}})
        self.spec_write = deep_update(spec_write, {"kvstore": {"path": str(path)}})
        self.spec_read = deep_update(spec_read, {"kvstore": {"path": str(path)}})

    def open_writer(self, spec=None):
        if not self.exists():
            return self.open_creator(spec)
        return ts.open(self.spec_write, write=True).result()

    def open_reader(self, spec=None):
        if spec is None:
            spec = self.spec_read
        else:
            spec = deep_update(copy.deepcopy(self.spec_read), spec)
        return ts.open(spec, read=True).result()

    def open_creator(self, spec=None, shape=None):
        if spec is None:
            spec = self.spec_create
        else:
            spec = deep_update(copy.deepcopy(self.spec_create), spec)
        if shape is not None:
            spec["metadata"]["shape"] = shape
        return ts.open(spec, create=True, delete_existing=True).result()

    def exists(self):
        return self.path.exists()

    @property
    def info(self):
        return {
            "disk_space": get_size(self.path),
        }

    def __getitem__(self, key):
        return self.open_reader()[key].read().result()

    def __setitem__(self, key, value):
        if not self.exists():
            writer = self.open_creator({"metadata": {"shape": value.shape}})
        else:
            writer = self.open_writer()

        writer[key] = value

    @property
    def shape(self):
        return self.open_reader().shape

    def __len__(self):
        return self.shape[0]

    @property
    def oindex(self):
        return OIndex(self.open_reader().oindex)

    def _repr_html_(self):
        shape = "[" + ",".join(str(x) for x in self.shape) + "]"
        if not str(self.path).startswith("memory"):
            size = format_size(get_size(self.path))
        else:
            size = ""
        return f"<span class='iconify' data-icon='mdi-axis-arrow'></span> <b>{self.name}</b> {shape}, {size}"

    def extend(self, value):
        if not self.exists():
            writer = self.open_creator()
        else:
            writer = self.open_writer()
            assert len(value.shape) == len(self.shape)
            assert value.shape[1:] == self.shape[1:]
        writer = writer.resize(exclusive_max=tuple([writer.shape[0] + value.shape[0], *writer.shape[1:]])).result()

        future = writer[writer.shape[0] - value.shape[0] :].write(value)
        future.result()


class OIndex:
    def __init__(self, reader):
        self.reader = reader

    def __getitem__(self, key):
        return self.reader[key].read().result()

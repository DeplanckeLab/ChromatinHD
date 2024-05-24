import copy
import os

import numpy as np
import shutil
import pickle

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
        return folder / (self.name + ".dat")

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
            if path.is_dir():
                shutil.rmtree(path)
            else:
                os.remove(path)
        self.__get__(obj)[:] = value

    def exists(self, obj):
        return self.__get__(obj).exists()

    def _repr_html_(self, obj):
        instance = self.__get__(obj)
        return instance._repr_html_()


class TensorstoreInstance:
    _obj = None

    _fixed_reader = None

    def __init__(self, name, path, spec_create, spec_write, spec_read):
        self.name = name
        self.path = path
        self.spec_create = deep_update(copy.deepcopy(spec_create), {"kvstore": {"path": str(path)}})
        self.spec_write = deep_update(copy.deepcopy(spec_write), {"kvstore": {"path": str(path)}})
        self.spec_read = deep_update(copy.deepcopy(spec_read), {"kvstore": {"path": str(path)}})

    @property
    def path_metadata(self):
        return self.path.with_suffix(".meta")

    def open_reader(self, spec=None, old=False):
        if self._fixed_reader is not None:
            return self._fixed_reader
        if spec is None:
            spec = self.spec_read
        else:
            spec = deep_update(copy.deepcopy(self.spec_read), spec)

        metadata = self.open_metadata()
        path = self.path
        if old:
            path = path.with_suffix(".dat.old")

        if not str(path).startswith("memory"):
            fp = np.memmap(path, dtype=metadata["dtype"], mode="r", shape=tuple(metadata["shape"]))
        else:
            fp = self._obj
        return fp

    def fix_reader(self):
        if self._fixed_reader is None:
            self._fixed_reader = self.open_reader()

    def open_writer(self, spec=None):
        if spec is None:
            spec = self.spec_write
        else:
            spec = deep_update(copy.deepcopy(self.spec_write), spec)
        metadata = self.open_metadata()
        if not str(self.path).startswith("memory"):
            fp = np.memmap(self.path, dtype=metadata["dtype"], mode="r+", shape=metadata["shape"])
        else:
            fp = self._obj
        pickle.dump(metadata, self.path_metadata.open("wb"))
        return fp

    def open_creator(self, spec=None, shape=None, dtype=None):
        if spec is None:
            spec = self.spec_create
        else:
            spec = deep_update(copy.deepcopy(self.spec_create), spec)
        if "metadata" not in spec:
            spec["metadata"] = {}
        metadata = spec["metadata"]

        if shape is not None:
            metadata["shape"] = shape

        if dtype is not None:
            metadata["dtype"] = dtype
        assert "dtype" in metadata
        assert "shape" in metadata

        if self.path.exists():
            self.path.unlink()
        if self.path_metadata.exists():
            self.path_metadata.unlink()

        if np.prod(metadata["shape"]) == 0:
            return None

        if not str(self.path).startswith("memory"):
            fp = np.memmap(self.path, dtype=metadata["dtype"], mode="w+", shape=tuple(metadata["shape"]))
        else:
            self._obj = np.zeros(metadata["shape"], dtype=metadata["dtype"])
            fp = self._obj
        pickle.dump(metadata, self.path_metadata.open("wb"))
        return fp

    def exists(self):
        return (self._obj is not None) or (self.path.exists() and self.path_metadata.exists())

    @property
    def info(self):
        return {
            "disk_space": get_size(self.path),
        }

    def __getitem__(self, key):
        return self.open_reader()[key]

    def __setitem__(self, key, value):
        if not self.exists():
            writer = self.open_creator(shape=value.shape, dtype=value.dtype.name)
        else:
            writer = self.open_writer()

        writer[key] = value

    def open_metadata(self):
        return pickle.load(self.path_metadata.open("rb"))

    @property
    def shape(self):
        return self.open_metadata()["shape"]

    def __len__(self):
        return self.shape[0]

    def _repr_html_(self):
        shape = "[" + ",".join(str(x) for x in self.shape) + "]"
        if not str(self.path).startswith("memory"):
            size = format_size(get_size(self.path))
        else:
            size = ""
        return f"<span class='iconify' data-icon='mdi-axis-arrow'></span> <b>{self.name}</b> {shape}, {size}"

    def extend(self, value):
        if not self.exists():
            writer = self.open_creator(dtype=value.dtype, shape=value.shape)
            writer[:] = value
        else:
            if not str(self.path).startswith("memory"):
                writer = self.open_writer()
                assert len(value.shape) == len(self.shape)
                assert value.shape[1:] == self.shape[1:]

                self.path.rename(self.path.with_suffix(".dat.old"))

                reader = self.open_reader(old=True)
                metadata = self.open_metadata()
                metadata["shape"] = (metadata["shape"][0] + value.shape[0], *metadata["shape"][1:])

                writer = self.open_creator(dtype=metadata["dtype"], shape=metadata["shape"])

                writer[: writer.shape[0] - value.shape[0]] = reader[:]
                writer[writer.shape[0] - value.shape[0] :] = value

                self.path.with_suffix(".dat.old").unlink()

                pickle.dump(metadata, self.path_metadata.open("wb"))
            else:
                self._obj = np.concatenate([self._obj, value], axis=0)
                metadata = self.open_metadata()
                metadata["shape"] = self._obj.shape
                pickle.dump(metadata, self.path_metadata.open("wb"))

    @property
    def oindex(self):
        return OIndex(self.open_reader())


class OIndex:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        if len(key) == 2:
            return self.value[key[0]][:, key[1]]
        return self.value[key]

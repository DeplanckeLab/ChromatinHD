import gzip
import importlib
import json
import pathlib
import pickle
import shutil
from typing import Union

import numpy as np
import torch

from .objects import is_instance, is_obj

PathLike = Union[str, pathlib.Path]


class Flowable(type):
    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)

        cls._obj_map = {}

        # compile objects
        for attr_id, attr in cls.__dict__.items():
            if is_obj(attr):
                assert isinstance(attr_id, str)
                attr.name = attr_id
                cls._obj_map[attr_id] = attr

    def __setattr__(cls, key, value):
        super().__setattr__(key, value)


class FlowObjects:
    def __init__(self, flow):
        self.flow = flow

    def __getattr__(self, key):
        if key != "flow":
            return self.flow.get(key)
        else:
            return super().__getattr__(key)


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

        self._obj_instances = {}

    def __copy__(self):
        return self.__class__(path=None)

    def __deepcopy__(self, memo):
        return self.__class__(path=None)

    @classmethod
    def create(cls, path=None, reset=False, **kwargs):
        if isinstance(path, str):
            path = pathlib.Path(path)
        self = cls(path=path, reset=reset)

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
        if k in self._obj_instances:
            return self._obj_instances[k]
        return self._obj_map[k]

    @property
    def o(self):
        return FlowObjects(self)

    def _repr_html_(self):
        import json

        from . import tipyte
        import pkg_resources
        from IPython.display import HTML, display
        import random

        lines = []
        lines.append(
            """
        """
        )
        lines += [
            "<div class='la-flow'>",
            "<strong>"
            + str(self.path.name)
            + "</strong>"
            + " "
            + f"<span class='soft'>({self.__class__.__module__}.{self.__class__.__name__})</span>",
        ]
        lines += ["<ul class='instances'>"]
        for obj_id, obj in self._obj_map.items():
            if obj_id in self._obj_instances:
                instance = self._obj_instances[obj_id]
                if instance.exists():
                    html_repr = instance._repr_html_()
                    lines.append(f"<li>{html_repr}</li>")
            elif is_obj(obj):
                if obj.exists(self):
                    html_repr = obj._repr_html_(self)
                    lines.append(f"<li>{html_repr}</li>")
        lines += ["</ul>"]

        # create template
        template = tipyte.template_to_function(
            pkg_resources.resource_filename("chromatinhd", "flow/flow_template.jinja2"), escaper=lambda x: x
        )

        parameters = {"div_id": random.randint(0, 50000), "html": "".join(lines)}
        html = template(**parameters)

        return html

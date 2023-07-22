import os
import pathlib
import pickle
import pandas as pd

from .torch import interpolate_1d
from .numpy import indices_to_indptr, indptr_to_indices
from .ansi import colorcodes
from .testing import repeated_kfold_corrected_t_test
from . import ecdf

__all__ = [
    "get_git_root",
    "get_output",
    "get_code",
    "name_window",
    "paircor",
    "interpolate_1d",
    "indices_to_indptr",
    "indptr_to_indices",
    "colorcodes",
    "repeated_kfold_corrected_t_test",
    "ecdf",
]


def get_git_root(cwd=None):
    """
    Gets the first parent root with a a .git folder
    """
    if cwd is None:
        cwd = os.getcwd()
    # go back until we find the git directory, signifying project root
    while ".git" not in os.listdir(cwd) and os.path.realpath(cwd) != "/":
        cwd = os.path.dirname(cwd)

    return pathlib.Path(cwd)


def get_output():
    return get_git_root() / "output"


def get_code():
    return get_git_root() / "code"


def name_window(window_info):
    return (
        window_info["chrom" if "chrom" in window_info.index else "chr"]
        + ":"
        + str(window_info["start"])
        + "-"
        + str(window_info["end"])
    )


def paircor(x, y, dim=-2):
    import torch
    import pandas as pd
    import numpy as np

    if isinstance(x, pd.DataFrame):
        x_ = x.values
        y_ = y.values
        divisor = y_.std(dim) * x_.std(dim)
        divisor[np.isclose(divisor, 0)] = 1.0
        cor = (
            (x_ - x_.mean(dim, keepdims=True)) * (y_ - y_.mean(dim, keepdims=True))
        ).mean(dim) / divisor
        cor = pd.Series(cor, x.index if dim == 1 else x.columns)
    elif torch.is_tensor(x):
        cor = ((x - x.mean(dim, keepdim=True)) * (y - y.mean(dim, keepdim=True))).mean(
            dim
        ) / (y.std(dim) * x.std(dim))
    else:
        divisor = y.std(dim) * x.std(dim)
        divisor[np.isclose(divisor, 0)] = 1.0
        cor = (
            (x - x.mean(dim, keepdims=True)) * (y - y.mean(dim, keepdims=True))
        ).mean(dim) / divisor
    return cor


def paircorr(x, y, dim=-2):
    import numpy as np

    divisor = y.std(dim, keepdims=True) * x.std(dim, keepdims=True)
    divisor[np.isclose(divisor, 0)] = 1.0
    cor = (x - x.mean(dim, keepdims=True)) * (
        y - y.mean(dim, keepdims=True)
    )  # / divisor
    return cor


def paircos(x, y, dim=0):
    import torch
    import pandas as pd
    import numpy as np

    if isinstance(x, pd.DataFrame):
        x_ = x.values
        y_ = y.values
        divisor = (y_**2).sum(dim) * (x_**2).sum(dim)
        divisor[np.isclose(divisor, 0)] = 1.0

        dot = (x_ * y_).sum(dim)
        cos = dot / divisor
        cos = pd.Series(cos, x.index if dim == 1 else x.columns)
    elif torch.is_tensor(x):
        cos = (x * y).sum(dim) / (
            torch.sqrt((y**2).sum(dim)) * torch.sqrt((x**2).sum(dim))
        )
    else:
        divisor = np.sqrt((y**2).sum(dim)) * np.sqrt((x**2).sum(dim))
        divisor[divisor == 0] = 1.0
        cos = (x * y).sum(dim) / divisor
    return cos


def fix_class(obj):
    import importlib

    module = importlib.import_module(obj.__class__.__module__)
    cls = getattr(module, obj.__class__.__name__)
    try:
        obj.__class__ = cls
    except TypeError:
        pass


class Pickler(pickle.Pickler):
    def reducer_override(self, obj):
        if any(
            obj.__class__.__module__.startswith(module) for module in ["chromatinhd."]
        ):
            fix_class(obj)
        else:
            # For any other object, fallback to usual reduction
            return NotImplemented

        return NotImplemented


def save(obj, fh, pickler=None, **kwargs):
    if pickler is None:
        pickler = Pickler
    return pickler(fh).dump(obj)


def crossing(*dfs):
    dfs = [df.copy() if isinstance(df, pd.DataFrame) else df.to_frame() for df in dfs]
    for df in dfs:
        df["___key"] = 0
    if len(dfs) == 0:
        return pd.DataFrame()
    dfs = [df for df in dfs if df.shape[0] > 0]  # remove empty dfs
    base = dfs[0]
    for df in dfs[1:]:
        base = pd.merge(base, df, on="___key")
    return base.drop(columns=["___key"])


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("peakfreeatac"):
            module = module.replace("peakfreeatac", "chromatinhd")
        return super().find_class(module, name)


def load(file):
    return Unpickler(file).load()

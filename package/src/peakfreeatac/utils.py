import os
import pathlib
import pickle

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


def name_window(window_info):
    return window_info["chrom" if "chrom" in window_info.index else "chr"] + ":" + str(window_info["start"]) + "-" + str(window_info["end"])

def paircor(x, y, dim = 0):
    import torch
    import pandas as pd
    import numpy as np
    if isinstance(x, pd.DataFrame):
        x_ = x.values
        y_ = y.values
        divisor = (y_.std(dim) * x_.std(dim))
        divisor[np.isclose(divisor, 0)] = 1.
        cor = ((x_ - x_.mean(dim, keepdims = True)) * (y_ - y_.mean(dim, keepdims = True))).mean(dim) / divisor
        cor = pd.Series(cor, x.index if dim == 1 else x.columns)
    elif torch.is_tensor(x):
        cor = ((x - x.mean(dim, keepdim = True)) * (y - y.mean(dim, keepdim = True))).mean(dim) / (y.std(dim) * x.std(dim))
    else:
        divisor = (y.std(dim) * x.std(dim))
        divisor[np.isclose(divisor, 0)] = 1.
        cor = ((x - x.mean(dim, keepdims = True)) * (y - y.mean(dim, keepdims = True))).mean(dim) / divisor
    return cor


def paircos(x, y, dim = 0):
    import torch
    import pandas as pd
    import numpy as np
    if isinstance(x, pd.DataFrame):
        x_ = x.values
        y_ = y.values
        divisor = ((y_**2).sum(dim) * (x_**2).sum(dim))
        divisor[np.isclose(divisor, 0)] = 1.

        dot = (x_ * y_).sum(dim)
        cos = dot / divisor
        cos = pd.Series(cos, x.index if dim == 1 else x.columns)
    elif torch.is_tensor(x):
        cos = (x * y).sum(dim) / (torch.sqrt((y**2).sum(dim)) * torch.sqrt((x**2).sum(dim)))
    else:
        divisor = (np.sqrt((y**2).sum(dim)) * np.sqrt((x**2).sum(dim)))
        divisor[divisor == 0] = 1.
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
            obj.__class__.__module__.startswith(module)
            for module in ["peakfreeatac."]
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
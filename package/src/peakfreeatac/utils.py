import os
import pathlib

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
    if isinstance(x, pd.DataFrame):
        x_ = x.values
        y_ = y.values
        cor = ((x_ - x_.mean(dim, keepdims = True)) * (y_ - y_.mean(dim, keepdims = True))).mean(dim) / (y_.std(dim) * x_.std(dim))
        cor = pd.Series(cor, x.index if dim == 1 else x.columns)
    elif torch.is_tensor(x):
        cor = ((x - x.mean(dim, keepdim = True)) * (y - y.mean(dim, keepdim = True))).mean(dim) / (y.std(dim) * x.std(dim))
    else:
        cor = ((x - x.mean(dim, keepdims = True)) * (y - y.mean(dim, keepdims = True))).mean(dim) / (y.std(dim) * x.std(dim))
    return cor
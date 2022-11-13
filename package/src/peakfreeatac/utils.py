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
    if torch.is_tensor(x):
        return ((x - x.mean(dim, keepdim = True)) * (y - y.mean(dim, keepdim = True))).mean(dim) / (y.std(dim) * x.std(dim))
    else:
        return ((x - x.mean(dim, keepdims = True)) * (y - y.mean(dim, keepdims = True))).mean(dim) / (y.std(dim) * x.std(dim))
import numpy as np


def indices_to_indptr(x, n):
    return np.pad(np.cumsum(np.bincount(x, minlength=n), 0), (1, 0))

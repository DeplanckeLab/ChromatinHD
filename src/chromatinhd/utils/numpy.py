import numpy as np


def indices_to_indptr(x, n):
    return np.pad(np.cumsum(np.bincount(x, minlength=n), 0), (1, 0))

ind2ptr = indices_to_indptr

def indptr_to_indices(x):
    n = len(x) - 1
    return np.repeat(np.arange(n), np.diff(x))

ptr2ind = indptr_to_indices
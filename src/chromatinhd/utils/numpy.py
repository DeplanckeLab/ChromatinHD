import numpy as np


def indices_to_indptr(x, n, dtype=np.int32):
    return np.pad(np.cumsum(np.bincount(x, minlength=n), 0, dtype=dtype), (1, 0))


ind2ptr = indices_to_indptr


def indptr_to_indices(x):
    n = len(x) - 1
    return np.repeat(np.arange(n), np.diff(x))


ptr2ind = indptr_to_indices


def indices_to_indptr_chunked(x, n, dtype=np.int32, batch_size=10e3):
    counts = np.zeros(n + 1, dtype=dtype)
    cur_value = 0
    for a, b in zip(
        np.arange(0, len(x), batch_size, dtype=int), np.arange(batch_size, len(x) + batch_size, batch_size, dtype=int)
    ):
        x_ = x[a:b]
        assert (x_ >= cur_value).all()
        bincount = np.bincount(x_ - cur_value)
        counts[(cur_value + 1) : (cur_value + len(bincount) + 1)] += bincount
        cur_value = x_[-1]
    indptr = np.cumsum(counts, dtype=dtype)
    return indptr

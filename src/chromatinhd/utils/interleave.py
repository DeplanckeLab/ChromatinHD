import torch
import numpy as np


def interleave(x, repeats=np.array([1, 2, 4, 8, 16])):
    assert isinstance(repeats, np.ndarray)
    out_shape = int(x.shape[-1] // (1 / repeats).sum())
    out = torch.zeros((*x.shape[:-1], out_shape), dtype=x.dtype, device=x.device)
    k = out_shape // repeats
    i = 0
    for k, r in zip(k, repeats):
        out += torch.repeat_interleave(x[..., i : i + k], r)
        i += k
    return out


def deinterleave(y, repeats=np.array([1, 2, 4, 8, 16])):
    assert isinstance(repeats, np.ndarray)
    x = []
    for r in repeats[::-1]:
        x_ = y.reshape((*y.shape[:-1], -1, r)).mean(dim=-1)
        y = y - torch.repeat_interleave(x_, r)
        x.append(x_)
    return torch.cat(x[::-1], -1)

import torch


def interpolate_1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    a = (fp[..., 1:] - fp[..., :-1]) / (xp[..., 1:] - xp[..., :-1])
    b = fp[..., :-1] - (a.mul(xp[..., :-1]))

    indices = torch.searchsorted(xp.contiguous(), x.contiguous(), right=False) - 1
    indices = torch.clamp(indices, 0, a.shape[-1] - 1)
    slope = a.index_select(a.ndim - 1, indices)
    intercept = b.index_select(a.ndim - 1, indices)
    return x * slope + intercept


def indices_to_indptr(x, n):
    return torch.nn.functional.pad(
        torch.cumsum(torch.bincount(x, minlength=n), 0), (1, 0)
    )

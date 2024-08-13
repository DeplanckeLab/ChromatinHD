import chromatinhd as chd
import torch
import numpy as np


def test_interleave():
    y = chd.utils.interleave.interleave(torch.tensor([1, 2, 3, 4, 2, 3], dtype=float), repeats=np.array([1, 2]))
    assert torch.allclose(y, torch.tensor([3.0, 4.0, 6.0, 7.0], dtype=float))

    x = chd.utils.interleave.deinterleave(y, repeats=np.array([1, 2]))
    assert torch.allclose(x, torch.tensor([-0.5000, 0.5000, -0.5000, 0.5000, 3.5000, 6.5000], dtype=torch.float64))

    y2 = chd.utils.interleave.interleave(x, repeats=np.array([1, 2]))
    assert torch.allclose(y, y2)


def test_interleave2():
    y = torch.tensor([1, 2, 3, 4, 2, 3], dtype=float)
    x = chd.utils.interleave.deinterleave(y, repeats=np.array([1, 2, 3]))
    y2 = chd.utils.interleave.interleave(x, repeats=np.array([1, 2, 3]))
    assert torch.allclose(y, y2)

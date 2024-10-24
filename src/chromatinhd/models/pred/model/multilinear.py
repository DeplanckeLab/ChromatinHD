import torch
from chromatinhd.embedding import FeatureParameter
import math


class MultiLinear(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        bias: bool = True,
        device=None,
        dtype=None,
        weight_constructor=None,
        bias_constructor=None,
    ):
        super().__init__()

        self.out_features = out_features

        if bias:
            if bias_constructor is None:

                def bias_constructor(shape):
                    stdv = 1.0 / math.sqrt(shape[-1])
                    return torch.empty(shape, device=device, dtype=dtype).uniform_(-stdv, stdv)

            bias = FeatureParameter(n_heads, (out_features,), constructor=bias_constructor)
            self.register_module("bias", bias)
        else:
            self.bias = None

        if weight_constructor is None:

            def weight_constructor(shape):
                stdv = 1.0 / math.sqrt(shape[-1])
                return torch.empty(shape, device=device, dtype=dtype).uniform_(-stdv, stdv)

            torch.nn.Linear

        self.weight = FeatureParameter(
            n_heads,
            (
                in_features,
                out_features,
            ),
            constructor=weight_constructor,
        )

    def forward(self, input: torch.Tensor, indptr, regions_oi):
        outputs = []

        if self.bias is not None:
            for ix, start, end in zip(regions_oi, indptr[:-1], indptr[1:]):
                outputs.append(torch.matmul(input[start:end], self.weight[ix]) + self.bias[ix])
        else:
            for ix, start, end in zip(regions_oi, indptr[:-1], indptr[1:]):
                outputs.append(torch.matmul(input[start:end], self.weight[ix]))
        output = torch.cat(outputs, dim=0)
        return output

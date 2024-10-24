from __future__ import annotations
import torch
import numpy as np


class EmbeddingTensor(torch.nn.Embedding):
    """
    Simple wrapper around torch.nn.Embedding which allows an embedding of any number of dimensions instead of just 1

    The actual tensor underyling this embedding can be accessed using
    `x.weight`, although this will have dimensions [num_embeddings, prod(embedding_dims)]

    To get the underlying tensor in the correct dimensions, you can use
    `x.data`, which will have dimensions [num_embeddings, *embedding_dims]. This can also be used to set the value.
    """

    def __init__(self, num_embeddings, embedding_dims, *args, constructor=None, **kwargs):
        if not isinstance(embedding_dims, tuple):
            embedding_dims = tuple(embedding_dims)
        if len(embedding_dims) == 0:
            embedding_dim = 1
            embedding_dims = tuple([1])
        else:
            embedding_dim = np.prod(embedding_dims)
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)
        if constructor is not None:
            self.weight.data = constructor(self.weight.data.shape)
        self.embedding_dims = embedding_dims

    def forward(self, input):
        if isinstance(input, int):
            return super().forward(torch.tensor([input])).view(self.embedding_dims)
        elif input.ndim == 0:
            return super().forward(input.unsqueeze(0)).view(self.embedding_dims)
        return super().forward(input).view((input.shape[0], *self.embedding_dims))

    def extra_repr(self) -> str:
        s = "{num_embeddings}, {embedding_dims}"
        if self.padding_idx is not None:
            s += ", padding_idx={padding_idx}"
        if self.max_norm is not None:
            s += ", max_norm={max_norm}"
        if self.norm_type != 2:
            s += ", norm_type={norm_type}"
        if self.scale_grad_by_freq is not False:
            s += ", scale_grad_by_freq={scale_grad_by_freq}"
        if self.sparse is not False:
            s += ", sparse=True"
        return s.format(**self.__dict__)

    def get_full_weight(self):
        return self.weight.view((self.weight.shape[0], *self.embedding_dims))

    @property
    def data(self):
        """
        The data of the parameter in dimensions [num_embeddings, *embedding_dims]
        """
        return self.get_full_weight().data

    @data.setter
    def data(self, value):
        if value.ndim == 2:
            self.weight.data = value
        else:
            self.weight.data = value.reshape(self.weight.data.shape)

    @property
    def shape(self):
        """
        The shape of the parameter, i.e. [num_embeddings, *embedding_dims]
        """
        return (self.weight.shape[0], *self.embedding_dims)

    def __getitem__(self, k):
        return self.forward(k)

    @classmethod
    def from_pretrained(cls, pretrained: EmbeddingTensor):
        self = cls(pretrained.num_embeddings, pretrained.embedding_dims)
        self.data = pretrained.data
        self.weight.requires_grad = False

        return self


class FeatureParameter(torch.nn.Module):
    _params = tuple()

    def __init__(self, num_embeddings, embedding_dims, constructor=torch.zeros, *args, **kwargs):
        super().__init__()
        params = []
        for i in range(num_embeddings):
            params.append(torch.nn.Parameter(constructor(embedding_dims, *args, **kwargs)))
            self.register_parameter(str(i), params[-1])
        self._params = tuple(params)

    def __getitem__(self, k):
        return self._params[k]

    def __setitem__(self, k, v):
        self._params = list(self._params)
        self._params[k] = v
        self._params = tuple(self._params)

    def __call__(self, ks):
        return torch.stack([self._params[k] for k in ks], 0)

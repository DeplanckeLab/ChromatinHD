import torch
import numpy as np


class EmbeddingTensor(torch.nn.Embedding):
    """
    Simple wrapper around torch.nn.Embedding which allows an embedding of any dimensions

    The actual tensor underyling this embedding can be accessed using
    `x.weight`, although this will have dimensions [num_embeddings, prod(embedding_dims)]

    To get the underlying tensor in the correct dimensions, you can use
    `x.get_full_weight()`, which will have dimensions [num_embeddings, *embedding_dims]

    To set the value of this parameter, without tracking gradients, use `x.data`
    """

    def __init__(self, num_embeddings, embedding_dims, *args, **kwargs):
        if not isinstance(embedding_dims, tuple):
            embedding_dims = tuple(embedding_dims)
        if len(embedding_dims) == 0:
            embedding_dim = 1
            embedding_dims = tuple([1])
        else:
            embedding_dim = np.prod(embedding_dims)
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)
        self.embedding_dims = embedding_dims

    def forward(self, input):
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
        assert value.shape == self.weight.shape
        self.weight.data = value

    @property
    def shape(self):
        """
        The shape of the parameter, i.e. [num_embeddings, *embedding_dims]
        """
        return (self.weight.shape[0], *self.embedding_dims)

    def __getitem__(self, k):
        return self.forward(k)

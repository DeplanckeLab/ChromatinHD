
import torch
import numpy as np

class EmbeddingTensor(torch.nn.Embedding):
    """
    Simple wrapper around torch.nn.Embedding which allows an embedding of any dimensions
    """
    
    def __init__(self, num_embeddings, embedding_dims, *args, **kwargs):
        if not isinstance(embedding_dims, tuple):
            embedding_dims = tuple(embedding_dims)
        embedding_dim = np.prod(embedding_dims)
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)
        self.embedding_dims = embedding_dims
        
    def forward(self, input):
        return super().forward(input).view((input.shape[0], *self.embedding_dims))

    def extra_repr(self) -> str:
        s = '{num_embeddings}, {embedding_dims}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        return s.format(**self.__dict__)
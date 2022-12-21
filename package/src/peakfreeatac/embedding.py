
import torch
import numpy as np

class EmbeddingTensor(torch.nn.Embedding):
    """
    Simple wrapper around torch.nn.Embedding which allows an embedding of any dimensions
    """
    
    def __init__(self, num_embeddings, embedding_dims, *args, **kwargs):
        embedding_dim = np.prod(embedding_dims)
        super().__init__(num_embeddings, embedding_dim, *args, **kwargs)
        self.embedding_dims = embedding_dims
        
    def forward(self, input):
        return super().forward(input).view((input.shape[0], *self.embedding_dims))

import torch
class SparseDenseAdam(torch.optim.Optimizer):
    def __init__(self, parameters_sparse, parameters_dense, lr = 1e-3, weight_decay = 0., **kwargs):
        if len(parameters_sparse) == 0:
            self.optimizer_sparse = None
        else:
            self.optimizer_sparse = torch.optim.SparseAdam(
                parameters_sparse,
                lr = lr,
                **kwargs
            )
        if len(parameters_dense) == 0:
            self.optimizer_dense = None
        else:
            self.optimizer_dense = torch.optim.Adam(
                parameters_dense,
                lr=lr,
                weight_decay = weight_decay,
                **kwargs
            )
            
    def zero_grad(self):
        if self.optimizer_sparse is not None:
            self.optimizer_sparse.zero_grad()
        if self.optimizer_dense is not None:
            self.optimizer_dense.zero_grad()
            
    def step(self):
        if self.optimizer_sparse is not None:
            self.optimizer_sparse.step()
        if self.optimizer_dense is not None:
            self.optimizer_dense.step()
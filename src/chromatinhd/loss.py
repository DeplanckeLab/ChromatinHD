import torch

def normlinreg(x, y):
    assert x.ndim == 2
    assert x.shape == y.shape
    X = torch.stack([torch.ones(x.shape, device = x.device), x], 1)
    
    coefficients = (torch.linalg.inv(X @ X.transpose(-1, -2)) @ X) @ y.unsqueeze(-1)
    
    return x * coefficients[:, 1] + coefficients[:, 0]

class MSENormLoss2(torch.nn.MSELoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        input_normalized = normlinreg(input, target)
        return super().forward(input_normalized, target) * 100

class MSENormLoss(torch.nn.MSELoss):
    def __init__(self, *args, dim = 0, eps = 1e-1, **kwargs):
        self.dim = dim
        self.eps = eps
        super().__init__(*args, **kwargs)
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        input_normalized = (
            (input - input.mean(self.dim, keepdim = True) + target.mean(self.dim, keepdim = True)) * 
            (target.std(self.dim, keepdim = True) / (input.std(self.dim, keepdim = True) + self.eps))
        )
        return super().forward(input_normalized, target) * 100
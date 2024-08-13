import torch
import itertools


class SparseDenseAdam(torch.optim.Optimizer):
    """
    Optimize both sparse and densre parameters using ADAM
    """

    def __init__(self, parameters_sparse, parameters_dense, lr=1e-3, weight_decay=0.0, **kwargs):
        if len(parameters_sparse) == 0:
            self.optimizer_sparse = None
        else:
            self.optimizer_sparse = torch.optim.SparseAdam(parameters_sparse, lr=lr, **kwargs)
        if len(parameters_dense) == 0:
            self.optimizer_dense = None
        else:
            self.optimizer_dense = torch.optim.Adam(parameters_dense, lr=lr, weight_decay=weight_decay, **kwargs)

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

    @property
    def param_groups(self):
        return itertools.chain(
            self.optimizer_dense.param_groups if self.optimizer_dense is not None else [],
            self.optimizer_sparse.param_groups if self.optimizer_sparse is not None else [],
        )


class AdamPerFeature(torch.optim.Optimizer):
    def __init__(self, parameters, n_features, lr=1e-3, weight_decay=0.0, **kwargs):
        self.n_features = n_features

        self.adams = []
        for i in range(n_features):
            parameters_features = [p[i] for p in parameters]
            self.adams.append(torch.optim.Adam(parameters_features, lr=lr, weight_decay=weight_decay, **kwargs))

    def zero_grad(self, feature_ixs):
        for i in feature_ixs:
            self.adams[i].zero_grad()

    def step(self, feature_ixs):
        for i in feature_ixs:
            self.adams[i].step()

    @property
    def param_groups(self):
        return itertools.chain(*[adam.param_groups for adam in self.adams])


class SGDPerFeature(torch.optim.Optimizer):
    def __init__(self, parameters, n_features, lr=1e-3, weight_decay=0.0, **kwargs):
        self.n_features = n_features

        self.adams = []
        for i in range(n_features):
            parameters_features = [p[i] for p in parameters]
            self.adams.append(
                torch.optim.SGD(parameters_features, lr=lr, weight_decay=weight_decay, momentum=0.5, **kwargs)
            )

    def zero_grad(self, feature_ixs):
        for i in feature_ixs:
            self.adams[i].zero_grad()

    def step(self, feature_ixs):
        for i in feature_ixs:
            self.adams[i].step()

    @property
    def param_groups(self):
        return itertools.chain(*[adam.param_groups for adam in self.adams])

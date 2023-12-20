import torch
from chromatinhd.flow import Flow, Stored


def get_sparse_parameters(module, parameters):
    for module in module._modules.values():
        for p in module.parameters_sparse() if hasattr(module, "parameters_sparse") else []:
            parameters.add(p)
        parameters = get_sparse_parameters(module, parameters)
    return parameters


class HybridModel:
    def parameters_dense(self, autoextend=True):
        """
        Get all dense parameters of the model
        """
        parameters = [
            parameter
            for module in self._modules.values()
            for parameter in (module.parameters_dense() if hasattr(module, "parameters_dense") else [])
        ]

        # extend with any left over parameters that were not specified in parameters_dense or parameters_sparse
        def contains(x, y):
            return any([x is y_ for y_ in y])

        parameters_sparse = set(self.parameters_sparse())

        if autoextend:
            for p in self.parameters():
                if p not in parameters_sparse:
                    parameters.append(p)
        parameters = [p for p in parameters if p.requires_grad]
        return parameters

    def parameters_sparse(self, autoextend=True):
        """
        Get all sparse parameters in a model
        """
        parameters = set()

        parameters = get_sparse_parameters(self, parameters)
        parameters = [p for p in parameters if p.requires_grad]
        return parameters


class FlowModel(torch.nn.Module, HybridModel, Flow):
    state = Stored()

    def __init__(self, path=None, reset=False, **kwargs):
        torch.nn.Module.__init__(self)
        Flow.__init__(self, path=path, reset=reset, **kwargs)

        if self.o.state.exists(self):
            if reset:
                raise ValueError("Cannot reset a model that has already been initialized")
            self.restore_state()

    def save_state(self):
        from collections import OrderedDict

        state = OrderedDict()
        for k, v in self.__dict__.items():
            if k.lstrip("_") in self._obj_map:
                continue
            if k == "path":
                continue
            state[k] = v
        self.state = state

    @classmethod
    def restore(cls, path):
        self = cls.__new__(cls)
        Flow.__init__(self, path=path)
        self.restore_state()
        return self

    def restore_state(self):
        state = self.state
        for k, v in state.items():
            # if k.lstrip("_") in self._obj_map:
            #     continue
            self.__dict__[k] = v
        return self

import torch


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

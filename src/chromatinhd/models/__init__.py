class HybridModel:
    def parameters_dense(self, autoextend=True):
        """
        Get all dense parameters of the model
        """
        parameters = [
            parameter
            for module in self._modules.values()
            for parameter in (
                module.parameters_dense() if hasattr(module, "parameters_dense") else []
            )
        ]

        # extend with any left over parameters that were not specified in parameters_dense or parameters_sparse
        def contains(x, y):
            return any([x is y_ for y_ in y])

        if autoextend:
            parameters.extend(
                [
                    p
                    for p in self.parameters()
                    if (not contains(p, self.parameters_sparse()))
                    and (not contains(p, parameters))
                ]
            )
        return parameters

    def parameters_sparse(self):
        """
        Get all sparse parameters in a model
        """
        return [
            parameter
            for module in self._modules.values()
            for parameter in (
                module.parameters_sparse()
                if hasattr(module, "parameters_sparse")
                else []
            )
        ]

from . import pred
from . import diff
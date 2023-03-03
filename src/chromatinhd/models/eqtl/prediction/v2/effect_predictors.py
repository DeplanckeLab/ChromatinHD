import torch


class EffectPredictorLinear(torch.nn.Module):
    def __init__(self, n_embedding_dimensions, n_variantxgenes: int):
        super().__init__()
        self.n_embedding_dimensions = n_embedding_dimensions
        self.nn = torch.nn.Sequential(torch.nn.Linear(n_embedding_dimensions, 1))
        self.nn[0].weight.data.zero_()

        self.variantxgene_effect = torch.nn.Parameter(torch.zeros(n_variantxgenes))

    def forward(self, variantxgene_embedding, variantxgene_ixs):
        prioritization = torch.exp(self.nn(variantxgene_embedding).squeeze(-1))
        effect = self.variantxgene_effect[variantxgene_ixs] * prioritization

        return effect


class EffectPredictor(torch.nn.Module):
    def __init__(
        self,
        n_embedding_dimensions,
        n_variantxgenes: int,
        variantxgene_effect=None,
        n_layers=0,
    ):
        super().__init__()
        self.n_embedding_dimensions = n_embedding_dimensions

        if n_layers == 0:
            self.nn = torch.nn.Sequential(
                torch.nn.Linear(n_embedding_dimensions, 1),
            )
            self.nn[0].weight.data.zero_()
        else:
            layers = [torch.nn.Linear(n_embedding_dimensions, 10)]
            for layer_ix in range(n_layers):
                layers.extend(
                    [
                        torch.nn.ReLU(),
                        torch.nn.Linear(10, 10),
                    ]
                )
            layers.append(
                torch.nn.Linear(10, 1),
            )
            self.nn = torch.nn.Sequential(*layers)

        if variantxgene_effect is not None:
            self.register_buffer("variantxgene_effect", variantxgene_effect)
        else:
            self.variantxgene_effect = torch.nn.Parameter(torch.zeros(n_variantxgenes))

    def forward(self, variantxgene_embedding, variantxgene_ixs):
        prioritization = torch.sigmoid(self.nn(variantxgene_embedding))
        effect = self.variantxgene_effect[variantxgene_ixs] * prioritization.squeeze(-1)

        return effect

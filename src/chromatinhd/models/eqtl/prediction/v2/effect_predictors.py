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
            self.nn[0].bias.data[:] = -2.0
        else:
            n_intermediate_dimensions = 10
            layers = []

            current_n_dimensions = n_embedding_dimensions
            for layer_ix in range(n_layers):
                layers.extend(
                    [
                        torch.nn.Linear(
                            current_n_dimensions, n_intermediate_dimensions
                        ),
                        torch.nn.ReLU(),
                    ]
                )
                current_n_dimensions = n_intermediate_dimensions
            layers.append(
                torch.nn.Linear(current_n_dimensions, 1),
            )
            self.nn = torch.nn.Sequential(*layers)

        if variantxgene_effect is not None:
            self.register_buffer("variantxgene_effect", variantxgene_effect)
        else:
            self.variantxgene_effect = torch.nn.Parameter(torch.zeros(n_variantxgenes))

        self.embedding_bias = torch.nn.Parameter(torch.zeros(n_embedding_dimensions))

    def forward(self, variantxgene_embedding, variantxgene_ixs):
        # import matplotlib.pyplot as plt

        # fig, ax = plt.subplots()
        # ax.matshow(variantxgene_embedding[..., -1].cpu().detach().numpy())
        prioritization = torch.sigmoid(
            self.nn(variantxgene_embedding + self.embedding_bias)
        )
        self.prioritization = prioritization
        effect = self.variantxgene_effect[variantxgene_ixs] * prioritization.squeeze(-1)
        self.effect = effect

        return effect, prioritization

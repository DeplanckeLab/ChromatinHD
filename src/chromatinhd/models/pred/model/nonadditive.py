import torch
import torch_scatter
import math
from chromatinhd.embedding import EmbeddingTensor
from chromatinhd.models import FlowModel
from chromatinhd.flow import Stored, Linked
from .additive import Model as BaseModel


class LinearEmbedded(torch.nn.Module):
    def __init__(self, n_input_features, n_output_features, n_regions):
        super().__init__()
        self.bias1 = EmbeddingTensor(
            n_regions,
            tuple([n_output_features]),
            sparse=True,
        )
        self.bias1.data[:] = 0.0
        self.weight1 = EmbeddingTensor(
            n_regions,
            tuple([n_input_features, n_output_features]),
            sparse=True,
        )
        self.weight1.data[:] = 0.0

    def forward(self, x, region_ix):
        return torch.einsum("abc,bcd->abd", x, self.weight1(region_ix)) + self.bias1(region_ix)

    def parameters_sparse(self):
        return [self.bias1.weight, self.weight1.weight]


class EmbeddingToExpression(torch.nn.Module):
    """
    Predicts region expression using a [cell, region, component] embedding in a region-specific manner
    """

    def __init__(self, n_regions, n_embedding_dimensions=5, n_intermediate_dimensions=5, **kwargs):
        self.n_regions = n_regions
        self.n_embedding_dimensions = n_embedding_dimensions

        super().__init__()

        self.linear1 = LinearEmbedded(n_embedding_dimensions, n_intermediate_dimensions, n_regions)

        self.linear2 = LinearEmbedded(n_intermediate_dimensions, 1, n_regions)

    def forward(self, cell_region_embedding, region_ix):
        out = self.linear1(cell_region_embedding, region_ix)
        out = torch.sigmoid(out)
        out = self.linear2(out, region_ix).squeeze(-1)
        return out

    def parameters_sparse(self):
        return self.linear1.parameters_sparse() + self.linear2.parameters_sparse()


class Model(FlowModel):
    transcriptome = Linked()
    fragments = Linked()
    fold = Stored()

    layer = Stored()

    def __init__(
        self,
        path=None,
        base_model=None,
    ):
        super().__init__(
            path=path,
        )

        if base_model is not None:
            self.fragments = base_model.fragments
            self.transcriptome = base_model.transcriptome
            self.layer = base_model.layer
            self.fold = base_model.fold

            self.fragment_embedder = base_model.fragment_embedder
            self.embedding_region_pooler = base_model.embedding_region_pooler
            self.embedding_to_expression = base_model.embedding_to_expression
            self.embedding_to_expression2 = EmbeddingToExpression(
                base_model.embedding_to_expression.n_regions,
                base_model.embedding_to_expression.n_embedding_dimensions,
            )

    def forward(self, data):
        fragment_embedding = self.fragment_embedder(data.fragments.coordinates, data.fragments.regionmapping)
        cell_region_embedding = self.embedding_region_pooler(
            fragment_embedding, data.fragments.local_cellxregion_ix, data.minibatch.n_cells, data.minibatch.n_regions
        )
        expression_predicted = self.embedding_to_expression(
            cell_region_embedding, data.minibatch.regions_oi_torch
        ) + self.embedding_to_expression2(cell_region_embedding, data.minibatch.regions_oi_torch)
        return expression_predicted

    def forward_multiple(self, data, fragments_oi, extract_total=False):
        raise NotImplementedError()
        fragment_embedding = self.fragment_embedder(data.coordinates, data.regionmapping)

        for fragments_oi_ in fragments_oi:
            if fragments_oi_ is not None:
                fragment_embedding_ = fragment_embedding[fragments_oi_]
                local_cellxregion_ix = data.fragments.local_cellxregion_ix[fragments_oi_]
            else:
                fragment_embedding_ = fragment_embedding
                local_cellxregion_ix = data.fragments.local_cellxregion_ix

            cell_region_embedding = self.embedding_region_pooler(
                fragment_embedding_,
                local_cellxregion_ix,
                data.n_cells,
                data.n_regions,
            )
            expression_predicted = self.embedding_to_expression.forward(
                cell_region_embedding, data.regions_oi_torch
            ) + self.embedding_to_expression2.forward(cell_region_embedding, data.regions_oi_torch)

            if extract_total:
                n_fragments = torch.bincount(
                    data.local_cellxregion_ix,
                    minlength=data.n_regions * data.n_cells,
                ).reshape((data.n_cells, data.n_regions))
                yield expression_predicted, n_fragments
            else:
                if fragments_oi_ is None:
                    n_fragments_lost = 0
                else:
                    n_fragments_lost = torch.bincount(
                        data.local_cellxregion_ix[~fragments_oi_],
                        minlength=data.n_regions * data.n_cells,
                    ).reshape((data.n_cells, data.n_regions))
                yield expression_predicted, n_fragments_lost

    forward_loss = BaseModel.forward_loss
    train_model = BaseModel.train_model
    forward_region_loss = BaseModel.forward_region_loss
    get_prediction = BaseModel.get_prediction

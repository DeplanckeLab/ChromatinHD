"""
- a positional encoding per fragment
- summarizes the encoding using a linear layer to 3 dimensions
- self-attention
- summation over cellxgene
- summation over the component
- gene expression by adding an intercept

Intuitively, for each gene, a fragment at a particular position has a positive or negative impact on expression
This effect is simply summed, without any interactions between positions
"""


import torch
import torch_scatter
import math

def self_attention(x):
    dotproduct = torch.matmul(x, x.transpose(-1, -2)) / math.sqrt(x.shape[-2])
    weights = torch.nn.functional.softmax(dotproduct, -1)
    # print(weights)
    # print(weights.shape)
    # print(x.shape)
    y = torch.matmul(weights, x)
    return y

class SineEncoding(torch.nn.Module):
    def __init__(self, n_frequencies):
        super().__init__()

        self.register_buffer(
            "frequencies",
            # torch.tensor([[] * 2 for i in range(1, n_frequencies + 1)]).flatten(-2)
            torch.tensor([[1 / 1000**(2 * i/n_frequencies)] * 2 for i in range(1, n_frequencies + 1)]).flatten(-2)
        )
        self.register_buffer(
            "shifts",
            torch.tensor([[0, torch.pi/2] for _ in range(1, n_frequencies + 1)]).flatten(-2)
        )

        self.n_embedding_dimensions = n_frequencies * 2 * 2

    def forward(self, coordinates):
        embedding = torch.sin((coordinates[..., None] * self.frequencies + self.shifts).flatten(-2))
        return embedding

class FragmentEmbedder(torch.nn.Module):
    def __init__(self, n_genes, n_frequencies = 20, n_embedding_dimensions = 5, nonlinear = True,  selfatt = True, **kwargs):
        
        self.n_embedding_dimensions = n_embedding_dimensions

        self.nonlinear = nonlinear
        
        super().__init__(**kwargs)

        self.sine_encoding = SineEncoding(n_frequencies = n_frequencies)

        # default initialization same as a torch.nn.Linear
        self.weight1 = torch.nn.Parameter(torch.empty((n_genes, self.sine_encoding.n_embedding_dimensions, self.n_embedding_dimensions), requires_grad = True))
        stdv = 1. / math.sqrt(self.weight1.size(-1))# / 100
        self.weight1.data.uniform_(-stdv, stdv)

        # default initialization same as a torch.nn.Linear
        self.weight2 = torch.nn.Parameter(torch.empty((n_genes, self.sine_encoding.n_embedding_dimensions, self.n_embedding_dimensions), requires_grad = True))
        stdv = 1. / math.sqrt(self.weight1.size(-1))# / 100
        self.weight2.data.uniform_(-stdv, stdv)

        self.selfatt = selfatt
    
    def forward(self, coordinates, gene_ix, n):
        encoding = self.sine_encoding(coordinates)
        embedding = (encoding[..., None] * self.weight1[gene_ix]).sum(-2)

        # non-linear
        embedding = torch.sigmoid(embedding)

        # self-attention
        embedding = embedding.clone()

        if self.selfatt:
            for i, n_ in zip(range(2, len(n) + 2), n):
                embedding_att = (encoding[..., None] * self.weight2[gene_ix[i]]).sum(-2)
                x_stacked = embedding[n_]
                x = x_stacked.reshape((x_stacked.shape[0]//i, i, *x_stacked.shape[1:]))
                embedding[n_] = self_attention(x).reshape(x_stacked.shape)

        return embedding
    
class EmbeddingGenePooler(torch.nn.Module):
    """
    Pools fragments across genes and cells
    """
    def __init__(self, reduce = "sum"):
        self.reduce = reduce
        super().__init__()
    
    def forward(self, embedding, fragment_cellxgene_ix, cell_n, gene_n):
        if self.reduce == "sum":
            cellxgene_embedding = torch_scatter.segment_sum_coo(embedding, fragment_cellxgene_ix, dim_size = cell_n * gene_n)
        else:
            raise ValueError()
        cell_gene_embedding = cellxgene_embedding.reshape((cell_n, gene_n, cellxgene_embedding.shape[-1]))
        return cell_gene_embedding
    
class EmbeddingToExpression(torch.nn.Module):
    """
    Predicts gene expression using a [cell, gene, component] embedding in a gene-specific manner
    """
    def __init__(self, n_genes, mean_gene_expression, n_embedding_dimensions = 100, **kwargs):
        self.n_genes = n_genes
        self.n_embedding_dimensions = n_embedding_dimensions
        
        super().__init__()
        
        # set bias to empirical mean
        self.bias1 = torch.nn.Parameter(mean_gene_expression.clone().detach().to("cpu"), requires_grad = True)

        self.weight1 = torch.nn.Parameter(torch.ones((n_genes, n_embedding_dimensions), requires_grad = True))
        
    def forward(self, cell_gene_embedding, gene_ix):
        return (cell_gene_embedding * self.weight1[gene_ix]).sum(-1) + self.bias1[gene_ix]
    
class Model(torch.nn.Module):
    def __init__(
        self,
        loader,
        n_genes,
        mean_gene_expression,
        dummy = False,
        n_frequencies = 50,
        reduce = "sum",
        nonlinear = True,
        n_embedding_dimensions = 5,
        selfatt = True,
        **kwargs
    ):
        super().__init__()
        
        if dummy:
            self.fragment_embedder = FragmentEmbedderCounter()
        else:
            self.fragment_embedder = FragmentEmbedder(
                n_frequencies = n_frequencies,
                n_genes = n_genes,
                nonlinear = nonlinear,
                n_embedding_dimensions=n_embedding_dimensions,
                selfatt = selfatt
            )
        self.embedding_gene_pooler = EmbeddingGenePooler(reduce = reduce)
        self.embedding_to_expression = EmbeddingToExpression(
            n_genes = n_genes,
            n_embedding_dimensions = self.fragment_embedder.n_embedding_dimensions,
            mean_gene_expression = mean_gene_expression
        )

    def forward(
        self,
        data
    ):
        fragment_embedding = self.fragment_embedder(data.coordinates, data.genemapping, data.n)
        cell_gene_embedding = self.embedding_gene_pooler(fragment_embedding, data.local_cellxgene_ix, data.n_cells, data.n_genes)
        expression_predicted = self.embedding_to_expression(cell_gene_embedding, data.genes_oi)
        return expression_predicted
        

    def get_parameters(self):
        return self.parameters()
        # return self.embedding_gene_pooler.parameters()
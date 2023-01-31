"""
"""


import torch
import torch_scatter
    
class FragmentsToExpression(torch.nn.Module):
    """
    Pools fragments across genes and cells
    """
    def __init__(self, mean_gene_expression, n_hidden_dimensions = 50):
        super().__init__()

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(1, n_hidden_dimensions),
            torch.nn.Sigmoid(),
            torch.nn.Linear(n_hidden_dimensions, 1)
        )

        self.bias1 = torch.nn.Parameter(mean_gene_expression.clone().detach().to("cpu"), requires_grad = True)
    
    def forward(self, cellxgene_ix, n_cells, n_genes, gene_ix):
        # count the number of fragments
        fragment_counts = torch.bincount(cellxgene_ix, minlength = n_cells * n_genes).reshape((n_cells, n_genes)).to(torch.float)

        # learn an importance from the number of fragments
        cellxgene_embedding = self.nn(fragment_counts.unsqueeze(-1)).squeeze(-1)

        # convert to expression matrix
        nfragments_effect = cellxgene_embedding.reshape((n_cells, n_genes))

        # 
        expression = nfragments_effect + self.bias1[gene_ix]
        return expression
    
class Model(torch.nn.Module):
    def __init__(
        self,
        mean_gene_expression,
        **kwargs
    ):
        super().__init__()
        self.fragments_to_expression = FragmentsToExpression(
            mean_gene_expression = mean_gene_expression
        )
        
    def forward(
        self,
        data
    ):
        expression = self.fragments_to_expression(
            cellxgene_ix = data.local_cellxgene_ix,
            n_cells = data.n_cells,
            n_genes = data.n_genes,
            gene_ix = data.genes_oi
        )
        return expression

    def get_parameters(self):
        return self.parameters()
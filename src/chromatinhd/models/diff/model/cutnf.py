import torch
import numpy as np
import math
import tqdm.auto as tqdm
import torch_scatter

from chromatinhd.embedding import EmbeddingTensor
from . import spline
from chromatinhd.models import HybridModel

from chromatinhd.flow import Flow, Stored

from chromatinhd.models.diff.loader.minibatches import Minibatcher
from chromatinhd.models.diff.loader.clustering_cuts import (
    ClusteringCuts,
)
from chromatinhd.models.diff.trainer import Trainer
from chromatinhd.loaders import LoaderPool2
from chromatinhd.optim import SparseDenseAdam

import pickle


class Decoder(torch.nn.Module):
    def __init__(
        self,
        n_latent,
        n_genes,
        n_output_components,
        n_layers=1,
        n_hidden_dimensions=32,
        dropout_rate=0.2,
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                torch.nn.Linear(
                    n_hidden_dimensions if i > 0 else n_latent, n_hidden_dimensions
                )
            )
            layers.append(torch.nn.BatchNorm1d(n_hidden_dimensions))
            layers.append(torch.nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(torch.nn.Dropout(0.2))
        self.nn = torch.nn.Sequential(*layers)

        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_output_components = n_output_components

        self.logit_weight = EmbeddingTensor(
            n_genes,
            (n_hidden_dimensions if n_layers > 0 else n_latent, n_output_components),
            sparse=True,
        )
        stdv = 1.0 / math.sqrt(self.logit_weight.weight.size(1))
        # self.logit_weight.weight.data.uniform_(-stdv, stdv)
        self.logit_weight.weight.data.zero_()

        self.rho_weight = EmbeddingTensor(
            n_genes, (n_hidden_dimensions if n_layers > 0 else n_latent,), sparse=True
        )
        stdv = 1.0 / math.sqrt(self.rho_weight.weight.size(1))
        self.rho_weight.weight.data.uniform_(-stdv, stdv)

    def forward(self, latent, genes_oi):
        # genes oi is only used to get the logits
        # we calculate the rho for all genes because we pool using softmax later
        logit_weight = self.logit_weight(genes_oi)
        rho_weight = self.rho_weight.get_full_weight()
        nn_output = self.nn(latent)

        # nn_output is broadcasted across genes and across components
        logit = torch.matmul(nn_output.unsqueeze(1).unsqueeze(2), logit_weight).squeeze(
            -2
        )

        # nn_output has to be broadcasted across genes
        rho = torch.matmul(nn_output.unsqueeze(1), rho_weight.T).squeeze(-2)

        return logit, rho

    def parameters_dense(self):
        return [*self.nn.parameters(), self.rho_weight.weight]

    def parameters_sparse(self):
        return [self.logit_weight.weight]


class BaselineDecoder(torch.nn.Module):
    def __init__(
        self, n_latent, n_genes, n_output_components, n_layers=1, n_hidden_dimensions=32
    ):
        super().__init__()

        layers = []
        for i in range(n_layers):
            layers.append(
                torch.nn.Linear(
                    n_hidden_dimensions if i > 0 else n_latent, n_hidden_dimensions
                )
            )
            layers.append(torch.nn.BatchNorm1d(n_hidden_dimensions))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(0.2))
        self.nn = torch.nn.Sequential(*layers)

        self.n_hidden_dimensions = n_hidden_dimensions
        self.n_output_components = n_output_components

        self.rho_weight = EmbeddingTensor(
            n_genes, (n_hidden_dimensions if n_layers > 0 else n_latent,), sparse=True
        )
        stdv = 1.0 / math.sqrt(self.rho_weight.weight.size(1)) / 100
        self.rho_weight.weight.data.uniform_(-stdv, stdv)

    def forward(self, latent, genes_oi):
        rho_weight = self.rho_weight(genes_oi)
        nn_output = self.nn(latent)

        # nn_output is broadcasted across genes and across components
        logit = torch.zeros(
            (latent.shape[0], len(genes_oi), self.n_output_components),
            device=latent.device,
        )

        rho = torch.matmul(nn_output, rho_weight.T).squeeze(-1)
        return logit, rho

    def parameters_dense(self):
        return [*self.nn.parameters()]

    def parameters_sparse(self):
        return [self.rho_weight.weight]


class Model(torch.nn.Module, HybridModel):
    def __init__(
        self,
        fragments,
        clustering,
        nbins=(
            128,
            64,
            32,
        ),
        decoder_n_layers=0,
        baseline=False,
        mixture_delta_p_scale_free=False,
        scale_likelihood=False,
        rho_delta_p_scale_free=False,
        mixture_delta_p_scale_dist="normal",
        mixture_delta_p_scale=1.0,
    ):
        super().__init__()

        self.n_total_genes = fragments.n_genes

        self.n_clusters = clustering.n_clusters

        from .spline import DifferentialQuadraticSplineStack, TransformedDistribution

        transform = DifferentialQuadraticSplineStack(
            nbins=nbins,
            n_genes=fragments.n_genes,
        )
        self.mixture = TransformedDistribution(transform)
        n_delta_mixture_components = sum(transform.split_deltas)

        if not baseline:
            self.decoder = Decoder(
                self.n_clusters,
                fragments.n_genes,
                n_delta_mixture_components,
                n_layers=decoder_n_layers,
            )
        else:
            self.decoder = BaselineDecoder(
                self.n_clusters,
                fragments.n_genes,
                n_delta_mixture_components,
                n_layers=decoder_n_layers,
            )

        libsize = torch.bincount(fragments.mapping[:, 0], minlength=fragments.n_cells)
        rho_bias = (
            torch.bincount(fragments.mapping[:, 1], minlength=fragments.n_genes)
            / fragments.n_cells
            / libsize.to(torch.float).mean()
        )
        min_rho_bias = 1e-5
        rho_bias = min_rho_bias + (1 - min_rho_bias) * rho_bias

        self.register_buffer("libsize", libsize)
        self.register_buffer("rho_bias", rho_bias)

        self.track = {}

        if mixture_delta_p_scale_free:
            self.mixture_delta_p_scale = torch.nn.Parameter(
                torch.tensor(math.log(mixture_delta_p_scale), requires_grad=True)
            )
        else:
            self.register_buffer(
                "mixture_delta_p_scale", torch.tensor(math.log(mixture_delta_p_scale))
            )

        self.n_total_cells = fragments.n_cells

        self.scale_likelihood = scale_likelihood

        if rho_delta_p_scale_free:
            self.rho_delta_p_scale = torch.nn.Parameter(
                torch.log(torch.tensor(0.1, requires_grad=True))
            )
        else:
            self.register_buffer("rho_delta_p_scale", torch.tensor(math.log(1.0)))

        self.mixture_delta_p_scale_dist = mixture_delta_p_scale_dist

    def forward_(
        self,
        coordinates,
        clustering,
        genes_oi,
        local_cellxgene_ix,
        localcellxgene_ix,
        local_gene_ix,
    ):
        # decode
        mixture_delta, rho_delta = self.decoder(clustering.to(torch.float), genes_oi)

        # rho
        rho = torch.nn.functional.softmax(torch.log(self.rho_bias) + rho_delta, -1)
        rho_cuts = rho.flatten()[localcellxgene_ix]

        # rho delta kl
        rho_delta_p = torch.distributions.Normal(0.0, torch.exp(self.rho_delta_p_scale))
        rho_delta_kl = rho_delta_p.log_prob(self.decoder.rho_weight(genes_oi))

        # fragment counts
        mixture_delta_cellxgene = mixture_delta.view(
            np.prod(mixture_delta.shape[:2]), mixture_delta.shape[-1]
        )
        mixture_delta = mixture_delta_cellxgene[local_cellxgene_ix]

        self.track["likelihood_mixture"] = likelihood_mixture = self.mixture.log_prob(
            coordinates, genes_oi, local_gene_ix, mixture_delta
        )

        self.track["likelihood_overall"] = likelihood_overall = torch.log(
            rho_cuts
        ) + math.log(self.n_total_genes)

        # overall likelihood
        likelihood = self.track["likelihood"] = likelihood_mixture + likelihood_overall
        likelihood_scale = 1.0

        # mixture kl
        mixture_delta_p = torch.distributions.Normal(
            0.0, torch.exp(self.mixture_delta_p_scale)
        )
        mixture_delta_kl = mixture_delta_p.log_prob(self.decoder.logit_weight(genes_oi))

        # ELBO
        elbo = (
            -likelihood.sum() * likelihood_scale
            - mixture_delta_kl.sum()
            - rho_delta_kl.sum()
        )

        return elbo

    def forward(self, data):
        return self.forward_(
            coordinates=data.cuts.coordinates,
            clustering=data.clustering.onehot,
            genes_oi=data.minibatch.genes_oi_torch,
            local_gene_ix=data.cuts.local_gene_ix,
            local_cellxgene_ix=data.cuts.local_cellxgene_ix,
            localcellxgene_ix=data.cuts.localcellxgene_ix,
        )

    def train_model(self, fragments, clustering, fold, device="cuda", n_epochs=30):
        # set up minibatchers and loaders
        minibatcher_train = Minibatcher(
            fold["cells_train"],
            range(fragments.n_genes),
            n_genes_step=500,
            n_cells_step=200,
        )
        minibatcher_validation = Minibatcher(
            fold["cells_validation"],
            range(fragments.n_genes),
            n_genes_step=10,
            n_cells_step=10000,
            permute_cells=False,
            permute_genes=False,
        )

        loaders_train = LoaderPool2(
            ClusteringCuts,
            dict(
                clustering=clustering,
                fragments=fragments,
                cellxgene_batch_size=minibatcher_train.cellxgene_batch_size,
            ),
            n_workers=10,
        )
        loaders_validation = LoaderPool2(
            ClusteringCuts,
            dict(
                clustering=clustering,
                fragments=fragments,
                cellxgene_batch_size=minibatcher_validation.cellxgene_batch_size,
            ),
            n_workers=5,
        )

        trainer = Trainer(
            self,
            loaders_train,
            loaders_validation,
            minibatcher_train,
            minibatcher_validation,
            SparseDenseAdam(
                self.parameters_sparse(),
                self.parameters_dense(),
                lr=1e-2,
                weight_decay=1e-5,
            ),
            n_epochs=n_epochs,
            checkpoint_every_epoch=1,
            optimize_every_step=1,
            device=device,
        )

        trainer.train()

    def _get_likelihood_cell_gene(
        self, likelihood, local_cellxgene_ix, n_cells, n_genes
    ):
        return torch_scatter.segment_sum_coo(
            likelihood, local_cellxgene_ix, dim_size=n_cells * n_genes
        ).reshape((n_cells, n_genes))

    def get_prediction(
        self,
        fragments,
        clustering,
        cells=None,
        cell_ixs=None,
        genes=None,
        gene_ixs=None,
        device="cuda",
        return_raw=False,
    ):
        """
        Returns the prediction of a dataset
        """
        if cell_ixs is None:
            if cells is None:
                cells = fragments.obs.index
            fragments.obs["ix"] = np.arange(len(fragments.obs))
            cell_ixs = fragments.obs.loc[cells]["ix"].values
        if cells is None:
            cells = fragments.obs.index[cell_ixs]

        if gene_ixs is None:
            if genes is None:
                genes = fragments.var.index
            fragments.var["ix"] = np.arange(len(fragments.var))
            gene_ixs = fragments.var.loc[genes]["ix"].values
        if genes is None:
            genes = fragments.var.index[gene_ixs]

        minibatches = Minibatcher(
            cell_ixs,
            gene_ixs,
            n_genes_step=500,
            n_cells_step=200,
            use_all_cells=True,
            use_all_genes=True,
            permute_cells=False,
            permute_genes=False,
        )
        loaders = LoaderPool2(
            ClusteringCuts,
            dict(
                clustering=clustering,
                fragments=fragments,
                cellxgene_batch_size=minibatches.cellxgene_batch_size,
            ),
            n_workers=5,
        )
        loaders.initialize(minibatches)

        likelihood_mixture = np.zeros((len(cell_ixs), len(gene_ixs)))
        likelihood_overall = np.zeros((len(cell_ixs), len(gene_ixs)))

        cell_mapping = np.zeros(fragments.n_cells, dtype=np.int64)
        cell_mapping[cell_ixs] = np.arange(len(cell_ixs))

        gene_mapping = np.zeros(fragments.n_genes, dtype=np.int64)
        gene_mapping[gene_ixs] = np.arange(len(gene_ixs))

        device = "cuda"
        self.eval()
        self = self.to(device)

        for data in loaders:
            data = data.to(device)
            with torch.no_grad():
                self.forward(data)

            likelihood_mixture[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    gene_mapping[data.minibatch.genes_oi],
                )
            ] += (
                self._get_likelihood_cell_gene(
                    self.track["likelihood_mixture"],
                    data.cuts.local_cellxgene_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_genes,
                )
                .cpu()
                .numpy()
            )
            likelihood_overall[
                np.ix_(
                    cell_mapping[data.minibatch.cells_oi],
                    gene_mapping[data.minibatch.genes_oi],
                )
            ] += (
                self._get_likelihood_cell_gene(
                    self.track["likelihood_overall"],
                    data.cuts.local_cellxgene_ix,
                    data.minibatch.n_cells,
                    data.minibatch.n_genes,
                )
                .cpu()
                .numpy()
            )

        self = self.to("cpu")

        if return_raw:
            return predicted, expected, n_fragments

        import xarray as xr

        result = xr.Dataset(
            {
                "likelihood_mixture": xr.DataArray(
                    likelihood_mixture,
                    dims=("cell", "gene"),
                    coords={"cell": cells, "gene": fragments.var.index},
                ),
                "likelihood_overall": xr.DataArray(
                    likelihood_overall,
                    dims=("cell", "gene"),
                    coords={"cell": cells, "gene": fragments.var.index},
                ),
            }
        )
        return result

    def evaluate_pseudo(self, coordinates, clustering=None, gene_oi=None, gene_ix=None):
        from chromatinhd.models.diff.loader.cuts import Result as CutsResult
        from chromatinhd.models.diff.loader.minibatches import Minibatch
        from chromatinhd.models.diff.loader.clustering import Result as ClusteringResult
        from chromatinhd.models.diff.loader.clustering_cuts import (
            Result as ClusteringCutsResult,
        )

        device = coordinates.device
        if not torch.is_tensor(clustering):
            if clustering is None:
                clustering = 0.0
            clustering = torch.ones((1, self.n_clusters), device=device) * clustering

            print(clustering)

        cells_oi = torch.ones((1,), dtype=torch.long)

        local_cellxgene_ix = torch.tensor([], dtype=torch.long)
        if gene_ix is None:
            if gene_oi is None:
                gene_oi = 0
            genes_oi = torch.tensor([gene_oi], device=device, dtype=torch.long)
            local_gene_ix = torch.zeros_like(coordinates).to(torch.long)
            local_cellxgene_ix = torch.zeros_like(coordinates).to(torch.long)
            localcellxgene_ix = torch.ones_like(coordinates).to(torch.long) * gene_oi
        else:
            assert len(gene_ix) == len(coordinates)
            genes_oi = torch.unique(gene_ix)

            local_gene_mapping = torch.zeros(
                genes_oi.max() + 1, device=device, dtype=torch.long
            )
            local_gene_mapping.index_add_(
                0, genes_oi, torch.arange(len(genes_oi), device=device)
            )

            local_gene_ix = local_gene_mapping[gene_ix]
            local_cell_ix = torch.arange(
                clustering.shape[0], device=local_gene_ix.device
            )
            local_cellxgene_ix = local_cell_ix * len(genes_oi) + local_gene_ix
            localcellxgene_ix = local_cell_ix * self.n_total_genes + gene_ix

        data = ClusteringCutsResult(
            cuts=CutsResult(
                coordinates=coordinates,
                local_cellxgene_ix=local_cellxgene_ix,
                localcellxgene_ix=localcellxgene_ix,
                n_genes=len(genes_oi),
            ),
            clustering=ClusteringResult(
                onehot=clustering,
            ),
            minibatch=Minibatch(
                cells_oi=cells_oi.numpy(),
                genes_oi=genes_oi.numpy(),
            ),
        )

        with torch.no_grad():
            self.forward(data)

        prob = self.track["likelihood"].detach().cpu()
        return prob.detach().cpu()

    def rank(
        self,
        window: np.ndarray,
        n_latent: int,
        device: torch.DeviceObjType = "cuda",
        how: str = "probs_diff_masked",
        prob_cutoff: float = None,
    ) -> np.ndarray:
        n_genes = self.rho_bias.shape[0]

        import pandas as pd
        from chromatinhd.utils import crossing

        self = self.to(device).eval()

        # create design for inference
        design_gene = pd.DataFrame({"gene_ix": np.arange(n_genes)})
        design_latent = pd.DataFrame({"active_latent": np.arange(n_latent)})
        design_coord = pd.DataFrame(
            {"coord": np.arange(window[0], window[1] + 1, step=25)}
        )
        design = crossing(design_gene, design_latent, design_coord)
        batch_size = 100000
        design["batch"] = np.floor(np.arange(design.shape[0]) / batch_size).astype(int)

        # infer
        probs = []
        for _, design_subset in tqdm.tqdm(design.groupby("batch")):
            pseudocoordinates = torch.from_numpy(design_subset["coord"].values).to(
                device
            )
            pseudocoordinates = (pseudocoordinates - window[0]) / (
                window[1] - window[0]
            )
            pseudolatent = torch.nn.functional.one_hot(
                torch.from_numpy(design_subset["active_latent"].values).to(device),
                n_latent,
            ).to(torch.float)
            gene_ix = torch.from_numpy(design_subset["gene_ix"].values).to(device)

            prob = self.evaluate_pseudo(
                pseudocoordinates.to(device),
                latent=pseudolatent.to(device),
                gene_ix=gene_ix,
            )

            probs.append(prob)
        probs = np.hstack(probs)

        probs = probs.reshape(
            (design_gene.shape[0], design_latent.shape[0], design_coord.shape[0])
        )

        # calculate the score we're gonna use: how much does the likelihood of a cut in a window change compared to the "mean"?
        probs_diff = probs - probs.mean(-2, keepdims=True)

        # apply a mask to regions with very low likelihood of a cut
        if prob_cutoff is None:
            prob_cutoff = np.log(1.0)
        mask = probs >= prob_cutoff

        probs_diff_masked = probs_diff.copy()
        probs_diff_masked[~mask] = -np.inf

        ## Single base-pair resolution
        # interpolate the scoring from above but now at single base pairs
        # we may have to smooth this in the future, particularly for very detailed models that already look at base pair resolution
        x = (design["coord"].values).reshape(
            (design_gene.shape[0], design_latent.shape[0], design_coord.shape[0])
        )

        def interpolate(
            x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor
        ) -> torch.Tensor:
            a = (fp[..., 1:] - fp[..., :-1]) / (xp[..., 1:] - xp[..., :-1])
            b = fp[..., :-1] - (a.mul(xp[..., :-1]))

            indices = (
                torch.searchsorted(xp.contiguous(), x.contiguous(), right=False) - 1
            )
            indices = torch.clamp(indices, 0, a.shape[-1] - 1)
            slope = a.index_select(a.ndim - 1, indices)
            intercept = b.index_select(a.ndim - 1, indices)
            return x * slope + intercept

        desired_x = torch.arange(*window)

        probs_diff_interpolated = interpolate(
            desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs_diff)
        ).numpy()
        probs_interpolated = interpolate(
            desired_x, torch.from_numpy(x)[0][0], torch.from_numpy(probs)
        ).numpy()

        if how == "probs_diff_masked":
            # again apply a mask
            probs_diff_interpolated_masked = probs_diff_interpolated.copy()
            mask_interpolated = probs_interpolated >= prob_cutoff
            probs_diff_interpolated_masked[~mask_interpolated] = -np.inf

            return probs_diff_interpolated_masked
        else:
            raise NotImplementedError()


class Models(Flow):
    n_models = Stored("n_models")

    @property
    def models_path(self):
        path = self.path / "models"
        path.mkdir(exist_ok=True)
        return path

    def train_models(
        self, fragments, clustering, folds, device="cuda", n_epochs=30, **kwargs
    ):
        self.n_models = len(folds)
        for fold_ix, fold in [(fold_ix, fold) for fold_ix, fold in enumerate(folds)]:
            desired_outputs = [self.models_path / ("model_" + str(fold_ix) + ".pkl")]
            force = False
            if not all([desired_output.exists() for desired_output in desired_outputs]):
                force = True

            if force:
                model = Model(fragments, clustering, **kwargs)
                model.train_model(
                    fragments, clustering, fold, device=device, n_epochs=n_epochs
                )

                model = model.to("cpu")

                pickle.dump(
                    model,
                    open(self.models_path / ("model_" + str(fold_ix) + ".pkl"), "wb"),
                )

    def __getitem__(self, ix):
        return pickle.load(
            (self.models_path / ("model_" + str(ix) + ".pkl")).open("rb")
        )

    def __len__(self):
        return self.n_models

    def __iter__(self):
        for ix in range(len(self)):
            yield self[ix]

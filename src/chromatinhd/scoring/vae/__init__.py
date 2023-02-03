import pandas as pd
import numpy as np
import torch
import tqdm.auto as tqdm

import chromatinhd as chd

import scanpy as sc


## OVERLAP
import faiss


def search(X, k):
    X = np.ascontiguousarray(X)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    D, I = index.search(X, k)
    return I


def convert(I):
    return I + (np.arange(I.shape[0])[:, None]) * I.shape[0]


def calculate_overlap(indices_A, indices_B):
    overlap = np.in1d(convert(indices_A), convert(indices_B)).reshape(
        indices_A.shape[0], indices_A.shape[1]
    )[:, 1:]
    return overlap


def score_overlap(overlap, ks=(50,)):
    scores = []
    for k in ks:
        jac = overlap[:, :k].mean()
        scores.append({"k": k, "overlap": jac})
    return pd.DataFrame(scores)


def score_overlap_phased(overlap, phases, ks=(50,)):
    scores = []
    scores_cells = []
    for k in ks:
        for phase, cells in phases.items():
            jac = overlap[cells, :k].mean()
            scores.append({"k": k, "overlap": jac, "phase": phase})

            jac_cells = overlap[cells, :k].mean(1)
            scores_cells.append(
                pd.DataFrame(
                    {"k": k, "overlap": jac_cells, "cell_ix": cells, "phase": phase}
                )
            )
    return pd.DataFrame(scores), pd.concat(scores_cells)


def score_overlap_phased_weighted(overlap, phases, weights, ks=(50,)):
    scores = []
    for k in ks:
        for phase, cells in phases.items():
            jac = (overlap[cells, :k].mean(1) * weights[cells]).sum() / weights[
                cells
            ].sum()
            scores.append({"k": k, "overlap": jac, "phase": phase})
    return pd.DataFrame(scores)


## ARI
import igraph as ig
import leidenalg
import sklearn.metrics


def partition(indices):
    edges = np.vstack(
        [np.repeat(np.arange(indices.shape[0]), indices.shape[1]), indices.flatten()]
    ).T

    graph = ig.Graph(edges=edges)
    partition = leidenalg.find_partition(
        graph, leidenalg.RBConfigurationVertexPartition, resolution_parameter=1.0
    )
    return np.array(partition.membership)


def score_ari_phased(clusters_B, indices_A, phases, ks=(50,)):
    scores = []
    for k in ks:
        cluster_A = partition(indices_A[:, 1 : (k + 1)])
        cluster_B = clusters_B[k]
        for phase, cells in phases.items():
            ari = sklearn.metrics.adjusted_rand_score(
                cluster_A[cells], cluster_B[cells]
            )
            ami = sklearn.metrics.adjusted_mutual_info_score(
                cluster_A[cells], cluster_B[cells]
            )

            scores.append({"k": k, "ari": ari, "ami": ami, "phase": phase})
    return pd.DataFrame(scores)


class Scorer:
    def __init__(
        self, models, transcriptome, folds, loaders, gene_ids, cell_ids, device="cuda"
    ):
        assert len(models) == len(folds)
        self.models = models
        self.device = device
        self.folds = folds
        self.loaders = loaders
        self.gene_ids = gene_ids
        self.cell_ids = cell_ids

        # indices transcriptomics
        B = transcriptome.adata.obsm["X_pca"].astype(np.float32)
        self.ks = (1, 2, 5, 10, 20, 50)
        self.K = max(self.ks) + 1
        self.indices_B = search(B, self.K)

        # cluster transcriptomics
        clusters_B = {}
        for k in self.ks:
            cluster_B = partition(self.indices_B[:, 1 : (k + 1)])
            clusters_B[k] = cluster_B
        self.clusters_B = clusters_B

        # stratification preprocessing
        sc.tl.leiden(transcriptome.adata, resolution=1.0)
        groups = transcriptome.adata.obs["leiden"]
        group_weights = 1 / groups.value_counts()
        cell_weights = group_weights[transcriptome.adata.obs["leiden"]].values
        self.cell_weights = cell_weights / cell_weights.sum()

    def score(self, loader_kwargs=None):
        """
        gene_ids: mapping of gene ix to gene id
        """

        embeddings = {}

        if loader_kwargs is None:
            loader_kwargs = {}

        next_task_sets = []
        for fold in self.folds:
            next_task_sets.append({"tasks": fold["minibatches"]})
        next_task_sets[0]["loader_kwargs"] = loader_kwargs
        self.loaders.initialize(next_task_sets=next_task_sets)

        scores_overlap = []
        scores_overlap_cells = []
        scores_ari = []
        scores_woverlap = []
        scores = []
        for model_ix, (model, fold) in tqdm.tqdm(
            enumerate(zip(self.models, self.folds)), leave=False, total=len(self.models)
        ):
            # get embedding
            embedding = np.zeros((len(self.cell_ids), model.n_latent_dimensions))
            embeddings[model_ix] = embedding

            model = model.to(self.device).eval()

            for data in self.loaders:
                data = data.to(self.device)
                with torch.no_grad():
                    latent = model.evaluate_latent(data)
                    if torch.is_tensor(latent):
                        latent = latent.detach().cpu().numpy()
                    embedding[data.cells_oi] = latent

                self.loaders.submit_next()

            # score
            A = embedding.astype(np.float32)

            indices_A = search(A, self.K)

            phases = {phase: cells for phase, (cells, genes) in fold["phases"].items()}

            overlap = calculate_overlap(indices_A, self.indices_B)

            scores_overlap_, scores_overlap_cells_ = score_overlap_phased(
                overlap, phases, ks=self.ks
            )
            scores_overlap_["model"] = model_ix
            scores_overlap_cells_["model"] = model_ix
            scores_ari_ = score_ari_phased(self.clusters_B, indices_A, phases, ks=(50,))
            scores_ari_["model"] = model_ix
            scores_woverlap_ = score_overlap_phased_weighted(
                overlap, phases, self.cell_weights, ks=self.ks
            )
            scores_woverlap_["model"] = model_ix

            scores_overlap.append(scores_overlap_)
            scores_overlap_cells.append(scores_overlap_cells_)
            scores_ari.append(scores_ari_)
            scores_woverlap.append(scores_woverlap_)

        scores_overlap = pd.concat(scores_overlap)
        scores_overlap_agg = scores_overlap.groupby(["phase"])[["overlap"]].mean()

        scores_woverlap = pd.concat(scores_woverlap)
        scores_woverlap_agg = scores_woverlap.groupby(["phase"])[["overlap"]].mean()
        scores_woverlap_agg.columns = ["woverlap"]

        scores_ari = pd.concat(scores_ari)
        scores_ari_agg = scores_ari.groupby(["phase"])[["ari", "ami"]].mean()

        scores_overlap_cells = pd.concat(scores_overlap_cells)
        scores_overlap_cells_agg = scores_overlap_cells.groupby(["phase", "cell_ix"])[
            ["overlap"]
        ].mean()

        scores = pd.concat(
            [
                scores_overlap_agg,
                scores_woverlap_agg,
                scores_ari_agg,
            ],
            axis=1,
        )

        scores_cells = scores_overlap_cells_agg

        return scores, scores_cells

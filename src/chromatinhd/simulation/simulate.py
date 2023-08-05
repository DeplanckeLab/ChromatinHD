import scipy.stats
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from chromatinhd.data.regions import Regions
from chromatinhd.data.fragments import Fragments
from chromatinhd.data.clustering import Clustering
import torch


def transform_nb2(mu, dispersion, eps=1e-8):
    if isinstance(mu, (float, int)):
        mu = np.array(mu)
    if isinstance(dispersion, (float, int)):
        dispersion = np.array(dispersion)
    # avoids NaNs induced in gradients when mu is very low
    dispersion = np.clip(dispersion, 0, 20.0)

    logits = np.log(mu + eps) - np.log(1 / dispersion + eps)

    total_count = 1 / dispersion

    # return total_count, logits
    return total_count, logits


def sample_nb(_shape, total_count, logits):
    return scipy.stats.poisson(np.random.standard_gamma(total_count, _shape) / np.exp(-logits)).rvs(_shape)


class Dist:
    def __init__(self, loc, scale, myclip_a, myclip_b):
        a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale
        self.dist = scipy.stats.truncnorm(loc=loc, scale=scale, a=a, b=b)

    def __call__(self, n):
        return np.round(self.dist.rvs(n))


def sample(w, dists):
    assert w.shape[-1] == len(dists)
    n = w.shape[0]
    if n == 0:
        raise ValueError
    samples = np.stack([dist(n) for dist in dists], 1)
    i = np.random.rand(n)
    indices = np.argmax(i[:, None] < np.cumsum(w, 1), 1)
    samples = samples[np.arange(n), indices]
    return samples


class Simulation:
    def __init__(self, n_cells=1000, n_genes=1000, window=(-5000, 5000)):
        self.window = np.array(window)
        self.n_cells = n_cells
        self.n_genes = n_genes

    def create_regions(self):
        self.regions = Regions.create(
            coordinates=pd.DataFrame(
                {
                    "chrom": np.repeat("chr1", self.n_genes),
                    "start": 0,
                    "end": (self.window[1] - self.window[0]),
                    "gene": ["G" + str(i) for i in np.arange(self.n_genes)],
                    "tss": 0,
                    "strand": 1,
                }
            ).set_index("gene"),
            window=self.window,
        )

    def create_obs(self):
        self.obs = pd.DataFrame(
            {
                "cell": ["C" + str(i) for i in np.arange(self.n_cells)],
                "celltype": ["Ct" + str(i) for i in np.random.choice(np.arange(5), self.n_cells)],
            }
        ).set_index("cell")

        self.celltypes = pd.DataFrame({"celltype": self.obs["celltype"].unique()}).set_index("celltype")

        self.clustering = Clustering.from_labels(self.obs["celltype"])

    def create_fragments(self):
        # peaks
        n_peaks = self.n_genes * 50
        self.peaks = pd.DataFrame(
            {
                "center": np.random.choice(np.arange(*self.window), n_peaks, replace=True),
                # "scale": 100,
                "scale": np.exp(np.random.normal(size=n_peaks, scale=5) + np.log(50)),
                "gene": np.random.choice(self.n_genes, n_peaks, replace=True),
                "height": np.random.normal(size=n_peaks, scale=0.1) + 1,
            }
        )
        self.peaks["ix"] = np.arange(self.peaks.shape[0])
        self.peaks["dist"] = [
            Dist(loc, scale, self.window[0], self.window[1])
            for loc, scale in zip(self.peaks["center"], self.peaks["scale"])
        ]

        # cell latent space
        self.cell_latent_space = np.stack(
            [(self.obs["celltype"] == celltype_).values.astype(float) for celltype_ in self.celltypes.index],
        ).T
        self.n_cell_components = self.cell_latent_space.shape[-1]
        w_coef = np.random.normal(size=(n_peaks, self.n_cell_components)) * 2

        # lib
        sensitivity = 10.0
        library_size = np.exp(np.random.normal(np.log(self.n_genes * sensitivity), 0.2, self.n_cells))

        # gene average
        gene_average = scipy.stats.norm(0, 1).rvs(self.n_genes)
        gene_average = np.exp(gene_average) / np.exp(gene_average).sum()

        # fragments
        coordinates = []
        mapping = []
        for gene_ix in tqdm.tqdm(range(self.n_genes)):
            n_fragments_gene_mean = library_size * gene_average[gene_ix]

            n_fragments_gene = sample_nb(self.n_cells, *transform_nb2(n_fragments_gene_mean, 1.0))
            cell_indices_gene = np.repeat(np.arange(self.n_cells), n_fragments_gene)

            peaks_oi = self.peaks.loc[self.peaks["gene"] == gene_ix]

            w = peaks_oi["height"].values + (self.cell_latent_space[cell_indices_gene] @ w_coef[peaks_oi["ix"]].T)
            w = np.exp(w) / np.exp(w).sum(1, keepdims=True)

            coordinates1 = sample(w, peaks_oi["dist"].values)

            # sample coordinates 2
            rate = 0.1

            def distance_weight(dist):
                return rate * np.exp(-rate * dist)

            w2 = w * distance_weight(np.abs(coordinates1[:, None] - peaks_oi["center"].values[None, :]))

            coordinates2 = sample(w2, peaks_oi["dist"].values)

            # coordinates
            coordinates_gene = np.stack(
                [
                    np.minimum(coordinates1, coordinates2),
                    np.maximum(coordinates2, coordinates1),
                ]
            )
            coordinates.append(coordinates_gene)

            mapping_gene = np.stack([cell_indices_gene, np.repeat(gene_ix, n_fragments_gene.sum())])
            mapping.append(mapping_gene)

        # combine coordinates
        self.coordinates = torch.from_numpy(np.hstack(coordinates).T)
        self.mapping = torch.from_numpy(np.hstack(mapping).T)

        var = self.regions.coordinates.copy()

        # Sort `coordinates` and `mapping` according to `mapping`
        sorted_idx = torch.argsort((self.mapping[:, 0] * var.shape[0] + self.mapping[:, 1]))
        self.mapping = self.mapping[sorted_idx]
        self.coordinates = self.coordinates[sorted_idx]

        # create fragments
        self.fragments = Fragments.create(
            coordinates=self.coordinates,
            mapping=self.mapping,
            regions=self.regions,
            obs=self.obs,
            var=var,
        )

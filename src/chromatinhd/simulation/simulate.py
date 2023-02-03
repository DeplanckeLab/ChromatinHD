import scipy.stats
import numpy as np
import pandas as pd
import tqdm.auto as tqdm


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
    return scipy.stats.poisson(
        np.random.standard_gamma(total_count, _shape) / np.exp(-logits)
    ).rvs(_shape)


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

        # peaks
        n_peaks = self.n_genes * 20
        self.peaks = pd.DataFrame(
            {
                "center": np.random.choice(
                    np.arange(*self.window), n_peaks, replace=True
                ),
                "scale": 100,
                # "scale": np.exp(np.random.normal(size = n_peaks) + np.log(30)),
                "gene": np.random.choice(self.n_genes, n_peaks, replace=True),
            }
        )
        self.peaks["ix"] = np.arange(self.peaks.shape[0])
        self.peaks["dist"] = [
            Dist(loc, scale, self.window[0], self.window[1])
            for loc, scale in zip(self.peaks["center"], self.peaks["scale"])
        ]

        # cell latent space
        celltype = np.random.choice([0, 1, 2], replace=True, size=(self.n_cells, 1))

        latent_0 = np.random.uniform(-1, 1, size=(self.n_cells, 1))
        self.cell_latent_space = np.hstack(
            [
                # latent_0,
                # latent_0**2,
                # np.random.uniform(-1, 1, size=(self.n_cells, 1))
                # * (celltype == 0).astype(float),
                # np.random.uniform(-1, 1, size=(self.n_cells, 1))
                # * (celltype == 1).astype(float),
                (celltype == 0).astype(float),
                (celltype == 1).astype(float),
                (celltype == 2).astype(float),
            ]
        )
        self.n_cell_components = self.cell_latent_space.shape[-1]
        w_coef = np.random.normal(size=(n_peaks, self.n_cell_components)) * 2

        # lib
        sensitivity = 10.0
        library_size = np.exp(
            np.random.normal(np.log(self.n_genes * sensitivity), 0.2, self.n_cells)
        )

        # gene average
        gene_average = scipy.stats.norm(0, 1).rvs(self.n_genes)
        gene_average = np.exp(gene_average) / np.exp(gene_average).sum()

        # fragments
        coordinates = []
        mapping = []
        for gene_ix in tqdm.tqdm(range(self.n_genes)):
            n_fragments_gene_mean = library_size * gene_average[gene_ix]

            n_fragments_gene = sample_nb(
                self.n_cells, *transform_nb2(n_fragments_gene_mean, 1.0)
            )
            cell_indices_gene = np.repeat(np.arange(self.n_cells), n_fragments_gene)

            peaks_oi = self.peaks.loc[self.peaks["gene"] == gene_ix]

            w = np.random.normal(size=peaks_oi.shape[0]) + (
                self.cell_latent_space[cell_indices_gene] @ w_coef[peaks_oi["ix"]].T
            )
            w = np.exp(w) / np.exp(w).sum(1, keepdims=True)

            coordinates1 = sample(w, peaks_oi["dist"].values)

            # sample coordinates 2
            rate = 0.1
            distance_weight = lambda dist: rate * np.exp(-rate * dist)
            w2 = w * distance_weight(
                np.abs(coordinates1[:, None] - peaks_oi["center"].values[None, :])
            )

            coordinates2 = sample(w2, peaks_oi["dist"].values)

            # coordinates
            coordinates_gene = np.stack(
                [
                    np.minimum(coordinates1, coordinates2),
                    np.maximum(coordinates2, coordinates1),
                ]
            )
            coordinates.append(coordinates_gene)

            mapping_gene = np.stack(
                [cell_indices_gene, np.repeat(gene_ix, n_fragments_gene.sum())]
            )
            mapping.append(mapping_gene)

        # combine coordinates
        self.coordinates = np.hstack(coordinates).T
        self.mapping = np.hstack(mapping).T

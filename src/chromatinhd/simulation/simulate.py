import scipy.stats
import numpy as np
import pandas as pd
import tqdm.auto as tqdm
from chromatinhd.data.regions import Regions
from chromatinhd.data.fragments import Fragments
from chromatinhd.data.clustering import Clustering
from chromatinhd.data.motifscan import Motifscan
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
    return samples, indices


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

    def create_fragments(self, mean_fragmentprob_size=200, n_peaks_per_gene=5, min_peaks_per_gene=3):
        # peaks
        n_peaks = self.n_genes * n_peaks_per_gene
        self.peaks = pd.DataFrame(
            {
                "center": np.random.choice(np.arange(*self.window), n_peaks, replace=True),
                # "center": np.random.choice([-5000, -6000], n_peaks, replace=True),
                # "scale": 100,
                "scale": np.random.choice([10, 100, 500], replace=True, size=n_peaks),
                "gene": list(np.repeat(np.arange(self.n_genes), min_peaks_per_gene))
                + list(np.random.choice(self.n_genes, size=n_peaks - self.n_genes * min_peaks_per_gene, replace=True)),
                "height": np.random.normal(size=n_peaks, scale=0.1) + 1,
            }
        )
        self.peaks["size_mean"] = np.random.choice([10, 100, 200, 300], replace=True, size=n_peaks)
        self.peaks["size_scale"] = np.random.choice([20, 50], replace=True, size=n_peaks)
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

        # calculate coefficients
        self.peak_celltype_weights = np.random.normal(size=(n_peaks, self.n_cell_components)) * 0.5

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

            w = peaks_oi["height"].values + (
                self.cell_latent_space[cell_indices_gene] @ self.peak_celltype_weights[peaks_oi["ix"]].T
            )
            w = np.exp(w) / np.exp(w).sum(1, keepdims=True)

            coordinate_mid, peak_indices = sample(w, peaks_oi["dist"].values)

            # sample coordinates 2
            size_mean = peaks_oi["size_mean"].iloc[peak_indices]
            size_scale = peaks_oi["size_scale"].iloc[peak_indices]
            # size_mean = np.take_along_axis(
            #     np.exp(self.cell_latent_space[cell_indices_gene] @ size_logscale_coef[peaks_oi["ix"]].T),
            #     peak_indices[:, None],
            #     1,
            # ).squeeze()

            def lognorm_loc(desired_loc, desired_scale):
                return np.log(desired_loc) - 0.5 * np.log(1 + desired_scale**2 / desired_loc**2)

            def lognorm_scale(desired_loc, desired_scale):
                return np.sqrt(np.log(1 + desired_scale**2 / desired_loc**2))

            size = np.exp(
                np.random.normal(
                    size=coordinate_mid.shape,
                    loc=lognorm_loc(size_mean, size_scale),
                    scale=lognorm_scale(size_mean, size_scale),
                )
            )
            size = np.clip(size, 1, 1000)
            coordinates1 = coordinate_mid - size / 2
            coordinates2 = coordinate_mid + size / 2
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

        # filter on outside of window
        inside_window = (self.coordinates[:, 0] >= self.window[0]) & (self.coordinates[:, 1] <= self.window[1])
        self.coordinates = self.coordinates[inside_window]
        self.mapping = self.mapping[inside_window]

        # create fragments
        self.fragments = Fragments.create(
            coordinates=self.coordinates,
            mapping=self.mapping,
            regions=self.regions,
            obs=self.obs,
            var=var,
        )

    def create_motifscan(self, n_motifs=10, n_sites_per_gene=10):
        motifs = pd.DataFrame(
            {
                "motif": ["M" + str(i) for i in np.arange(n_motifs)],
            }
        ).set_index("motif")

        motif_celltype_weights = np.random.choice([0, 1], size=(n_motifs, self.n_cell_components))

        positions = []
        indices = []
        strands = []
        scores = []

        for gene_ix in tqdm.tqdm(range(self.n_genes)):
            n_sites_gene = n_motifs * n_sites_per_gene

            # n_fragments_gene = sample_nb(self.n_cells, *transform_nb2(n_fragments_gene_mean, 1.0))
            # cell_indices_gene = np.repeat(np.arange(self.n_cells), n_fragments_gene)

            peaks_oi = self.peaks.loc[self.peaks["gene"] == gene_ix]

            motif_peak_oi_weight = peaks_oi["height"].values + (
                motif_celltype_weights @ self.peak_celltype_weights[peaks_oi["ix"]].T
            )
            motif_peak_oi_weight = np.exp(motif_peak_oi_weight) / np.exp(motif_peak_oi_weight).sum(1, keepdims=True)

            # first sample the chosen indices
            motif_indices_gene = np.random.choice(np.arange(n_motifs), size=n_sites_gene, replace=True)

            site_peak_oi_weights = motif_peak_oi_weight[motif_indices_gene]

            positions_gene, peak_indices = sample(site_peak_oi_weights, peaks_oi["dist"].values)

            positions.append(positions_gene.astype(int) - self.window[0] + (self.window[1] - self.window[0]) * gene_ix)
            indices.append(motif_indices_gene)
            strands.append(np.random.choice([-1, 1], size=positions_gene.shape))
            scores.append(np.random.normal(size=positions_gene.shape))

        self.motifscan = Motifscan.from_positions(
            regions=self.regions,
            motifs=motifs,
            positions=np.hstack(positions),
            indices=np.hstack(indices),
            strands=np.hstack(strands),
            scores=np.hstack(scores),
        )

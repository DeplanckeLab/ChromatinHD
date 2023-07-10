import numpy as np
import scipy.signal
import pandas as pd


class DifferentialSlices:
    def __init__(
        self, positions, gene_ixs, cluster_ixs, window, n_genes, n_clusters, scores=None
    ):
        assert positions.ndim == 2
        assert positions.shape[1] == 2
        self.positions = positions
        self.gene_ixs = gene_ixs
        self.cluster_ixs = cluster_ixs
        self.window = window
        self.n_genes = n_genes
        self.n_clusters = n_clusters
        self.scores = scores

    def get_slicescores(self):
        slicescores = pd.DataFrame(
            {
                "start": self.positions[:, 0],
                "end": self.positions[:, 1],
                "length": (self.positions[:, 1] - self.positions[:, 0]),
                "gene_ix": self.gene_ixs,
                "cluster_ix": self.cluster_ixs,
                "mid": self.positions[:, 0]
                + (self.positions[:, 1] - self.positions[:, 0]) / 2
                + self.window[0],
            }
        )
        slicescores.index = np.arange(len(slicescores))
        return slicescores

    def get_slicelocations(self, promoters):
        slicelocations = self.get_slicescores().join(
            promoters[["start", "end", "strand", "chr", "gene_ix"]].set_index(
                "gene_ix"
            ),
            on="gene_ix",
            rsuffix="_promoter",
        )

        slicelocations["start_genome"] = np.where(
            slicelocations["strand"] == 1,
            slicelocations["start_promoter"] + slicelocations["start"],
            slicelocations["end_promoter"] - slicelocations["end"],
        )
        slicelocations["end_genome"] = np.where(
            slicelocations["strand"] == 1,
            slicelocations["start_promoter"] + slicelocations["end"],
            slicelocations["end_promoter"] - slicelocations["start"],
        )
        return slicelocations

    def get_randomslicelocations(self, promoters, n_random=10):
        slicescores = self.get_slicescores().copy()
        gene_ixs = np.random.choice(
            len(promoters), len(slicescores) * n_random, replace=True
        )
        slicescores = slicescores.iloc[np.repeat(np.arange(len(slicescores)), n_random)]
        slicescores["gene_ix"] = gene_ixs
        slicelocations = slicescores.join(
            promoters[["start", "end", "strand", "chr", "gene_ix"]].set_index(
                "gene_ix"
            ),
            on="gene_ix",
            rsuffix="_promoter",
        )

        slicelocations["start_genome"] = np.where(
            slicelocations["strand"] == 1,
            slicelocations["start_promoter"] + slicelocations["start"],
            slicelocations["end_promoter"] - slicelocations["end"],
        )
        slicelocations["end_genome"] = np.where(
            slicelocations["strand"] == 1,
            slicelocations["start_promoter"] + slicelocations["end"],
            slicelocations["end_promoter"] - slicelocations["start"],
        )
        return slicelocations

    def get_sliceaverages(self, probs):
        probs_baseline = probs.mean(1)
        slice_average_baseline = []
        slice_max_baseline = []
        slice_std_baseline = []
        slice_average_oi = []
        slice_max_oi = []
        slice_std_oi = []
        slice_average_lfc = []
        slice_max_lfc = []
        slice_summit = []
        for start, end, gene_ix, cluster_ix in zip(
            self.positions[:, 0], self.positions[:, 1], self.gene_ixs, self.cluster_ixs
        ):
            prob_oi = probs[
                gene_ix,
                cluster_ix,
                (start):(end),
            ]

            prob_baseline = probs_baseline[
                gene_ix,
                (start):(end),
            ]
            slice_average_oi.append(prob_oi.mean())
            slice_max_oi.append(prob_oi.max())
            slice_std_oi.append(prob_oi.std())

            slice_average_baseline.append(prob_baseline.mean())
            slice_max_baseline.append(prob_baseline.max())
            slice_std_baseline.append(prob_baseline.std())

            slice_lfc = prob_oi - prob_baseline

            slice_average_lfc.append(slice_lfc.mean())
            slice_max_lfc.append(slice_lfc.max())

            slice_summit.append(start + np.argmax(prob_oi))

        slice_average_baseline = np.hstack(slice_average_baseline)
        slice_max_baseline = np.hstack(slice_max_baseline)
        slice_std_baseline = np.hstack(slice_std_baseline)

        slice_average_oi = np.hstack(slice_average_oi)
        slice_max_oi = np.hstack(slice_max_oi)
        slice_std_oi = np.hstack(slice_std_oi)

        slice_average_lfc = np.hstack(slice_average_lfc)
        slice_max_lfc = np.hstack(slice_max_lfc)

        slice_summit = np.hstack(slice_summit)

        return pd.DataFrame(
            {
                "average_baseline": slice_average_baseline,
                "max_baseline": slice_max_baseline,
                "std_baseline": slice_std_baseline,
                "average": slice_average_oi,
                "max": slice_max_oi,
                "std": slice_std_oi,
                "average_lfc": slice_average_lfc,
                "max_lfc": slice_max_lfc,
                "summit": slice_summit,
            }
        )

    @property
    def position_chosen(self):
        position_chosen = np.zeros(
            (self.n_genes, self.n_clusters, (self.window[1] - self.window[0])),
            # self.n_clusters * self.n_genes * (self.window[1] - self.window[0]),
            dtype=bool,
        )
        for start, end, gene_ix, cluster_ix in zip(
            self.positions[:, 0], self.positions[:, 1], self.gene_ixs, self.cluster_ixs
        ):
            position_chosen[gene_ix, cluster_ix, start:end] = True
        return position_chosen.flatten()

    @property
    def position_ranked(self):
        position_chosen = np.zeros(
            (self.n_genes, self.n_clusters, (self.window[1] - self.window[0])),
            # self.n_clusters * self.n_genes * (self.window[1] - self.window[0]),
        )
        for start, end, gene_ix, cluster_ix, score in zip(
            self.positions[:, 0],
            self.positions[:, 1],
            self.gene_ixs,
            self.cluster_ixs,
            self.scores,
        ):
            position_chosen[gene_ix, cluster_ix, start:end] = score
        return position_chosen

    @property
    def position_indices(self):
        position_chosen = self.position_chosen
        position_indices = np.where(position_chosen)[0]
        return position_indices

    @classmethod
    def from_positions(
        cls,
        positions,
        gene_ixs,
        cluster_ixs,
        window,
        n_genes,
        n_clusters,
        resolution=1,
    ):
        groups = np.hstack(
            [
                0,
                np.cumsum(
                    (
                        np.diff(
                            cluster_ixs * gene_ixs * ((window[1] - window[0]) + 1)
                            + positions
                        )
                        != 1
                    )
                ),
            ]
        )
        cuts = np.where(np.hstack([True, (np.diff(groups) != 0), True]))[0]

        position_slices = (
            np.vstack((positions[cuts[:-1]], positions[cuts[1:] - 1] + 1)).T
            * resolution
        )
        gene_ixs = gene_ixs[cuts[:-1]]
        cluster_ixs = cluster_ixs[cuts[:-1]]
        return cls(position_slices, gene_ixs, cluster_ixs, window, n_genes, n_clusters)

    @classmethod
    def from_peakscores(
        cls,
        peakscores,
        window,
        n_genes,
        logfoldchanges_cutoff=1.0,
        pvals_adj_cutoff=0.05,
    ):

        # get significant
        peakscores["significant"] = (
            peakscores["logfoldchanges"] > logfoldchanges_cutoff
        ) & (peakscores["pvals_adj"] < pvals_adj_cutoff)

        # join peaks that overlap or are adjacent in any way
        # we create an adjacency matrix with the number of shared nucleotides as weight
        # we then check for connected components and merge significant peaks

        import scipy.sparse.csgraph

        peakscores_significant_joined = []
        for (cluster, gene_ix), peakscores_oi in peakscores.query(
            "significant"
        ).groupby(["cluster", "gene_ix"]):
            # only check for overlaps if there are more than 2 signficant peaks
            if len(peakscores_oi) > 1:
                # calculate peak-wise membership within the gene window
                membership = np.zeros((len(peakscores_oi), window[1] - window[0]))
                for i, start, end in zip(
                    np.arange(len(peakscores_oi)),
                    peakscores_oi.relative_start,
                    peakscores_oi.relative_end,
                ):
                    membership[i, (start - window[0]) : ((end - window[0]) + 1)] = 1

                # calculate adjacency
                adjacency = np.matmul(membership, membership.T)

                # get connected components
                connected_components = scipy.sparse.csgraph.connected_components(
                    adjacency, directed=False
                )

                # merge
                peakscores_oi["component"] = connected_components[1]
                peakscores_oi_joined = peakscores_oi.groupby("component").agg(
                    {
                        "relative_start": min,
                        "relative_end": max,
                        "logfoldchanges": "mean",
                    }
                )
                peakscores_oi_joined = peakscores_oi_joined.assign(
                    cluster=cluster, gene_ix=gene_ix
                )
            else:
                peakscores_oi_joined = peakscores_oi.reset_index()[
                    [
                        "gene_ix",
                        "cluster",
                        "relative_start",
                        "relative_end",
                        "logfoldchanges",
                    ]
                ].copy()
            peakscores_significant_joined.append(peakscores_oi_joined)

        peakscores_significant = pd.concat(
            peakscores_significant_joined, ignore_index=True
        )
        peakscores_significant["cluster"] = pd.Categorical(
            peakscores_significant["cluster"],
            categories=peakscores["cluster"].cat.categories,
        )

        # get positions
        positions = peakscores_significant[["relative_start", "relative_end"]].values
        scores = peakscores_significant["logfoldchanges"].values
        positions = positions - window[0]
        gene_ixs = peakscores_significant["gene_ix"].values
        cluster_ixs = peakscores_significant["cluster"].cat.codes

        return cls(
            positions,
            gene_ixs,
            cluster_ixs,
            window,
            n_genes,
            scores=scores,
            n_clusters=len(peakscores["cluster"].cat.categories),
        )

    @classmethod
    def from_basepair_ranking(cls, basepair_ranking, window, cutoff, resolution=1):
        """
        :param: cutoff

        """
        n_genes = basepair_ranking.shape[0]
        n_clusters = basepair_ranking.shape[1]

        basepairs_oi = basepair_ranking > cutoff

        gene_ixs, cluster_ixs, positions = np.where(basepairs_oi)

        return cls.from_positions(
            positions,
            gene_ixs,
            cluster_ixs,
            window,
            n_genes,
            n_clusters,
            resolution=resolution,
        )

    def get_slicetopologies(self, probs):
        probs_mean = probs.mean(1)
        import itertools

        prominences = []
        n_subpeaks = []
        balances = []
        balances_raw = []
        dominances = []
        shadows = []
        differentialdominances = []
        for gene_ix, cluster_ix in itertools.product(
            range(self.n_genes), range(self.n_clusters)
        ):
            slices_oi = (self.gene_ixs == gene_ix) & (self.cluster_ixs == cluster_ix)
            position_slices_gene = self.positions[slices_oi]
            probs_gene = probs[gene_ix, cluster_ix]
            probs_mean_gene = probs_mean[gene_ix]
            x = np.exp(probs_gene)
            peaks = []
            for slice in position_slices_gene:
                peaks.append(slice[0] + np.argmax(x[slice[0] : (slice[1] + 1)]))

                # determine number of supbeaks
                height_threshold = 0.25
                distance_threshold = 80
                x_ = x[slice[0] : (slice[1] + 1)]
                y_ = np.exp(probs_mean_gene[slice[0] : (slice[1] + 1)])
                subpeaks, subpeaks_info = scipy.signal.find_peaks(
                    x_, height=x_.max() * height_threshold, distance=distance_threshold
                )
                n_subpeaks.append(len(subpeaks))

                # determine local neighbourhood
                distance = 800

                neighborhoud_left = [
                    np.clip(slice[0] - distance, *(self.window - self.window[0])),
                    slice[0],
                ]
                neighborhoud_right = [
                    slice[1],
                    np.clip(slice[1] + distance, *(self.window - self.window[0])),
                ]

                x_neighborhoud = np.exp(
                    np.hstack(
                        [
                            probs_gene[
                                neighborhoud_left[0] : (neighborhoud_left[1] + 1)
                            ],
                            probs_gene[
                                neighborhoud_right[0] : (neighborhoud_right[1] + 1),
                            ],
                        ]
                    )
                )
                y_neighborhoud = np.exp(
                    np.hstack(
                        [
                            probs_mean_gene[
                                neighborhoud_left[0] : (neighborhoud_left[1] + 1)
                            ],
                            probs_mean_gene[
                                neighborhoud_right[0] : (neighborhoud_right[1] + 1)
                            ],
                        ]
                    )
                )

                dominances.append((x_.max() / max(x_.max(), x_neighborhoud.max())))

                differentialdominances.append(
                    (x_ / y_).max()
                    / max((x_ / y_).max(), (x_neighborhoud / y_neighborhoud).max())
                )

                # determine local "balancing"
                balances.append(
                    np.clip(y_neighborhoud - x_neighborhoud, 0, np.inf).sum()
                    / (x_ - y_).sum()
                )
                balances_raw.append((x_neighborhoud / y_neighborhoud).min())

                # determine local shadow
                shadows.append(x_.mean() / x_.max())

            peaks = np.array(peaks, dtype=int)

            # calculate prominence
            # we have to pad the left and right side because peaks in the margins do not get the correct prominence
            prominence, left_bases, right_bases = scipy.signal.peak_prominences(
                np.hstack([0, x, 0]), peaks + 1
            )
            relative_prominence = prominence / x[peaks]
            prominences.append(relative_prominence)

        prominences = np.hstack(prominences)
        shadows = np.hstack(shadows)

        return pd.DataFrame(
            {
                "prominence": prominences,
                "n_subpeaks": n_subpeaks,
                "log1p_n_subpeaks": n_subpeaks,
                "balances_raw": balances_raw,
                "balance": balances,
                "dominance": dominances,
                "differentialdominance": differentialdominances,
                "shadow": shadows,
            }
        )

    def get_slicedifferential(self, probs):
        probs_diff = probs - probs.mean(1, keepdims=True)
        import itertools

        slice_differential = []
        for start, end, gene_ix, cluster_ix in zip(
            self.positions[:, 0], self.positions[:, 1], self.gene_ixs, self.cluster_ixs
        ):
            probs_diff_oi = probs_diff[
                gene_ix,
                cluster_ix,
                (start):(end),
            ]
            slice_differential.append((probs_diff_oi > np.log(2)).mean())
        slice_differential = np.hstack(slice_differential)

        return pd.DataFrame(
            {
                "differential_positions": slice_differential,
            }
        )

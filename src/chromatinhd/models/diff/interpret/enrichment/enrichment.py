import scipy.stats
import tqdm.auto as tqdm
import numpy as np
import pandas as pd
from chromatinhd.utils import fdr


def enrichment_foreground_vs_background(slicescores_foreground, slicescores_background, slicecounts, motifs=None):
    if motifs is None:
        motifs = slicecounts.columns

    x_foreground = slicecounts.loc[slicescores_foreground["slice"], motifs].sum(0)
    x_background = slicecounts.loc[slicescores_background["slice"], motifs].sum(0)
    n_foreground = slicescores_foreground["length"].sum()
    n_background = slicescores_background["length"].sum()

    contingencies = (
        np.stack(
            [
                n_background - x_background,
                x_background,
                n_foreground - x_foreground,
                x_foreground,
            ],
            axis=1,
        )
        .reshape(-1, 2, 2)
        .astype(np.int64)
    )

    odds = (contingencies[:, 1, 1] * contingencies[:, 0, 0] + 1) / (contingencies[:, 1, 0] * contingencies[:, 0, 1] + 1)

    p_values = np.array(
        [
            scipy.stats.chi2_contingency(c).pvalue if (c > 5).all() else scipy.stats.fisher_exact(c).pvalue
            for c in contingencies
        ]
    )
    q_values = fdr(p_values)

    return pd.DataFrame(
        {
            "odds": odds,
            "p_value": p_values,
            "q_value": q_values,
            "motif": motifs,
            "contingency": [c for c in contingencies],
        }
    ).set_index("motif")


def enrichment_cluster_vs_clusters(slicescores, slicecounts, clusters=None, motifs=None, pbar=True):
    if clusters is None:
        clusters = slicescores["cluster"].cat.categories
    enrichment = []

    progress = clusters
    if pbar:
        progress = tqdm.tqdm(progress)
    for cluster in progress:
        selected_slices = slicescores["cluster"] == cluster
        slicescores_foreground = slicescores.loc[selected_slices]
        slicescores_background = slicescores.loc[~selected_slices]

        enrichment.append(
            enrichment_foreground_vs_background(
                slicescores_foreground,
                slicescores_background,
                slicecounts,
                motifs=motifs,
            )
            .assign(cluster=cluster)
            .reset_index()
        )

    enrichment = pd.concat(enrichment, axis=0).set_index(["cluster", "motif"])
    return enrichment
